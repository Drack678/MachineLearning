"""Medical imaging data handling, dermatoscopic preprocessing, leakage-free splits, and DICOM PHI audit.

Implements Phase 1 of Workshop 6 (HAM10000 / ISIC 2018 Task 3):
- Reproducible download from Harvard Dataverse (doi:10.7910/DVN/DBW86T) with MD5 & SHA-256 verification.
- Patient/lesion-grouped 70/15/15 splits (`GroupShuffleSplit` by `lesion_id`) with zero leakage checks.
- Modality-specific dermoscopy preprocessing: circular vignette cropping, DullRazor morphological hair
  removal (Black-Hat + Telea inpainting), and Shades of Gray color constancy (Minkowski p=6).
- Clinically valid dermoscopy data augmentations (dihedral group D4, bounded brightness/contrast, no hue shift).
- Class imbalance utilities: Class-Balanced weights, Focal Loss, and train-only WeightedRandomSampler.
- Clinical DICOM interoperability (`pydicom 3.0.2`): strict technical-tag allowlist, Modality/VOI LUT
  handling, and recursive HIPAA Safe Harbor PHI cleanliness audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import requests
import torch
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import ExplicitVRLittleEndian, VLPhotographicImageStorage, generate_uid
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.device import seed_everything  # noqa: E402
from Workshop_6.src.config import (  # noqa: E402
    AGE_BUCKETS,
    CHECKSUMS_FILE,
    CLASS_NAMES,
    DATASET_META_FILE,
    DATAVERSE_BASE_URL,
    DATAVERSE_DATASET_URL,
    DATAVERSE_FILES,
    DEFAULT_IMAGE_SIZE,
    DX_TO_ID,
    FIGURES_DIR,
    HAM10000_DOI,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LOCALIZATION_GROUPS,
    MALIGNANT_CLASSES,
    NUM_CLASSES,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    SEED,
    SPLITS_FILE,
    to_relative_path,
)

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Workshop6-MedicalAI-Research/1.0"}

# Strict allowlist of non-PHI technical tags for clinical DICOM inspection (pydicom skill guideline)
DICOM_TECHNICAL_ALLOWLIST: tuple[str, ...] = (
    "SOPClassUID",
    "Modality",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "NumberOfFrames",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
)

# Representative HIPAA Safe Harbor direct/quasi-identifier DICOM keywords to flag during PHI audit
HIPAA_PHI_DICOM_KEYWORDS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "DeviceSerialNumber",
)


# ---------------------------------------------------------------------------
# Hashing, Download & Extraction
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file_stream(
    url: str,
    dest: Path,
    *,
    expected_md5: str | None = None,
    timeout: int = 120,
    force: bool = False,
) -> Path:
    """Download a file from Harvard Dataverse with MD5 integrity verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if expected_md5 is None or compute_file_hash(dest, "md5") == expected_md5:
            return dest

    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)

    if expected_md5 is not None:
        actual_md5 = compute_file_hash(tmp_path, "md5")
        if actual_md5 != expected_md5:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual_md5}"
            )
    tmp_path.replace(dest)
    return dest


def download_ham10000(
    raw_dir: Path = RAW_DATA_DIR,
    *,
    include_masks: bool = True,
    force: bool = False,
) -> dict[str, Path]:
    """Download official HAM10000 metadata, images, and Tschandl lesion segmentations."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "segmentations"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    checksum_lines: list[str] = []

    for fname, info in DATAVERSE_FILES.items():
        if not include_masks and "segmentation" in fname:
            continue
        url = f"{DATAVERSE_BASE_URL}/{info['file_id']}"
        dest = raw_dir / fname
        print(f"[HAM10000] Downloading/verifying {fname} from Dataverse...")
        download_file_stream(url, dest, expected_md5=str(info["md5"]), force=force)
        sha256 = compute_file_hash(dest, "sha256")
        checksum_lines.append(f"{sha256}  {fname}")
        downloaded[fname] = dest

    # Extract image archives safely (prevent zip-slip path traversal)
    for part in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
        if part in downloaded:
            _extract_images_flat(downloaded[part], images_dir, suffix=".jpg")

    if include_masks and "HAM10000_segmentations_lesion_tschandl.zip" in downloaded:
        _extract_images_flat(
            downloaded["HAM10000_segmentations_lesion_tschandl.zip"], masks_dir, suffix=".png"
        )

    CHECKSUMS_FILE.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    meta = {
        "dataset": "HAM10000 (Human Against Machine with 10000 training images)",
        "doi": HAM10000_DOI,
        "url": DATAVERSE_DATASET_URL,
        "license": "CC BY-NC 4.0 (Harvard Dataverse v4.0)",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "image_count": len(list(images_dir.glob("*.jpg"))),
        "mask_count": len(list(masks_dir.glob("*.png"))),
        "files": {k: to_relative_path(v) for k, v in downloaded.items()},
    }
    DATASET_META_FILE.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return downloaded


def _extract_images_flat(zip_path: Path, target_dir: Path, *, suffix: str) -> int:
    """Safely extract files with the expected suffix into `target_dir` without path traversal."""
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if not name.lower().endswith(suffix):
                continue
            out_path = target_dir / name
            if out_path.exists() and out_path.stat().st_size == member.file_size:
                continue
            with zf.open(member, "r") as src, out_path.open("wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


# ---------------------------------------------------------------------------
# Metadata Cleaning & Patient/Lesion-Grouped Splits (Zero Data Leakage)
# ---------------------------------------------------------------------------


def bucket_age(age: Any) -> str:
    if pd.isna(age):
        return "unknown"
    try:
        val = float(age)
    except (TypeError, ValueError):
        return "unknown"
    for label, lo, hi in AGE_BUCKETS:
        if lo <= val < hi:
            return label
    return "unknown"


def group_localization(loc: Any) -> str:
    if pd.isna(loc):
        return "unknown"
    key = str(loc).strip().lower()
    return LOCALIZATION_GROUPS.get(key, "other")


def load_ham10000_metadata(
    meta_path: Path | None = None,
    images_dir: Path = RAW_IMAGES_DIR,
    masks_dir: Path = RAW_MASKS_DIR,
) -> pd.DataFrame:
    """Load and standardize HAM10000 clinical metadata table."""
    if meta_path is None:
        tab_path = RAW_DATA_DIR / "HAM10000_metadata.tab"
        csv_path = RAW_DATA_DIR / "HAM10000_metadata.csv"
        meta_path = tab_path if tab_path.exists() else csv_path
    if not meta_path.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata file not found at {meta_path}. "
            "Run `python -m Workshop_6.src.medical_preprocessing --download --prepare`."
        )

    sep = "\t" if meta_path.suffix.lower() == ".tab" else ","
    df = pd.read_csv(meta_path, sep=sep)
    required_cols = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"HAM10000 metadata missing required columns: {sorted(missing)}")

    df = df.copy()
    df["lesion_id"] = df["lesion_id"].astype(str).str.strip()
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["dx"] = df["dx"].astype(str).str.strip().str.lower()
    df = df[df["dx"].isin(DX_TO_ID)].drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    df["label"] = df["dx"].map(DX_TO_ID).astype(np.int64)
    df["is_malignant"] = df["dx"].isin(MALIGNANT_CLASSES).astype(np.int64)
    df["age_group"] = df["age"].apply(bucket_age)
    df["sex"] = df["sex"].fillna("unknown").astype(str).str.strip().str.lower()
    df["dx_type"] = df["dx_type"].fillna("unknown").astype(str).str.strip().str.lower()
    df["localization_group"] = df["localization"].apply(group_localization)

    if images_dir.exists():
        df["image_path"] = df["image_id"].apply(lambda x: str(images_dir / f"{x}.jpg"))
        df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    if masks_dir.exists():
        df["mask_path"] = df["image_id"].apply(lambda x: str(masks_dir / f"{x}_segmentation.png"))
        df["mask_exists"] = df["mask_path"].apply(lambda p: Path(p).exists())

    return df


def make_lesion_grouped_splits(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Create leakage-free 70/15/15 splits grouped strictly by `lesion_id`."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")
    if "lesion_id" not in df.columns or "image_id" not in df.columns:
        raise ValueError("DataFrame must include 'lesion_id' and 'image_id'")

    groups = df["lesion_id"].to_numpy()
    y = df["label"].to_numpy()

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=(1.0 - train_ratio), random_state=seed)
    train_idx, temp_idx = next(gss_outer.split(df, y, groups))

    temp_df = df.iloc[temp_idx]
    val_frac_of_temp = val_ratio / (val_ratio + test_ratio)
    gss_inner = GroupShuffleSplit(
        n_splits=1, test_size=(1.0 - val_frac_of_temp), random_state=seed + 1
    )
    val_rel_idx, test_rel_idx = next(
        gss_inner.split(temp_df, temp_df["label"].to_numpy(), temp_df["lesion_id"].to_numpy())
    )
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    splits = {
        "train": sorted(df.iloc[train_idx]["lesion_id"].unique().tolist()),
        "val": sorted(df.iloc[val_idx]["lesion_id"].unique().tolist()),
        "test": sorted(df.iloc[test_idx]["lesion_id"].unique().tolist()),
    }
    verify_lesion_splits(df, splits)
    return splits


def verify_lesion_splits(df: pd.DataFrame, lesion_splits: dict[str, list[str]]) -> dict[str, Any]:
    """Assert zero lesion_id and image_id leakage across train, val, and test splits."""
    sets = {k: set(v) for k, v in lesion_splits.items()}
    overlaps = {
        "train&val": len(sets["train"] & sets["val"]),
        "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Lesion leakage detected across splits: {overlaps}")

    img_sets: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for name in ("train", "val", "test"):
        sub = df[df["lesion_id"].isin(sets[name])]
        img_sets[name] = set(sub["image_id"].tolist())
        counts[name] = {
            "lesions": int(sub["lesion_id"].nunique()),
            "images": int(len(sub)),
            "malignant_images": int(sub["is_malignant"].sum()),
        }

    img_overlaps = {
        "train&val": len(img_sets["train"] & img_sets["val"]),
        "train&test": len(img_sets["train"] & img_sets["test"]),
        "val&test": len(img_sets["val"] & img_sets["test"]),
    }
    if any(img_overlaps.values()):
        raise AssertionError(f"Image leakage detected across splits: {img_overlaps}")

    return {"lesion_overlaps": overlaps, "image_overlaps": img_overlaps, "counts": counts}


# ---------------------------------------------------------------------------
# Modality-Specific Dermoscopy Preprocessing (DullRazor + Shades of Gray + Crop)
# ---------------------------------------------------------------------------


def crop_dermoscope_vignette(img_rgb: np.ndarray, dark_threshold: int = 12) -> np.ndarray:
    """Remove black circular dermoscope borders while preserving full lesion content."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {img_rgb.shape}")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > dark_threshold
    if not np.any(mask):
        return img_rgb

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    # Guard against degenerate over-cropping (require at least 55% of original dimensions)
    h, w = img_rgb.shape[:2]
    if (y1 - y0) < int(0.55 * h) or (x1 - x0) < int(0.55 * w):
        return img_rgb
    return img_rgb[y0:y1, x0:x1]


def remove_hair_dullrazor(
    img_rgb: np.ndarray,
    *,
    kernel_size: int = 17,
    threshold: int = 18,
    inpaint_radius: int = 3,
) -> np.ndarray:
    """Morphological DullRazor hair artifact removal (Black-Hat + Telea inpainting)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # Cap inpainted area to avoid erasing darkly pigmented lesions (safety guard: <=15% pixels)
    hair_fraction = float((hair_mask > 0).mean())
    if hair_fraction == 0.0 or hair_fraction > 0.15:
        return img_rgb

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cleaned_bgr = cv2.inpaint(bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)


def shades_of_gray_color_constancy(img_rgb: np.ndarray, power: float = 6.0) -> np.ndarray:
    """Apply Shades of Gray (Minkowski norm p=6) color constancy across dermatoscopes."""
    img_f = img_rgb.astype(np.float32)
    img_pow = np.power(np.maximum(img_f, 1e-6), power)
    channel_norms = np.power(img_pow.mean(axis=(0, 1)), 1.0 / power)
    norm_l2 = float(np.sqrt(np.sum(channel_norms**2)))
    if norm_l2 < 1e-6:
        return img_rgb
    scale = (norm_l2 / np.sqrt(3.0)) / np.maximum(channel_norms, 1e-6)
    # Clamp gain to avoid extreme color amplification on very dark or monochrome frames
    scale = np.clip(scale, 0.75, 1.35)
    corrected = np.clip(img_f * scale[None, None, :], 0.0, 255.0)
    return corrected.astype(np.uint8)


def preprocess_dermoscopy_image(
    img_rgb: np.ndarray,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    enable_medical_preproc: bool = True,
) -> np.ndarray:
    """Full dermoscopy preprocessing pipeline (togglable for Ablation 2)."""
    out = img_rgb
    if enable_medical_preproc:
        out = crop_dermoscope_vignette(out)
        out = remove_hair_dullrazor(out)
        out = shades_of_gray_color_constancy(out, power=6.0)
    if out.shape[0] != image_size or out.shape[1] != image_size:
        out = cv2.resize(out, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return out


def compute_train_channel_stats(
    image_paths: list[str],
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    enable_medical_preproc: bool = True,
    max_samples: int = 2000,
    seed: int = SEED,
) -> tuple[list[float], list[float]]:
    """Estimate RGB mean/std strictly on training images (zero val/test leakage)."""
    rng = np.random.default_rng(seed)
    paths = (
        list(rng.choice(image_paths, size=min(max_samples, len(image_paths)), replace=False))
        if len(image_paths) > max_samples
        else list(image_paths)
    )
    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        proc = preprocess_dermoscopy_image(
            img, image_size=image_size, enable_medical_preproc=enable_medical_preproc
        )
        arr = proc.astype(np.float64) / 255.0
        sum_c += arr.sum(axis=(0, 1))
        sum_sq_c += (arr**2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

    if n_pixels == 0:
        return list(IMAGENET_MEAN), list(IMAGENET_STD)
    mean = sum_c / float(n_pixels)
    var = np.maximum(sum_sq_c / float(n_pixels) - mean**2, 1e-8)
    return [float(x) for x in mean], [float(x) for x in np.sqrt(var)]


# ---------------------------------------------------------------------------
# Clinically Valid Augmentations & PyTorch Dataset
# ---------------------------------------------------------------------------


class RandomRightAngleRotation:
    """Exact 90-degree rotations (valid in dermoscopy due to rotational symmetry)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return img
        return img.rotate(90 * k)


def build_medical_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    is_train: bool = True,
    mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
    std: tuple[float, float, float] | list[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Clinically valid dermoscopy transforms (D4 symmetry + mild photometric, NO hue distortion)."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRightAngleRotation(),
                transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.90, 1.10)),
                # Hue strictly 0.0: melanin/vascular color is a primary ABCDE diagnostic cue
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


class HAM10000Dataset(TorchDataset):
    """PyTorch Dataset for HAM10000 with optional precomputed cache, masks, and subgroup metadata."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        enable_medical_preproc: bool = True,
        transform: transforms.Compose | None = None,
        cache_dir: Path | None = None,
        return_metadata: bool = False,
        load_mask: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.enable_medical_preproc = bool(enable_medical_preproc)
        self.transform = transform
        self.cache_dir = cache_dir
        self.return_metadata = bool(return_metadata)
        self.load_mask = bool(load_mask)

    def __len__(self) -> int:
        return len(self.df)

    def _load_rgb(self, row: pd.Series) -> np.ndarray:
        image_id = str(row["image_id"])
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{image_id}.png"
            if cached.exists():
                return np.asarray(Image.open(cached).convert("RGB"), dtype=np.uint8)
        raw = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)
        return preprocess_dermoscopy_image(
            raw,
            image_size=self.image_size,
            enable_medical_preproc=self.enable_medical_preproc,
        )

    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]
        rgb = self._load_rgb(row)
        pil_img = Image.fromarray(rgb)
        x = (
            self.transform(pil_img)
            if self.transform is not None
            else transforms.ToTensor()(pil_img)
        )
        y = int(row["label"])

        if not self.return_metadata and not self.load_mask:
            return x, y

        extra: dict[str, Any] = {
            "image_id": str(row["image_id"]),
            "lesion_id": str(row["lesion_id"]),
            "dx": str(row["dx"]),
            "is_malignant": int(row["is_malignant"]),
            "age_group": str(row["age_group"]),
            "sex": str(row["sex"]),
            "localization_group": str(row["localization_group"]),
            "dx_type": str(row["dx_type"]),
        }
        if self.load_mask:
            mask_path = Path(str(row.get("mask_path", "")))
            if mask_path.exists():
                m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
                m = cv2.resize(
                    m, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
                )
                extra["mask"] = torch.from_numpy((m > 127).astype(np.float32))
            else:
                extra["mask"] = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        return x, y, extra


# ---------------------------------------------------------------------------
# Class Imbalance Handling (Class-Balanced Loss, Focal Loss, Sampler)
# ---------------------------------------------------------------------------


def compute_class_balanced_weights(
    class_counts: np.ndarray | list[int],
    *,
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute effective-number class weights (Cui et al., CVPR 2019), normalized to mean 1."""
    counts = np.asarray(class_counts, dtype=np.float64)
    safe_counts = np.maximum(counts, 1.0)
    effective_num = 1.0 - np.power(beta, safe_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    """Multi-class Focal Loss with optional Class-Balanced per-class weights and label smoothing."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * ce).mean()


def build_weighted_sampler(labels: np.ndarray | list[int]) -> WeightedRandomSampler:
    """Create inverse-sqrt frequency WeightedRandomSampler strictly for training split."""
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    sample_weights = torch.tensor(inv_sqrt[y], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(y), replacement=True)


# ---------------------------------------------------------------------------
# Clinical DICOM Interoperability & HIPAA PHI De-identification Audit (pydicom)
# ---------------------------------------------------------------------------


def read_dicom_medical_image(dicom_path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Safely read DICOM pixels and a strict non-PHI technical allowlist (pydicom 3.0.2)."""
    path = Path(dicom_path)
    ds_meta = pydicom.dcmread(
        path,
        stop_before_pixels=True,
        specific_tags=list(DICOM_TECHNICAL_ALLOWLIST),
    )
    technical: dict[str, Any] = {
        tag: str(ds_meta.get(tag)) if tag in ds_meta else None for tag in DICOM_TECHNICAL_ALLOWLIST
    }

    ds_full = pydicom.dcmread(path)
    arr = ds_full.pixel_array

    if arr.ndim == 2:
        mod_vals = apply_modality_lut(arr, ds_full)
        disp_vals = apply_voi_lut(mod_vals, ds_full, index=0).astype(np.float32)
        if str(ds_full.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            disp_vals = disp_vals.max() - disp_vals
        lo, hi = float(disp_vals.min()), float(disp_vals.max())
        norm = (disp_vals - lo) / max(hi - lo, 1e-6)
        uint8_gray = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb = np.stack([uint8_gray, uint8_gray, uint8_gray], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        rgb = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported DICOM pixel shape: {arr.shape}")

    return rgb, technical


def audit_dicom_phi_cleanliness(dicom_path: Path | str) -> dict[str, Any]:
    """Audit a DICOM file recursively for non-empty HIPAA Safe Harbor tags and private elements.

    Never emits raw PHI element values into logs or output; returns only tag names and counts.
    """
    path = Path(dicom_path)
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    flagged_phi_tags: list[str] = []
    private_tag_count = 0

    def _walk(dataset: Dataset, prefix: str = "") -> None:
        nonlocal private_tag_count
        for elem in dataset:
            if elem.tag.is_private:
                private_tag_count += 1
                continue
            if elem.VR == "SQ":
                for idx, item in enumerate(elem.value):
                    _walk(item, prefix=f"{prefix}{elem.keyword}[{idx}].")
                continue
            if elem.keyword in HIPAA_PHI_DICOM_KEYWORDS:
                val = elem.value
                is_non_empty = val is not None and str(val).strip() != ""
                if is_non_empty:
                    flagged_phi_tags.append(f"{prefix}{elem.keyword}")

    _walk(ds)
    return {
        "file": path.name,
        "phi_tags_present": sorted(set(flagged_phi_tags)),
        "phi_tag_count": len(flagged_phi_tags),
        "private_tag_count": private_tag_count,
        "passes_basic_phi_check": (len(flagged_phi_tags) == 0 and private_tag_count == 0),
        "disclaimer": (
            "Technical metadata pre-check only; does not inspect burned-in pixel PHI "
            "or substitute for legal HIPAA Safe Harbor / DICOM PS3.15 expert review."
        ),
    }


def create_synthetic_dermoscopy_dicom(
    output_path: Path | str,
    rgb_array: np.ndarray,
    *,
    include_synthetic_phi: bool = False,
) -> Path:
    """Create a standards-compliant synthetic Visible Light Dermoscopy DICOM fixture for tests."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3 or rgb_array.dtype != np.uint8:
        raise ValueError("rgb_array must have shape (H, W, 3) and dtype uint8")

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = VLPhotographicImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(out), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "XC"  # External-camera photography / dermoscopy
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = int(rgb_array.shape[0]), int(rgb_array.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = rgb_array.tobytes()

    if include_synthetic_phi:
        ds.PatientName = "SYNTHETIC^TESTPATIENT"
        ds.PatientID = "SYNTH-0001"
        ds.InstitutionName = "Synthetic Test Clinic"

    pydicom.dcmwrite(str(out), ds, enforce_file_format=True, overwrite=True)
    return out


# ---------------------------------------------------------------------------
# Preparation, Caching & EDA Figures
# ---------------------------------------------------------------------------


def prepare_splits_and_cache(
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    precompute_cache: bool = True,
    seed: int = SEED,
) -> dict[str, Any]:
    """Build leakage-free splits.json, train normalization stats, EDA plots, and optional cache."""
    seed_everything(seed, deterministic=True)
    df = load_ham10000_metadata()
    if "image_exists" in df.columns:
        missing_imgs = int((~df["image_exists"]).sum())
        if missing_imgs > 0:
            raise FileNotFoundError(
                f"{missing_imgs} HAM10000 images are missing in {RAW_IMAGES_DIR}. "
                "Re-run with `--download`."
            )

    if SPLITS_FILE.exists():
        saved = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
        lesion_splits = saved["lesion_ids"]
        report = verify_lesion_splits(df, lesion_splits)
    else:
        lesion_splits = make_lesion_grouped_splits(df, seed=seed)
        report = verify_lesion_splits(df, lesion_splits)

    train_df = df[df["lesion_id"].isin(set(lesion_splits["train"]))].reset_index(drop=True)
    mean_pre, std_pre = compute_train_channel_stats(
        train_df["image_path"].tolist(),
        image_size=image_size,
        enable_medical_preproc=True,
        seed=seed,
    )
    mean_raw, std_raw = compute_train_channel_stats(
        train_df["image_path"].tolist(),
        image_size=image_size,
        enable_medical_preproc=False,
        seed=seed,
    )

    class_dist = {
        split_name: {
            cls: int((df[df["lesion_id"].isin(set(lesion_splits[split_name]))]["dx"] == cls).sum())
            for cls in CLASS_NAMES
        }
        for split_name in ("train", "val", "test")
    }

    payload = {
        "dataset": "HAM10000",
        "doi": HAM10000_DOI,
        "seed": seed,
        "image_size": image_size,
        "split_strategy": "GroupShuffleSplit(lesion_id, 70/15/15)",
        "lesion_ids": lesion_splits,
        "verification": report,
        "class_distribution": class_dist,
        "train_stats": {
            "medical_preproc": {"mean": mean_pre, "std": std_pre},
            "raw": {"mean": mean_raw, "std": std_raw},
        },
    }
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if precompute_cache:
        _precompute_image_cache(df, image_size=image_size, enable_medical_preproc=True)
        _precompute_image_cache(df, image_size=image_size, enable_medical_preproc=False)

    plot_preprocessing_examples(df, image_size=image_size)
    plot_dataset_eda_summary(df)
    return payload


def get_cache_dir(image_size: int, enable_medical_preproc: bool) -> Path:
    mode = "preproc" if enable_medical_preproc else "raw"
    return PROCESSED_DIR / f"{mode}_{image_size}"


def _precompute_image_cache(
    df: pd.DataFrame, *, image_size: int, enable_medical_preproc: bool
) -> Path:
    out_dir = get_cache_dir(image_size, enable_medical_preproc)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    if existing >= len(df):
        return out_dir

    mode_name = "medical_preproc" if enable_medical_preproc else "raw"
    print(f"[Cache] Precomputing {len(df):,} images ({mode_name}, {image_size}x{image_size})...")
    for _, row in df.iterrows():
        out_path = out_dir / f"{row['image_id']}.png"
        if out_path.exists():
            continue
        raw = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)
        proc = preprocess_dermoscopy_image(
            raw, image_size=image_size, enable_medical_preproc=enable_medical_preproc
        )
        Image.fromarray(proc).save(out_path, compress_level=1)
    return out_dir


def load_split_dataframes(
    splits_path: Path = SPLITS_FILE,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load `train`, `val`, `test` DataFrames using persisted `splits.json`."""
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {splits_path}. Run `python -m Workshop_6.src.medical_preprocessing --prepare`."
        )
    saved = json.loads(splits_path.read_text(encoding="utf-8"))
    df = load_ham10000_metadata()
    verify_lesion_splits(df, saved["lesion_ids"])

    dfs = {
        name: df[df["lesion_id"].isin(set(saved["lesion_ids"][name]))].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    return dfs, saved


def plot_preprocessing_examples(df: pd.DataFrame, *, image_size: int = DEFAULT_IMAGE_SIZE) -> Path:
    """Save visual before/after comparison of dermoscopy preprocessing steps."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sample_classes = ["mel", "bcc", "akiec", "nv"]
    chosen = []
    for cls in sample_classes:
        sub = df[df["dx"] == cls]
        if len(sub) > 0:
            chosen.append(sub.iloc[0])

    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.0 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = [
        "Original Dermoscopy",
        "1. Vignette Crop",
        "2. DullRazor Hair Removal",
        "3. Shades of Gray (p=6)",
    ]
    for r, row in enumerate(chosen):
        raw = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)
        s1 = crop_dermoscope_vignette(raw)
        s2 = remove_hair_dullrazor(s1)
        s3 = shades_of_gray_color_constancy(s2, power=6.0)
        imgs = [
            cv2.resize(raw, (image_size, image_size), interpolation=cv2.INTER_AREA),
            cv2.resize(s1, (image_size, image_size), interpolation=cv2.INTER_AREA),
            cv2.resize(s2, (image_size, image_size), interpolation=cv2.INTER_AREA),
            cv2.resize(s3, (image_size, image_size), interpolation=cv2.INTER_AREA),
        ]
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, fontweight="bold")
        axes[r, 0].set_ylabel(f"{row['dx'].upper()}\n({row['image_id']})", fontsize=9)

    fig.tight_layout()
    out_path = FIGURES_DIR / "preprocessing_examples.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dataset_eda_summary(df: pd.DataFrame) -> Path:
    """Save publication-quality EDA summary (class imbalance, age, sex, dx_type)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # 1. Class distribution (imbalance 58.3:1)
    counts = df["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    colors = ["#D55E00" if c in MALIGNANT_CLASSES else "#0072B2" for c in CLASS_NAMES]
    axes[0].bar(CLASS_NAMES, counts.values, color=colors)
    axes[0].set_title("Class Distribution (Malignant in Orange)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Images (log scale)")
    axes[0].set_yscale("log")
    for idx, val in enumerate(counts.values):
        axes[0].text(idx, val * 1.15, str(val), ha="center", va="bottom", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Age & malignancy prevalence
    age_order = ["<40", "40-59", "60-74", ">=75", "unknown"]
    age_prev = df.groupby("age_group")["is_malignant"].mean().reindex(age_order).fillna(0.0) * 100.0
    axes[1].bar(age_order, age_prev.values, color="#009E73")
    axes[1].set_title("Malignancy Prevalence by Age Group", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Malignant (%)")
    for idx, val in enumerate(age_prev.values):
        axes[1].text(idx, val + 0.8, f"{val:.1f}%", ha="center", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Diagnostic verification method (annotation quality variability)
    dx_counts = df["dx_type"].value_counts()
    axes[2].bar(dx_counts.index.tolist(), dx_counts.values, color="#CC79A7")
    axes[2].set_title("Ground-Truth Verification Method (dx_type)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Images")
    axes[2].tick_params(axis="x", rotation=15)
    for idx, val in enumerate(dx_counts.values):
        axes[2].text(idx, val + 80, str(val), ha="center", fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = FIGURES_DIR / "dataset_eda_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HAM10000 download, leakage-free splitting, and dermoscopy preprocessing."
    )
    parser.add_argument("--download", action="store_true", help="Download HAM10000 from Dataverse")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--prepare", action="store_true", help="Create splits.json and EDA plots")
    parser.add_argument(
        "--no-cache", action="store_true", help="Skip precomputing resized/preprocessed PNG caches"
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.download:
        download_ham10000(force=args.force)
    if args.prepare:
        info = prepare_splits_and_cache(
            image_size=args.image_size,
            precompute_cache=not args.no_cache,
            seed=args.seed,
        )
        print("[OK] Leakage-free splits saved to", to_relative_path(SPLITS_FILE))
        print(json.dumps(info["verification"], indent=2))
    if not args.download and not args.prepare:
        parser.print_help()


if __name__ == "__main__":
    main()
