"""Central configuration, HAM10000 clinical metadata constants, and portable paths for Workshop 6."""

from __future__ import annotations

import sys
from pathlib import Path

W6_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config import SEED, W2_BEST_CHECKPOINT, to_relative_path  # noqa: E402

# Directory layout (strictly confined inside Workshop_6/)
DATA_DIR: Path = W6_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_IMAGES_DIR: Path = RAW_DATA_DIR / "images"
RAW_MASKS_DIR: Path = RAW_DATA_DIR / "segmentations"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_FILE: Path = DATA_DIR / "splits.json"
CHECKSUMS_FILE: Path = RAW_DATA_DIR / "CHECKSUMS.txt"
DATASET_META_FILE: Path = RAW_DATA_DIR / "dataset_meta.json"

CHECKPOINTS_DIR: Path = W6_ROOT / "checkpoints"
RUNS_DIR: Path = W6_ROOT / "runs"
REPORTS_DIR: Path = W6_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
XAI_FIGURES_DIR: Path = FIGURES_DIR / "interpretability"
LATEX_DIR: Path = REPORTS_DIR / "latex"

# Official Harvard Dataverse record for HAM10000 (Tschandl et al., Scientific Data 2018)
HAM10000_DOI: str = "doi:10.7910/DVN/DBW86T"
DATAVERSE_BASE_URL: str = "https://dataverse.harvard.edu/api/access/datafile"
DATAVERSE_DATASET_URL: str = (
    "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
)
DATAVERSE_FILES: dict[str, dict[str, str | int]] = {
    "HAM10000_metadata.tab": {
        "file_id": 4338392,
        # Dataverse serves ingested TSV by default (md5 below); ?format=original CSV is 8f85fb1a...
        "md5": "1d13fed99da9c3e925fb523df87d527e",
    },
    "HAM10000_images_part_1.zip": {
        "file_id": 3172585,
        "md5": "4639bfa73ab251610530a97c898e6e46",
    },
    "HAM10000_images_part_2.zip": {
        "file_id": 3172584,
        "md5": "da43d6cc50f6613013be07e8986b384b",
    },
    "HAM10000_segmentations_lesion_tschandl.zip": {
        "file_id": 3838943,
        "md5": "6e8d252e09cfdb0189199f15985a5b84",
    },
}

# Canonical 7 diagnostic classes of HAM10000 / ISIC 2018 Task 3 (alphabetical order)
CLASS_NAMES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
NUM_CLASSES: int = len(CLASS_NAMES)
DX_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_DX: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

CLASS_FULL_NAMES: dict[str, str] = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma (Bowen's)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Clinical malignancy grouping: malignant / premalignant lesions requiring biopsy or intervention
MALIGNANT_CLASSES: tuple[str, ...] = ("akiec", "bcc", "mel")
MALIGNANT_CLASS_IDS: tuple[int, ...] = tuple(DX_TO_ID[c] for c in MALIGNANT_CLASSES)
MELANOMA_CLASS_ID: int = DX_TO_ID["mel"]

# Image geometry & model hyperparameters
DEFAULT_IMAGE_SIZE: int = 224
GAN_IMAGE_SIZE: int = 64
VAE_IMAGE_SIZE: int = 64

# Clinical operating points (fitted strictly on validation, evaluated on test)
SCREENING_SENSITIVITY_TARGET: float = 0.95
CONFIRMATION_SPECIFICITY_TARGET: float = 0.90

# Default normalization fallback (overridden by train-split statistics in splits.json)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Subgroup bucketing definitions for clinical fairness / hidden stratification audits
AGE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<40", 0.0, 40.0),
    ("40-59", 40.0, 60.0),
    ("60-74", 60.0, 75.0),
    (">=75", 75.0, 200.0),
)

LOCALIZATION_GROUPS: dict[str, str] = {
    "face": "head_neck",
    "scalp": "head_neck",
    "ear": "head_neck",
    "neck": "head_neck",
    "back": "trunk",
    "trunk": "trunk",
    "chest": "trunk",
    "abdomen": "trunk",
    "upper extremity": "upper_extremity",
    "lower extremity": "lower_extremity",
    "foot": "acral_special",
    "hand": "acral_special",
    "acral": "acral_special",
    "genital": "acral_special",
    "unknown": "unknown",
}

__all__ = [
    "AGE_BUCKETS",
    "CHECKSUMS_FILE",
    "CHECKPOINTS_DIR",
    "CLASS_FULL_NAMES",
    "CLASS_NAMES",
    "CONFIRMATION_SPECIFICITY_TARGET",
    "DATASET_META_FILE",
    "DATAVERSE_BASE_URL",
    "DATAVERSE_DATASET_URL",
    "DATAVERSE_FILES",
    "DATA_DIR",
    "DEFAULT_IMAGE_SIZE",
    "DX_TO_ID",
    "FIGURES_DIR",
    "GAN_IMAGE_SIZE",
    "HAM10000_DOI",
    "ID_TO_DX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LATEX_DIR",
    "LOCALIZATION_GROUPS",
    "MALIGNANT_CLASSES",
    "MALIGNANT_CLASS_IDS",
    "MELANOMA_CLASS_ID",
    "NUM_CLASSES",
    "PROCESSED_DIR",
    "RAW_DATA_DIR",
    "RAW_IMAGES_DIR",
    "RAW_MASKS_DIR",
    "REPORTS_DIR",
    "REPO_ROOT",
    "RUNS_DIR",
    "SCREENING_SENSITIVITY_TARGET",
    "SEED",
    "SPLITS_FILE",
    "VAE_IMAGE_SIZE",
    "W2_BEST_CHECKPOINT",
    "W6_ROOT",
    "XAI_FIGURES_DIR",
    "to_relative_path",
]
