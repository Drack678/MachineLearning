# Workshop No. 6 — Medical Domain Adaptation & Clinical Integration (Group 8)

**Course:** Machine Learning (Semester 2026-III) — Universidad Distrital Francisco José de Caldas  
**Original Domain (Workshops 1–5):** Character-level autoregressive text modeling on `flwrlabs/shakespeare`  
**Target Medical Domain (Workshop 6):** Dermatoscopic skin lesion classification & malignancy triage on **HAM10000 / ISIC 2018 Task 3**  
**Release Tag Target:** `v1.0-medical-final`

---

## 1. Medical Dataset & Provenance

- **Dataset:** HAM10000 (*Human Against Machine with 10000 training images*) — Tschandl et al., *Scientific Data* (2018)
- **Official Source:** [Harvard Dataverse — doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (v4.0, CC BY-NC 4.0)
- **Scope:** 10,015 dermatoscopic images from 7,470 unique pigmented skin lesions across 7 diagnostic classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`; 58.3:1 imbalance ratio; 19.5% malignant/premalignant)
- **Clinical Metadata Used for Fairness Audits:** `lesion_id`, `age`, `sex`, `localization`, and `dx_type` (`histo`, `follow_up`, `consensus`, `confocal`)
- **Anatomical Validation Masks:** Official Tschandl pixel-level lesion segmentation masks (`HAM10000_segmentations_lesion_tschandl.zip`) used to quantitatively score Grad-CAM and Attention Rollout alignment against lesion anatomy.

---

## 2. Repository Structure (`Workshop_6/`)

```text
Workshop_6/
├── README.md                              # Reproduction & clinical documentation (this file)
├── requirements.txt                       # Pinned Python dependencies (ROCm-safe; install PyTorch first)
├── .gitignore                             # Ignores raw/processed caches; preserves splits, checkpoints & reports
├── scripts/
│   └── run_all_rocm.sh                    # Single-command end-to-end pipeline for AMD RX 9060 XT (ROCm)
├── src/
│   ├── __init__.py
│   ├── config.py                          # Constants, Dataverse IDs/MD5, clinical classes & thresholds
│   ├── medical_preprocessing.py           # Dataverse download, lesion-grouped splits, DullRazor, Shades of Gray, DICOM PHI audit
│   ├── transfer.py                        # MedHybridTransformer (W2), ResNet18Medical, MedConvVAE2D (W4), MedCondWGANGP2D (W5) & A1-A4
│   ├── medical_metrics.py                 # Sens/Spec/PPV/NPV, Temp Scaling, Thresholds, DCA, Fairness & PSI/KS Drift
│   ├── interpretability.py                # Grad-CAM, Attention Rollout, Tschandl mask alignment, 15 cases & 5 failure modes
│   ├── evaluate_medical.py                # Checkpoint-only reproducible test evaluation & Master W1-W6 Table
│   └── build_report_assets.py             # Auto-generates LaTeX tables & metric macros from verified CSV/JSON outputs
├── tests/
│   ├── test_workshop6_preprocessing.py    # Zero lesion leakage, DullRazor/SoG, hue=0 augs, DICOM PHI checks
│   ├── test_workshop6_transfer.py         # W2->W6 block weight transfer, overfit smoke test, VAE, 2nd-order WGAN-GP
│   ├── test_workshop6_metrics.py          # Exact clinical metrics, Temp Scaling, DCA, fairness alerts, PSI/KS drift
│   └── test_workshop6_interpretability.py # Grad-CAM hook cleanup, Attention Rollout, lesion alignment metrics
├── data/
│   └── splits.json                        # Deterministic 70/15/15 lesion-grouped split manifest (zero leakage)
├── checkpoints/                           # Pure tensor checkpoints (*.pt loaded with weights_only=True + *.meta.json)
└── reports/
    ├── group8_medical_final_report.pdf    # Final compiled 20-30 page scientific report (after pipeline run)
    ├── figures/                           # 300-dpi publication figures & 15 interpretability panels
    └── latex/                             # Modular LaTeX sources, TikZ diagrams, and 39 verified BibTeX citations
```

---

## 3. Environment Setup (AMD Radeon RX 9060 XT — ROCm)

Run all commands from the **repository root** (`MachineLearning/`).

### Option A — AMD Radeon RX 9060 XT (RDNA 4 / `gfx1200`, ROCm — Recommended)
```bash
python3 -m venv Workshop_6/.venv_rocm
source Workshop_6/.venv_rocm/bin/activate

# 1. Install PyTorch + Torchvision ROCm wheels FIRST (do not overwrite with CPU wheel)
python -m pip install --upgrade pip
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx1200]" torchvision

# 2. Install Workshop 6 medical & evaluation libraries
python -m pip install -r Workshop_6/requirements.txt
python -m pip install --no-deps -e .

# 3. Verify GPU & 2nd-order autograd (WGAN-GP)
python -m Workshop_5.src.check_gpu --device rocm
```

### Option B — CPU / NVIDIA CUDA Fallback
```bash
python3 -m venv Workshop_6/.venv
source Workshop_6/.venv/bin/activate
python -m pip install torch torchvision
python -m pip install -r Workshop_6/requirements.txt
```

---

## 4. Reproducing All Experiments, Checkpoints, and Report

### Full End-to-End Execution on AMD RX 9060 XT (~45–60 min total, ≤4.5 GB VRAM)
```bash
PYTHON_BIN=Workshop_6/.venv_rocm/bin/python \
DEVICE=rocm PROFILE=rocm-rx9060 IMAGE_SIZE=224 EPOCHS=25 \
bash Workshop_6/scripts/run_all_rocm.sh
```

### Step-by-Step Execution
```bash
# 1. Download HAM10000 from Harvard Dataverse, verify MD5/SHA-256, create lesion-grouped splits & cache
python -m Workshop_6.src.medical_preprocessing --download --prepare --image-size 224

# 2. Run unit & invariant tests (zero leakage, DICOM PHI audit, W2->W6 transfer, metrics, XAI)
pytest Workshop_6/tests/ -v

# 3. Train all 4 Ablations (A1-A4) + W4 MedConvVAE2D + W5 MedCondWGANGP2D
python -m Workshop_6.src.transfer --run all --device rocm --profile rocm-rx9060 --epochs 25

# 4. Reproduce 100% of clinical test metrics, fairness audit, drift simulation & Master W1-W6 Table from checkpoints
python -m Workshop_6.src.evaluate_medical --device rocm --n-boot 1000

# 5. Generate Grad-CAM, Attention Rollout, 15 TP/FN/FP panels, anatomical alignment & 5 failure modes
python -m Workshop_6.src.interpretability --device rocm --n-per-group 5

# 6. Auto-populate LaTeX tables & macros from verified CSV/JSON reports
python -m Workshop_6.src.build_report_assets
```

### Fast Reproduction Directly From Submitted Checkpoints (No Retraining)
Once `Workshop_6/checkpoints/*.pt` and `Workshop_6/data/splits.json` are present alongside the downloaded HAM10000 images, reproduce every reported test metric, figure, and interpretability panel without retraining:
```bash
python -m Workshop_6.src.evaluate_medical --device auto --n-boot 1000
python -m Workshop_6.src.interpretability --device auto --n-per-group 5
python -m Workshop_6.src.build_report_assets
```

### Compiling the Final PDF Report (`group8_medical_final_report.pdf`)
```bash
cd Workshop_6/reports/latex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf ../group8_medical_final_report.pdf
```

---

## 5. Summary of Honest Domain Transfer Findings (W1–W5 → W6)

| Source | Transferred / Evaluated Component | Outcome |
|---|---|---|
| `shared/` | `device.py`, `checkpoints.py` (`weights_only=True`), `metrics.py` | **Direct Reuse** — portable ROCm/CUDA/CPU AMP and safe serialization |
| Workshop 1 | `GroupShuffleSplit` by entity (`character_id` → `lesion_id`) | **Methodological Transfer** — guarantees zero lesion leakage |
| Workshop 2 | Pre-LN Transformer blocks (`MedHybridTransformer`) + post-hoc Logit Adjustment | **Hybrid Transfer + Negative Control** — architecture & logit adjustment transfer; frozen Shakespeare text blocks serve as honest negative feature-reuse control in A1 |
| Workshop 3 | `TextGenerationEnv` + `Double+Dueling DQN` | **Honest Negative (Rejected)** — static single-step diagnosis is not a lexical MDP |
| Workshop 4 | `ResConv` + FiLM + Free Bits + Cyclical KL (`MedConvVAE2D`) | **1D→2D Adaptation** — benign-trained anomaly/OOD detector |
| Workshop 5 | FP32 WGAN-GP + classifier-filtered augmentation (`MedCondWGANGP2D`) | **1D→2D Adaptation (A4)** — continuous `tanh` replaces Gumbel-ST; empirically tested on rare classes |
