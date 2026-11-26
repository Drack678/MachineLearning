#!/usr/bin/env bash
# End-to-end reproducible pipeline for Workshop 6 on AMD Radeon RX 9060 XT (ROCm / gfx1200)
# Usage:
#   bash Workshop_6/scripts/run_all_rocm.sh              # Full 224x224 training on ROCm RX 9060 XT
#   DEVICE=auto PROFILE=auto EPOCHS=25 bash Workshop_6/scripts/run_all_rocm.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-rocm}"
PROFILE="${PROFILE:-rocm-rx9060}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
EPOCHS="${EPOCHS:-25}"
VAE_EPOCHS="${VAE_EPOCHS:-30}"
GAN_EPOCHS="${GAN_EPOCHS:-35}"

echo "======================================================================"
echo " Workshop 6 — Medical Domain Adaptation (HAM10000 / ISIC 2018)"
echo " Python: $PYTHON_BIN | Device: $DEVICE | Profile: $PROFILE | Size: ${IMAGE_SIZE}px"
echo "======================================================================"

echo "[1/7] Checking GPU / ROCm & 2nd-order autograd (WGAN-GP)..."
"$PYTHON_BIN" -m Workshop_5.src.check_gpu --device "$DEVICE"

echo "[2/7] Downloading HAM10000 (Harvard Dataverse) & building leakage-free splits + cache..."
"$PYTHON_BIN" -m Workshop_6.src.medical_preprocessing --download --prepare --image-size "$IMAGE_SIZE"

echo "[3/7] Running Unit & Invariant Tests..."
"$PYTHON_BIN" -m pytest Workshop_6/tests/ -v

echo "[4/7] Training Domain Transfer Models, 4 Ablations, W4 VAE & W5 cWGAN-GP 2D..."
"$PYTHON_BIN" -m Workshop_6.src.transfer \
  --run all \
  --device "$DEVICE" \
  --profile "$PROFILE" \
  --image-size "$IMAGE_SIZE" \
  --epochs "$EPOCHS" \
  --vae-epochs "$VAE_EPOCHS" \
  --gan-epochs "$GAN_EPOCHS"

echo "[5/7] Running Checkpoint-Driven Clinical Evaluation, Fairness Audit & Drift Simulation..."
"$PYTHON_BIN" -m Workshop_6.src.evaluate_medical --device "$DEVICE" --n-boot 1000

echo "[6/7] Running Clinical Interpretability (Grad-CAM, Attention Rollout, 15 cases, 5 failure modes)..."
"$PYTHON_BIN" -m Workshop_6.src.interpretability --device "$DEVICE" --n-per-group 5

echo "[7/7] Regenerating LaTeX Tables & Compiling Final PDF Report..."
"$PYTHON_BIN" -m Workshop_6.src.build_report_assets

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  (
    cd Workshop_6/reports/latex
    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
    cp main.pdf ../group8_medical_final_report.pdf
  )
  echo "[OK] Compiled Workshop_6/reports/group8_medical_final_report.pdf"
elif command -v tectonic >/dev/null 2>&1; then
  (
    cd Workshop_6/reports/latex
    tectonic main.tex -o ..
    mv ../main.pdf ../group8_medical_final_report.pdf
  )
  echo "[OK] Compiled Workshop_6/reports/group8_medical_final_report.pdf with tectonic"
else
  echo "[INFO] pdflatex/tectonic not found in PATH. Compile Workshop_6/reports/latex/main.tex on a TeX host or Overleaf."
fi

echo "=== Workshop 6 pipeline complete! ==="
