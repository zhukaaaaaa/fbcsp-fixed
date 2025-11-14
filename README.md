# FBCSP Fixed — 81.8% on BCI IV-2a

**Reproducible FBCSP** with:
- Nested Stratified CV
- **Automatic ICA** (Fp1/Fp2 + spikiness)
- **k=64** (60% folds)
- One-vs-Rest CSP + LinearSVC

**Results**: `81.8% ± 0.9%`, `κ = 0.757 ± 0.013`

---

## Run

```bash
pip install -r requirements.txt
python fbcsp_pipeline.py
DOI: https://doi.org/10.5281/zenodo.13870386
