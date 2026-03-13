# When Transformers Fail: Evaluating Robustness of Embedding-Based Vulnerability Detectors

This repository contains the project artifacts for the paper:

**When Transformers Fail: Evaluating Robustness of Embedding-Based Vulnerability Detectors**

The workspace implements and evaluates **EFORT-VD**, a framework for robustness analysis of transformer-based vulnerability detectors under semantic-preserving code transformations and attribution-guided adversarial perturbations.

---

## Abstract

Transformer-based models for vulnerability detection can achieve strong benchmark scores, but their reliability under realistic code variation is still unclear. This project evaluates that gap by generating multiple transformed datasets (Type-1 to Type-4 clones) and adversarially perturbed samples that preserve program behavior. The evaluation shows that all tested detectors degrade under semantic-preserving transformations, with model-specific failure modes in precision/recall trade-offs, representation shifts, and adversarial susceptibility.

---

## Core Contributions

- Introduces **EFORT-VD** for systematic robustness testing of vulnerability detectors.
- Implements cumulative semantic-preserving transformations:
  - **Type-1**: formatting/comments
  - **Type-2**: renaming
  - **Type-3**: structural perturbations (e.g., logging/call injection/dead code/reordering)
  - **Type-4**: heavy semantics-preserving refactoring via agent-based generation + validation
- Adds **attribution-guided adversarial generation** from token-level non-vulnerable signals.
- Evaluates four SOTA detectors across clean, transformed, and adversarial settings:
  - `VulBERTa`
  - `PDBERT`
  - `LineVul`
  - `StagedVulBERT`

---

## Research Questions

- **RQ1**: How do semantic-preserving transformations affect test performance?
- **RQ2**: Do failures persist on transformed training samples?
- **RQ3**: How do transformations alter representation geometry (PCA/density/boundaries)?
- **RQ4**: Can attribution-guided adversarial examples reveal systematic weaknesses?

---

## Repository Structure

### Models / Results
- `VulBERTa/`
- `PDBERT/`
- `LineVul/`
- `StagedVulBERT/`

### Analysis and Visualization
- `plot.ipynb` — main notebook for PCA, density views, decision boundaries, adversarial visualization.



### Outputs
- `*.pdf`, `*.png`, `*.html` generated plots and figures.

---

## Reproducibility Workflow

1. Ensure model output files are available under each model folder (e.g., `clean_*`, `type*_*`).
2. Open and execute `plot.ipynb` for full visual/metric analysis.

---

## Environment

Local environment directory:

- `env_efortvd/`

Typical packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `plotly`
- `jupyter`

---

## Notes

- Most files in this folder are experimental artifacts generated during evaluation.
- Some scripts use absolute paths; update paths if relocating the project.
- `plot.ipynb` is the canonical entry point for reproducing figures and diagnostics.
