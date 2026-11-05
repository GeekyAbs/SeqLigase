# Module 4 LSTM Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `module_4/module_4_LSTM.ipynb`, a self-contained unidirectional-LSTM SMILES-infilling notebook that trains on the same QM9 dataset and saves downloadable weights.

**Architecture:** One Jupyter notebook with all logic inline (no `%%writefile`, no `config.py`, no helper `.py` files). Cells: setup, dataset download, split, vocabulary, masking dataset, LSTM model, training utilities, training run, plots, test metrics, weight export, reload check. Training mirrors the optimized BiLSTM settings (embed 192, hidden 384, 3 layers, mask span 6, power 4.0, special-token logit ban).

**Tech Stack:** Python 3.11 on Kaggle T4, PyTorch, RDKit, pandas, scikit-learn, matplotlib, kagglehub.

**Spec:** User request in conversation: LSTM variant, same dataset, no config files/modules, training + weights only, no OpenRouter.

## Global Constraints

- Output file: `module_4/module_4_LSTM.ipynb`; no other source files.
- Dataset slug: `ssshaun/clean-qm9-dataset-fro-seqligase-project`.
- Split: provided `split` column when present, else sklearn two-stage 80/10/10 with `random_state=42`.
- Special tokens: `<PAD>`, `<UNK>`, `<MASK>`, `<SOS>`, `<EOS>` at ids 0-4.
- Masking: contiguous span, `max_mask_span=6`, `mask_span_power=4.0`, labels `-100` outside the span.
- Model: unidirectional `nn.LSTM`, `embed_dim=192`, `hidden_dim=384`, `num_layers=3`, `dropout=0.2`.
- Optimizer: AdamW `lr=1e-3`, `weight_decay=1e-4`, `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip 1.0, AMP on CUDA, early stop 8.
- Artefacts: `exports/seqlgase_lstm_best.pt`, `exports/lstm_history.json`, `exports/lstm_test_metrics.json`, `exports/lstm_span_length_metrics.csv`, `plots/lstm_*.png`.
- No OpenRouter, no CLI, no separate prediction/export modules.

## Review Focus

- Masks never touch `<SOS>`/`<EOS>`; labels align with `original_ids` inside the span.
- `<PAD>` positions are inert: zero embedding, ignored by loss, excluded from pack lengths.
- Special tokens can never win the argmax (logit restriction in train and eval).
- Provided `split` values (`train`, `validation`, `test`) map correctly; fallback is seeded.
- Weight payload reloads without the notebook's globals (class definition required, documented).

---

### Task 1: Build the notebook

**Files:**
- Create: `module_4/module_4_LSTM.ipynb`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `SeqLigaseLSTM`, `MaskedSMILESDataset`, `train_epoch`, `evaluate`, `fit` used only inside the notebook.

- [ ] **Step 1:** Generate the notebook JSON from a builder script, with all code inline.
- [ ] **Step 2:** Add `module_4/exports/*`, `module_4/plots/*`, `*.pt` coverage (already present) to `.gitignore`; confirm only the notebook is tracked.

### Task 2: Verify end-to-end

**Files:**
- Test input: `/tmp/seql_lstm_nb/data/raw/qm9_clean.csv` (synthetic 200 molecules)

- [ ] **Step 1:** Run `nbconvert --execute` on the notebook with the CPU venv.
- [ ] **Step 2:** Assert zero error outputs, 100% code cells executed, checkpoint exists, reload check passes.
- [ ] **Step 3:** Confirm the repo notebook has no saved outputs and the generated weights are ignored by Git.
