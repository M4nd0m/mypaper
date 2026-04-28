# hinos_ncut_offline

This project adapts the offline stage of HINOS from temporal community search to full-graph temporal community discovery.

Kept:
- adaptive TAPS sampling
- TPPR construction and caching
- temporal loss
- batch reconstruction loss
- TPPR-Cut community regularization
- TGC-style Student-t assignment and KL alignment
- HINOS TPPR-volume balance penalty
- offline embedding and cluster assignment export

Removed:
- query-aware community search
- BFS candidate generation
- top-k retrieval evaluation
- search-specific scoring arguments
- F1-based early stopping

## Current Loss Function

The training objective is:

\[
\mathcal{L}
=
\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
+
\lambda_{\mathrm{batch}}\mathcal{L}_{\mathrm{batch}}
+
\frac{\lambda_{\mathrm{com}}^{(e)}}{B}\mathcal{L}_{\mathrm{com}}.
\]

The community discovery term is:

\[
\mathcal{L}_{\mathrm{com}}
=
\rho_{\mathrm{cut}}\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
+
\rho_{\mathrm{KL}}\mathcal{L}_{\mathrm{TGC\text{-}KL}}
+
\rho_{\mathrm{bal}}\mathcal{L}_{\mathrm{HINOS\text{-}Bal}}.
\]

By default, \(S\) is a TGC-style Student-t assignment \(Q\) from learnable prototype centers initialized by KMeans on pretrained node2vec embeddings. `argmax_s` is the default final clustering output for `cut_main`.

See [docs/loss_function_design.md](docs/loss_function_design.md) for the full formula mapping.

## Main Parameters

- `--rho_cut`: weight for TPPR-Cut inside `L_com` (default `1.0`).
- `--rho_kl`: weight for DTGC/TGC batch-level KL inside `L_com` (default `1.0`).
- `--rho_bal`: weight for HINOS balance inside `L_com` (default `0.1`).
- `--prototype_lr_scale`: prototype-center learning-rate multiplier (default `0.1`).
- `--target_update_interval`: legacy compatibility option; `dynamic_tgc` now builds DTGC batch targets directly.
- `--kl_target_mode`: `dynamic_tgc`, `fixed_initial`, or `none` (default `dynamic_tgc`).
- `--balance_mode`: `hinos` or `none` (default `hinos`).

Legacy aliases are still accepted:

- `--rho_assign` maps to `--rho_kl` when `--rho_kl` is omitted.
- `--lambda_bal` can initialize `--rho_bal` when nonzero and `--rho_bal` is omitted.

## Recommended DBLP Run

Run this on the GPU server:

```bash
cd HINOS
python main.py \
  --dataset dblp \
  --objective_mode cut_main \
  --assign_mode prototype \
  --epoch 40 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 0.2 \
  --rho_cut 1.0 \
  --rho_kl 1.0 \
  --rho_bal 0.1 \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.1 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.1 \
  --run_tag dblp_dtgc_batchkl_hinos_bal_e40
```

## Recommended School Run

```bash
cd HINOS
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --assign_mode prototype \
  --epoch 50 \
  --lambda_temp 0.005 \
  --lambda_batch 0.005 \
  --lambda_com 0.5 \
  --rho_cut 1.0 \
  --rho_kl 1.0 \
  --rho_bal 0.1 \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.1 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 5 \
  --com_ramp_epochs 10 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.5 \
  --run_tag school_dtgc_batchkl_hinos_bal_e50
```

## TAPS Budget

`--taps_budget_mode sqrt_edges` keeps the original compatibility behavior:

\[
N_{\mathrm{TAPS}}=\lceil\sqrt{|\mathcal{E}_{\mathrm{static}}|}\rceil .
\]

For full-graph TPPR-Cut clustering, the default and recommended mode is `nlogn`:

\[
N_{\mathrm{TAPS}}
=
\left\lceil
\beta |\mathcal{V}|\log(|\mathcal{V}|+1)
\right\rceil .
\]

For School, `--taps_budget_beta 0.5` gives about 948 sampled paths.

## Outputs

- learned embeddings: `emb/<dataset>/<dataset>_TGC_<epoch>.emb`
- hard cluster predictions: `emb/<dataset>/<dataset>_TGC_<epoch>_pred.txt`
- soft assignment matrix: `emb/<dataset>/<dataset>_TGC_<epoch>_soft_assign.npy`
- training metrics: `emb/<dataset>/<dataset>_TGC_<epoch>_metrics.csv`

Evaluate clustering:

```bash
python evaluate.py --dataset school --epoch 30
```

Or pass files explicitly:

```bash
python evaluate.py --dataset school --pred_path emb/school/school_TGC_30_pred.txt --label_path dataset/school/node2label.txt
```

Metrics:
- ACC
- NMI
- F1 (macro)
- ARI

## Diagnostics

Track these columns in the metrics CSV:

- `loss_tppr_cut`
- `loss_assign_penalty` (TGC KL)
- `loss_hinos_bal`
- `weighted_cut`
- `weighted_kl`
- `weighted_hinos_bal`
- `assignment_entropy`
- `cluster_volume_min`
- `cluster_volume_max`
- `nmi_argmax_s`
- `nmi_kmeans_z`
- `nmi_spectral_pi`

Static checks for local workstation use:

```bash
python -m py_compile main.py trainer.py data_load.py sparsification.py clustering_utils.py evaluate.py
python main.py --help
```

Do full smoke tests and experiments on the GPU server.
