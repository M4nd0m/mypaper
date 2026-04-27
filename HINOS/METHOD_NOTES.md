# HINOS Method Notes

## Objective

The current training objective still has exactly three main loss terms:

```text
L = lambda_temp * L_temp + lambda_com * L_com + lambda_batch * L_batch
```

No separate compact, view, entropy, balance, or TPPR feedback loss is added.

The three terms have distinct roles:

- `L_temp`: preserves temporal interaction information through Hawkes-style positive/negative interaction modeling.
- `L_com`: applies differentiable cut and assignment regularization on the TPPR-induced graph.
- `L_batch`: performs community-aware compact reconstruction inside each mini-batch.

`loss_bal` and `weighted_bal` may still appear in metrics for compatibility with older scripts, but they are legacy aliases/diagnostics for the assignment-prior part of `L_com`. They are not a fourth loss term and `lambda_bal` is not added to `loss_total`.

## Community-Aware Compact Batch Reconstruction

The legacy batch reconstruction treated all observed positive edges as target `1`. That is reasonable for highly repeated and homophilous interactions, but it can be harmful for collaboration-style graphs where an observed edge may cross community boundaries.

`ones` is the current recommended default for compact batch reconstruction. Pseudo-label targets remain available for ablation and are generated from the current assignment matrix `S` with gradients stopped:

```text
M_uv = stopgrad(S_u S_v^T)
```

For a source node `u`, target node `v`, historical node `h`, and negative node `n`:

```text
L_batch =
  masked MSE(cos(z_u, z_v), M_uv)
  + masked MSE(cos(z_u, z_h), M_uh)
  + MSE(cos(z_u, z_n), 0)
```

Historical positions with mask value `0` do not produce reconstruction loss. Negative samples keep target `0`.

This borrows the pseudo-label reconstruction idea from CT-VAE-style compact reconstruction, but it does not import the full VAE/GMM framework and does not add any extra loss term.

## Batch Reconstruction Modes

Use `--batch_recon_mode` to choose the reconstruction target:

- `ones`: legacy behavior; all observed positive source-target and source-history pairs have target `1`.
- `soft_pseudo`: ablation mode; target is `stopgrad(S_u S_v^T)`.
- `hard_pseudo`: target is `1[argmax_k S_uk == argmax_k S_vk]`; this is stronger but may be less stable early in training.
- `hard_pseudo_gate`: optional experiment mode. It reconstructs only high-confidence same-pseudo-label observed edges, using `--pseudo_conf_threshold` to gate both endpoints. Cross-pseudo observed edges are ignored by the positive reconstruction term instead of being pushed apart.

Default behavior:

- `objective_mode=cut_main`: `batch_recon_mode=ones`
- `objective_mode=original`: `batch_recon_mode=ones`

`soft_pseudo` is no longer the default. When `S` has high entropy or is close to uniform, `S_u S_v^T` is close to `1/K`, which can lower the cosine target for observed positive edges and weaken `argmax_s`.

## Main Prediction

For `cut_main`, the default exported main prediction is `argmax_s`, because the assignment matrix `S` is the end-to-end clustering output. KMeans variants remain available as diagnostics and comparison outputs:

- `kmeans_z`
- `kmeans_s`
- `spectral_pi`
- `spectral_topk_pi`

## Why This Change

The old reconstruction assumed every observed positive edge should be pulled close in embedding space. On School, where repeated interactions and homophily are strong, this problem is often hidden because the dataset saturates quickly.

On DBLP, citation, and collaboration networks, observed edges may represent cross-community cooperation. Pulling every such edge close can blur community boundaries, but treating different pseudo labels as negative observed edges is also too strong. `hard_pseudo_gate` is therefore kept as an experiment: it reconstructs high-confidence same-pseudo observed edges and leaves cross-pseudo observed edges neutral.

This repair is based on experimental observations: DBLP compact_batch final `argmax_s` dropped clearly; School compact_batch still had strong `kmeans_z` while `argmax_s` dropped clearly. This points to degraded soft assignment `S`, not degraded embeddings.

Do not judge the method only by School, because School saturates quickly. DBLP is more useful for checking whether compact reconstruction improves community boundary quality.

## Server Training Commands

Run these on the server, not on a local laptop.

Recommended DBLP command:

```bash
cd HINOS
python main.py \
  --dataset dblp \
  --objective_mode cut_main \
  --epoch 100 \
  --lambda_temp 0.005 \
  --lambda_batch 0.01 \
  --lambda_com 0.01 \
  --rho_assign 25 \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --main_pred_mode argmax_s \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --run_tag dblp_compact_batch
```

Recommended School command:

```bash
cd HINOS
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 50 \
  --lambda_temp 0.005 \
  --lambda_batch 0.005 \
  --lambda_com 0.005 \
  --rho_assign 25 \
  --batch_recon_mode ones \
  --warmup_epochs 5 \
  --com_ramp_epochs 10 \
  --main_pred_mode argmax_s \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --run_tag school_compact_batch
```
