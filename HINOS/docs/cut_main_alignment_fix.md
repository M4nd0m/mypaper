# cut_main alignment fix

## Motivation

This change keeps the TPPR + Cut objective as the core of `cut_main`, while fixing three alignment issues in the end-to-end clustering setup:

1. The assignment regularization target is now a fixed prior target built from the initial node features, instead of a self-target regenerated from the current assignment.
2. The full-graph community loss is normalized by the number of mini-batches in the epoch, so the same full-graph term is not implicitly applied `B` times per epoch.
3. The default main prediction for `cut_main` is now the learned soft assignment output, `argmax_s`, instead of `kmeans_z`.

This does not change TPPR/TAPS temporal propagation, the temporal loss, the batch reconstruction loss, geometric symmetrization, or community-aware batch reconstruction.

## Previous Issues

### Self-target assignment regularization

The previous assignment penalty used a DEC-style target `P(S)` constructed from the current assignment matrix `S`, then optimized `KL(P(S) || S)`. Because the target came from the same trainable assignment it regularized, the target could move with unstable or collapsed assignments.

### Scale mismatch

`L_com` is a full-graph community loss, but it was added unchanged to every mini-batch update. With `B` mini-batches in an epoch, the effective epoch-level contribution became approximately `B * lambda_com * L_com`.

### Output mismatch

`cut_main` is intended to be TPPR-Cut guided end-to-end clustering, but the default exported prediction was still `kmeans_z`. That made the main output depend on a post-hoc KMeans over embeddings rather than the learned assignment head.

## Updated Formulation

Let `Z0` be the existing initial node feature matrix and let `C0` be KMeans centers fitted on `Z0`:

```text
C0 = KMeans(Z0)
```

The fixed Student-t prior assignment is:

```text
Q0_ik =
  (1 + ||z0_i - c0_k||^2 / alpha)^(-(alpha + 1) / 2)
  /
  sum_l (1 + ||z0_i - c0_l||^2 / alpha)^(-(alpha + 1) / 2)
```

Using the TPPR-cut graph degree distribution:

```text
degree_prob_i = d_pi(i) / sum_j d_pi(j)
f0_k = sum_i degree_prob_i * Q0_ik
```

The fixed prior target is:

```text
P0_ik =
  (Q0_ik^2 / (f0_k + eps))
  /
  sum_l (Q0_il^2 / (f0_l + eps))
```

Training uses the current assignment `S` only on the prediction side:

```text
R_pi(S) = KL(P0 || S)
        = sum_i degree_prob_i * sum_k P0_ik * log((P0_ik + eps) / (S_ik + eps))
```

For mini-batch `b` in epoch `e`, with `B` mini-batches:

```text
L^(b,e) =
  lambda_temp * L_temp^(b)
  + lambda_batch * L_batch^(b)
  + (lambda_com^(e) / B) * L_com
```

For `cut_main`, the default main prediction is:

```text
y_hat_i = argmax_k S_ik
```

Users can still explicitly set `--main_pred_mode kmeans_z`, `kmeans_s`, `spectral_pi`, or `spectral_topk_pi` for diagnostics or comparison.

## Z0 Usage

The implementation reuses the existing initial feature path:

- `self.feature = self.data.get_feature().astype(np.float32)` is the existing pretrained initial node feature matrix `Z0`.
- `self.node_emb` is initialized from `Z0`.
- In prototype assignment mode, `self.cluster_prototypes` is initialized by KMeans on `Z0`.
- The new fixed prior target `P0` is also built from `Z0` and KMeans centers on `Z0`.

KMeans remains useful for initialization and diagnostic outputs, but it is no longer the default final output for `cut_main`.

## Suggested Experiments

Start with `school`:

```bash
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 100 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 1.0 \
  --rho_assign 0.1 \
  --warmup_epochs 20 \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --run_tag fix_prior_scale_argmax
```

Compare `argmax_s` against `kmeans_z` before and after the change. In the metrics CSV, focus on:

- `acc_argmax_s`
- `nmi_argmax_s`
- `ari_argmax_s`
- `f1_argmax_s`
- `weighted_com`
- `weighted_temp`
- `weighted_batch`

If the community signal is too weak after division by `B`, increase `lambda_com` rather than reusing the old repeated full-graph scaling:

```bash
for lc in 1.0 2.0 5.0 10.0
do
  python main.py \
    --dataset school \
    --objective_mode cut_main \
    --epoch 100 \
    --lambda_temp 0.01 \
    --lambda_batch 0.01 \
    --lambda_com ${lc} \
    --rho_assign 0.1 \
    --warmup_epochs 20 \
    --eval_interval 5 \
    --grad_eval_interval 5 \
    --run_tag fix_prior_scale_lc_${lc}
done
```

## Validation notes

- `python -m py_compile main.py trainer.py data_load.py sparsification.py clustering_utils.py utils.py` passed.
- Local smoke testing was not completed on the workstation. The first attempt with default workers failed before training due to Windows DataLoader multiprocessing pipe permission error: `PermissionError: [WinError 5]`. A local retry was intentionally stopped because the workstation should not run the training workload. Run the smoke test on the target GPU server instead; the default `--device auto` will use CUDA when available:

```bash
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 1 \
  --batch_size 1024 \
  --eval_interval 1 \
  --grad_eval_interval 1 \
  --run_tag smoke_fix
```
