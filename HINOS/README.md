# hinos_ncut_offline

This project keeps only the offline training pipeline from the original HINOS codebase.

Kept:
- adaptive TAPS sampling
- TPPR construction and caching
- reconstruction loss
- temporal loss
- full-graph NCut loss
- offline embedding optimization
- offline cluster assignment export

Removed:
- query-aware community search
- BFS candidate generation
- top-k evaluation
- search-specific scoring arguments
- F1-based early stopping

## Current Loss Function

The current formal objective is:

\[
\mathcal{L}
=
\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
+
\lambda_{\mathrm{com}}\mathcal{L}_{\mathrm{com}}
+
\lambda_{\mathrm{batch}}\mathcal{L}_{\mathrm{batch}}.
\]

\[
\mathcal{L}_{\mathrm{com}}
=
\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
+
\rho_{\mathrm{assign}}\mathcal{R}_{\Pi}(S).
\]

TPPR + Cut is the core community-aware objective in this project. By default, the relaxed assignment \(S\) is computed by prototype-based Student-t assignment initialized from pretrained node2vec embeddings.

See [docs/loss_function_design.md](docs/loss_function_design.md) for the full design and implementation mapping.

## TAPS Budget

`--taps_budget_mode sqrt_edges` keeps the original compatibility behavior:

\[
N_{\mathrm{TAPS}}=\lceil\sqrt{|\mathcal{E}_{\mathrm{static}}|}\rceil .
\]

For full-graph TPPR-Cut clustering, the recommended mode is `nlogn`:

\[
N_{\mathrm{TAPS}}
=
\left\lceil
\beta |\mathcal{V}|\log(|\mathcal{V}|+1)
\right\rceil ,
\]

where \(\beta\) is `--taps_budget_beta`. This is graph-adaptive, not a fixed sample count, and better matches the \(O(n\log n)\) scale commonly used for cut/spectral structure preservation. For School, `--taps_budget_beta 0.5` gives about 947 sampled paths before ceiling, with `computed_N_TAPS=948`.

Run one dataset:

```bash
python main.py --dataset school --epoch 30 --lambda_community 0.1
```

Recommended School TPPR-Cut run:

```bash
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 40 \
  --assign_mode prototype \
  --prototype_alpha 1.0 \
  --lambda_temp 0.01 \
  --lambda_com 1.0 \
  --rho_assign 0.1 \
  --lambda_batch 0.01 \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.5 \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --main_pred_mode kmeans_z \
  --run_tag proto_kl_ramp_taps_nlogn_b05
```

If `node2label.txt` is unavailable or you do not want to rely on it for inferring the cluster count, pass `--num_clusters` explicitly:

```bash
python main.py --dataset school --epoch 30 --num_clusters 5
```

Outputs:
- learned embeddings: `emb/<dataset>/<dataset>_TGC_<epoch>.emb`
- hard cluster predictions: `emb/<dataset>/<dataset>_TGC_<epoch>_pred.txt`
- soft assignment matrix: `emb/<dataset>/<dataset>_TGC_<epoch>_soft_assign.npy`

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

Write evaluation results to CSV:

```bash
python eval_to_csv.py --datasets school --epoch 30 --csv_path results/evaluation_results.csv
```

Evaluate several datasets and append to the same CSV:

```bash
python eval_to_csv.py --datasets school dblp brain patent arXivAI arXivCS --epoch 100 --csv_path results/evaluation_results.csv --append --skip_missing
```

This is suitable for remote server runs because it writes a plain CSV file under the project directory by default:
`results/evaluation_results.csv`.

Run the default dataset sweep:

```bash
bash run_full.sh
```

The sweep script now runs `evaluate.py` after each dataset by default. To skip evaluation:

```bash
RUN_EVAL=0 bash run_full.sh
```

By default, `data_root`, `pretrain_emb_dir`, `cache_dir`, and `emb_root` all point to folders inside this project.
