# hinos_ncut_offline

This project keeps only the offline training pipeline from the original HINOS codebase.

Kept:
- sqrt(static_edges) TAPS sampling
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

Run one dataset:

```bash
python main.py --dataset school --epoch 30 --lambda_community 0.1
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
