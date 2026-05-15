import argparse
import os

from trainer import TGCTrainer
from utils import resolve_path, set_random_seed


OBJECTIVE_MODE = "search_proto"


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Offline temporal community representation learning with TAPS, TPPR, and Student-t NCut."
    )
    parser.add_argument("-d", "--dataset", type=str, default="school", help="dataset name")
    parser.add_argument("--directed", type=int, default=0, help="whether to read the graph as directed")
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=0,
        help="number of clusters; if 0, infer it from node2label.txt",
    )
    parser.add_argument("--neg_size", type=int, default=5, help="number of negative samples")
    parser.add_argument("--hist_len", type=int, default=10, help="history length H")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--neg_table_size", type=int, default=int(1e6), help="negative sampling table size")

    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"), help="dataset root")
    parser.add_argument("--emb_root", type=str, default=os.path.join(cur_dir, "emb"), help="embedding output root")
    parser.add_argument(
        "--pretrain_emb_dir",
        type=str,
        default=os.path.join(cur_dir, "pretrain"),
        help="directory containing pretrained embeddings",
    )
    parser.add_argument("--cache_dir", type=str, default=os.path.join(cur_dir, "cache"), help="TPPR/TAPS cache root")

    parser.add_argument("--tppr_alpha", type=float, default=0.2, help="TPPR alpha")
    parser.add_argument("--tppr_K", type=int, default=5, help="TPPR truncation order K")
    parser.add_argument("--taps_alpha", type=float, default=0.2, help="TAPS alpha")
    parser.add_argument("--taps_tau_eps", type=float, default=1.0, help="TAPS temporal smoothing tau_eps")
    parser.add_argument("--taps_rng_seed", type=int, default=42, help="TAPS RNG seed")
    parser.add_argument("--taps_T_cap", type=int, default=10, help="maximum TAPS time step cap; 0 means no cap")
    parser.add_argument(
        "--taps_budget_mode",
        type=str,
        default="nlogn",
        choices=["sqrt_edges", "nlogn"],
        help="TAPS sampling budget: sqrt(static_edges) for compatibility or beta*n*log(n+1)",
    )
    parser.add_argument("--taps_budget_beta", type=float, default=1.0, help="dimensionless TAPS nlogn budget scale")

    parser.add_argument(
        "--objective_mode",
        type=str,
        default=OBJECTIVE_MODE,
        choices=[OBJECTIVE_MODE],
        help="fixed objective: HINOS4search-style rec/temp/NCut with Student-t prototype assignment",
    )
    parser.add_argument("--prototype_alpha", type=float, default=1.0, help="Student-t assignment alpha")
    parser.add_argument(
        "--batch_recon_mode",
        type=str,
        default="ones",
        choices=["ones", "cebr"],
        help="batch reconstruction objective: DTGC-style ones or community-evidence CEBR",
    )
    parser.add_argument(
        "--cebr_hist_decay",
        type=float,
        default=0.1,
        help="temporal decay for CEBR history reconstruction over normalized time gaps",
    )
    parser.add_argument(
        "--freeze_prototypes",
        action="store_true",
        help="freeze prototype centers after KMeans initialization",
    )
    parser.add_argument(
        "--prototype_lr_scale",
        type=float,
        default=0.1,
        help="learning-rate multiplier for Student-t prototype centers",
    )
    parser.add_argument(
        "--main_pred_mode",
        type=str,
        default="argmax_s",
        choices=["kmeans_z", "argmax_s", "kmeans_s", "spectral_pi", "spectral_topk_pi"],
        help="prediction method exported as the main *_pred.txt file",
    )
    parser.add_argument("--lambda_batch", type=float, default=1.0, help="batch reconstruction loss weight")
    parser.add_argument("--lambda_ncut", type=float, default=0.5, help="NCut/community loss weight")
    parser.add_argument("--lambda_ncut_orth", type=float, default=5.0, help="NCut orthogonality regularization weight")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="full diagnostics interval; 0 uses every 5 epochs; ACC/NMI/ARI/F1 runs every epoch",
    )
    parser.add_argument("--spectral_topk", type=int, default=20, help="top-k sparsification for spectral Pi diagnostics")
    parser.add_argument("--run_tag", type=str, default="", help="optional suffix for output filenames")
    return parser.parse_args()


def apply_objective_defaults(args):
    return args


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.emb_root = resolve_path(cur_dir, args.emb_root)
    args.pretrain_emb_dir = resolve_path(cur_dir, args.pretrain_emb_dir)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args = apply_objective_defaults(args)

    set_random_seed(args.seed)

    print("\n==============================")
    print(f"[RUN] dataset            = {args.dataset}")
    print(f"[RUN] objective_mode     = {args.objective_mode}")
    print(f"[RUN] full_eval_interval = {args.eval_interval if args.eval_interval > 0 else 5}")
    print(f"[Graph] directed         = {bool(args.directed)}")
    print(f"[Device] device          = {args.device}")
    print(f"[TPPR] alpha={args.tppr_alpha}, K={args.tppr_K}")
    print(
        f"[TAPS] alpha={args.taps_alpha}, budget_mode={args.taps_budget_mode}, "
        f"budget_beta={args.taps_budget_beta}, tau_eps={args.taps_tau_eps}, T_cap={args.taps_T_cap}"
    )
    print(
        f"[Loss] lambda_batch={args.lambda_batch}, lambda_ncut={args.lambda_ncut}, "
        f"lambda_ncut_orth={args.lambda_ncut_orth}, "
        f"temp_weight=1.0, batch_recon={args.batch_recon_mode}, "
        f"cebr_hist_decay={args.cebr_hist_decay}, kl=off"
    )
    print(
        f"[Assign] mode=prototype, prototype_alpha={args.prototype_alpha}, "
        f"prototype_lr_scale={args.prototype_lr_scale}, freeze_prototypes={args.freeze_prototypes}, "
        f"main_pred={args.main_pred_mode}"
    )
    print(f"[NCut] spectral_topk={args.spectral_topk}, scope=fixed W_pi")
    print(f"[Cluster] num_clusters   = {args.num_clusters if args.num_clusters > 0 else 'auto-from-labels'}")
    print(f"[Path] data_root         = {args.data_root}")
    print(f"[Path] emb_root          = {args.emb_root}")
    print(f"[Path] pretrain_emb_dir  = {args.pretrain_emb_dir}")
    print(f"[Path] cache_dir         = {args.cache_dir}")
    print("==============================\n")

    trainer = TGCTrainer(args)
    summary = trainer.train()

    print("\n========== Summary ==========")
    print(f"EpochsTrained={summary['epochs_trained']}")
    print(f"BestEpoch={summary.get('best_epoch')}")
    best_eval = summary.get("best_eval") or {}
    main_pred = args.main_pred_mode
    print(
        f"BestResult[{main_pred}]: "
        f"ACC={float(best_eval.get(f'acc_{main_pred}', float('nan'))):.4f}, "
        f"NMI={float(best_eval.get(f'nmi_{main_pred}', float('nan'))):.4f}, "
        f"ARI={float(best_eval.get(f'ari_{main_pred}', float('nan'))):.4f}, "
        f"F1={float(best_eval.get(f'f1_{main_pred}', float('nan'))):.4f}"
    )
    print(f"EmbeddingPath={summary['embedding_path']}")
    print(f"PredictionPath={summary['prediction_path']}")
    print(f"SoftAssignmentPath={summary['soft_assignment_path']}")
    print(f"MetricsCsvPath={summary['metrics_csv_path']}")


if __name__ == "__main__":
    main()
