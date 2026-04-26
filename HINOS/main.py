import argparse
import os

from trainer import TGCTrainer
from utils import resolve_path, set_random_seed


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Offline temporal community representation learning with sqrt-TAPS, TPPR, and full-graph NCut."
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
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")
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

    parser.add_argument("--objective_mode", type=str, default="original", choices=["original", "cut_main"])
    parser.add_argument("--lambda_com", type=float, default=None, help="community-aware loss weight")
    parser.add_argument(
        "--lambda_community",
        "--lambda_cut",
        dest="lambda_community",
        type=float,
        default=None,
        help="legacy alias for --lambda_com",
    )
    parser.add_argument(
        "--rho_assign",
        type=float,
        default=0.1,
        help="TPPR-aware assignment penalty weight inside L_com",
    )
    parser.add_argument(
        "--lambda_temp",
        type=float,
        default=None,
        help="legacy temporal loss weight; not part of the canonical cut_main objective",
    )
    parser.add_argument("--lambda_batch", type=float, default=None, help="batch reconstruction loss weight")
    parser.add_argument("--lambda_bal", type=float, default=None, help="degree-aware balance penalty weight")
    parser.add_argument("--lambda_ncut_orth", type=float, default=None, help="legacy alias for balance weight")
    parser.add_argument("--cluster_hidden_dim", type=int, default=64, help="cluster MLP hidden dimension")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs before cut_main")
    parser.add_argument("--eval_interval", type=int, default=0, help="evaluation interval; 0 means final epoch only")
    parser.add_argument("--grad_eval_interval", type=int, default=0, help="gradient diagnostic interval; 0 follows eval_interval")
    parser.add_argument("--spectral_topk", type=int, default=20, help="top-k sparsification for spectral Pi diagnostics")
    parser.add_argument("--run_tag", type=str, default="", help="optional suffix for output filenames")
    return parser.parse_args()


def apply_objective_defaults(args):
    if args.lambda_bal is None and args.lambda_ncut_orth is not None:
        args.lambda_bal = args.lambda_ncut_orth
    if args.lambda_com is None and args.lambda_community is not None:
        args.lambda_com = args.lambda_community
    if args.lambda_community is None and args.lambda_com is not None:
        args.lambda_community = args.lambda_com

    if args.objective_mode == "cut_main":
        if args.lambda_com is None:
            args.lambda_com = 1.0
        args.lambda_community = args.lambda_com
        if args.lambda_temp is None:
            args.lambda_temp = 1.0
        if args.lambda_batch is None:
            args.lambda_batch = 0.01
        if args.lambda_bal is None:
            args.lambda_bal = 0.0
    else:
        if args.lambda_com is None:
            args.lambda_com = 0.1
        args.lambda_community = args.lambda_com
        if args.lambda_temp is None:
            args.lambda_temp = 1.0
        if args.lambda_batch is None:
            args.lambda_batch = 1.0
        if args.lambda_bal is None:
            args.lambda_bal = 0.0
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
    print(f"[RUN] warmup_epochs      = {args.warmup_epochs}")
    print(f"[RUN] eval_interval      = {args.eval_interval if args.eval_interval > 0 else 'final-only'}")
    print(f"[Graph] directed         = {bool(args.directed)}")
    print(f"[Device] device          = {args.device}")
    print(f"[TPPR] alpha={args.tppr_alpha}, K={args.tppr_K}")
    print(
        f"[TAPS] alpha={args.taps_alpha}, N=sqrt(static_edges), "
        f"tau_eps={args.taps_tau_eps}, T_cap={args.taps_T_cap}"
    )
    print(
        f"[Loss] lambda_com={args.lambda_com}, rho_assign={args.rho_assign}, "
        f"lambda_batch={args.lambda_batch}, legacy_lambda_temp={args.lambda_temp}, "
        f"legacy_lambda_bal={args.lambda_bal}"
    )
    print(f"[NCut] hidden_dim={args.cluster_hidden_dim}, spectral_topk={args.spectral_topk}, scope=full")
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
    print(f"EmbeddingPath={summary['embedding_path']}")
    print(f"PredictionPath={summary['prediction_path']}")
    print(f"SoftAssignmentPath={summary['soft_assignment_path']}")
    print(f"PiPath={summary['pi_path']}")
    print(f"PiRawPath={summary['pi_raw_path']}")
    print(f"PiCutPath={summary['pi_cut_path']}")
    print(f"MetricsCsvPath={summary['metrics_csv_path']}")


if __name__ == "__main__":
    main()
