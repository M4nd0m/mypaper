# DBLP High-Order Diagnostics

## Run Parameters

- `cache_dir`: `/home/cx/code/mypaper/HINOS/cache`
- `cross_topk`: `10`
- `data_root`: `/home/cx/code/mypaper/HINOS/dataset`
- `dataset`: `dblp`
- `knn_metric`: `cosine`
- `max_pairs_per_middle`: `5000`
- `output_dir`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03`
- `pretrain_emb_dir`: `/home/cx/code/mypaper/HINOS/pretrain`
- `seed`: `42`
- `taps_T_cap`: `10`
- `taps_alpha`: `0.2`
- `taps_budget_beta`: `0.3`
- `taps_budget_mode`: `nlogn`
- `taps_rng_seed`: `42`
- `taps_tau_eps`: `1.0`
- `topk_values`: `5,10,20`
- `tppr_K`: `3`
- `tppr_alpha`: `0.2`

## Output Files

- `edge_label_purity_by_multiplicity.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/edge_label_purity_by_multiplicity.csv`
- `second_order_path_purity_by_middle_degree.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/second_order_path_purity_by_middle_degree.csv`
- `tppr_directed_vs_symmetric_purity.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/tppr_directed_vs_symmetric_purity.csv`
- `cross_label_top_tppr_pairs.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/cross_label_top_tppr_pairs.csv`
- `cross_label_common_neighbor_frequency.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/cross_label_common_neighbor_frequency.csv`
- `cross_label_common_neighbor_degree_summary.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/cross_label_common_neighbor_degree_summary.csv`
- `neighborhood_purity_comparison.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/neighborhood_purity_comparison.csv`
- `summary.md`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta03/summary.md`

## Sampling Strategy

- Second-order paths enumerate all neighbor pairs when a middle node has at most `5000` pairs; otherwise exactly `5000` neighbor pairs are sampled with the fixed random seed.
- Middle nodes enumerated exactly: `27606`
- Middle nodes sampled: `52`
- Static one-hop neighbors use deterministic sorted neighbor order.
- Static PPR is skipped because this repository does not contain a reusable static PPR utility and this diagnostic avoids adding a new approximate PPR implementation.

## Core Result Tables

### Edge Label Purity By Multiplicity

| bucket | edge_count | same_label_edge_count | purity |
| --- | --- | --- | --- |
| count=1 | 108612 | 67023 | 0.617087 |
| count=2-3 | 32608 | 20443 | 0.626932 |
| count=4-10 | 8791 | 5529 | 0.628939 |
| count>10 | 557 | 369 | 0.662478 |
| overall | 150568 | 93364 | 0.620079 |

### Second-Order Path Purity By Middle Degree

| middle_degree_bucket | path_count | same_label_endpoint_count | purity |
| --- | --- | --- | --- |
| degree<=5 | 44839 | 26572 | 0.592609 |
| degree=6-10 | 285770 | 177040 | 0.619519 |
| degree=11-20 | 523552 | 288221 | 0.550511 |
| degree=21-50 | 1199972 | 592159 | 0.493477 |
| degree>50 | 1095465 | 450826 | 0.411538 |
| overall | 3149598 | 1534818 | 0.487306 |

### Directed TPPR Vs Symmetric TPPR Purity

| affinity_type | K | valid_node_count | purity_at_k | leakage_at_k |
| --- | --- | --- | --- | --- |
| directed_pi | 5 | 26342 | 0.610096 | 0.389904 |
| directed_pi | 10 | 26342 | 0.573461 | 0.426539 |
| directed_pi | 20 | 26342 | 0.530260 | 0.469740 |
| symmetrized_pi | 5 | 26342 | 0.619260 | 0.380740 |
| symmetrized_pi | 10 | 26342 | 0.579440 | 0.420560 |
| symmetrized_pi | 20 | 26342 | 0.535358 | 0.464642 |

### Cross-Label Common Neighbor Degree Summary

| degree_bucket | pair_count_covered | total_common_neighbor_occurrences |
| --- | --- | --- |
| degree<=5 | 8584 | 10415 |
| degree=6-10 | 30220 | 58624 |
| degree=11-20 | 38563 | 65104 |
| degree=21-50 | 43856 | 73268 |
| degree>50 | 24873 | 33762 |
| no_common_neighbor | 25244 | 0 |

### Top Cross-Label Common Neighbor Frequency

| common_neighbor | label | degree | frequency_in_cross_label_pairs |
| --- | --- | --- | --- |
| 992 | 2 | 312 | 591 |
| 3566 | 2 | 276 | 396 |
| 13 | 0 | 182 | 323 |
| 5388 | 2 | 231 | 314 |
| 2792 | 0 | 220 | 301 |
| 10202 | 5 | 153 | 263 |
| 485 | 0 | 98 | 253 |
| 20855 | 6 | 171 | 243 |
| 2691 | 0 | 130 | 236 |
| 456 | 0 | 195 | 225 |
| 1228 | 2 | 118 | 222 |
| 5236 | 5 | 154 | 219 |
| 1069 | 0 | 112 | 216 |
| 3493 | 0 | 152 | 212 |
| 4878 | 2 | 127 | 196 |
| 1531 | 0 | 103 | 187 |
| 1171 | 2 | 111 | 183 |
| 6311 | 2 | 98 | 176 |
| 230 | 0 | 92 | 175 |
| 3291 | 0 | 108 | 168 |

_Showing first 20 of 17824 rows._

### Cross-Label Top TPPR Pairs

| source | target | source_label | target_label | tppr_weight | num_common_neighbors |
| --- | --- | --- | --- | --- | --- |
| 0 | 1322 | 0 | 2 | 0.059412 | 2 |
| 0 | 27378 | 0 | 3 | 0.039308 | 1 |
| 0 | 1241 | 0 | 5 | 0.032055 | 2 |
| 0 | 27342 | 0 | 3 | 0.022968 | 1 |
| 0 | 26978 | 0 | 3 | 0.018762 | 3 |
| 1 | 22921 | 0 | 6 | 0.074815 | 1 |
| 1 | 8809 | 0 | 2 | 0.013210 | 0 |
| 1 | 1322 | 0 | 2 | 0.013074 | 3 |
| 1 | 14508 | 0 | 4 | 0.011841 | 1 |
| 1 | 20970 | 0 | 6 | 0.011523 | 1 |
| 1 | 20901 | 0 | 6 | 0.011307 | 1 |
| 1 | 18303 | 0 | 6 | 0.010633 | 0 |
| 2 | 12744 | 0 | 4 | 0.049871 | 3 |
| 2 | 13148 | 0 | 4 | 0.039402 | 2 |
| 2 | 27483 | 0 | 7 | 0.037305 | 6 |
| 2 | 25340 | 0 | 7 | 0.037151 | 2 |
| 3 | 6146 | 0 | 1 | 0.052667 | 0 |
| 3 | 5941 | 0 | 1 | 0.050424 | 8 |
| 3 | 10615 | 0 | 1 | 0.048131 | 1 |
| 3 | 6475 | 0 | 1 | 0.048078 | 1 |

_Showing first 20 of 109920 rows._

### Neighborhood Purity Comparison

| neighborhood_type | K | valid_node_count | purity_at_k | leakage_at_k |
| --- | --- | --- | --- | --- |
| static_onehop | 5 | 28085 | 0.638720 | 0.361280 |
| static_onehop | 10 | 28085 | 0.664238 | 0.335762 |
| static_onehop | 20 | 28085 | 0.671600 | 0.328400 |
| temporal_tppr | 5 | 26342 | 0.619260 | 0.380740 |
| temporal_tppr | 10 | 26342 | 0.579440 | 0.420560 |
| temporal_tppr | 20 | 26342 | 0.535358 | 0.464642 |
| node2vec_knn | 5 | 28085 | 0.705558 | 0.294442 |
| node2vec_knn | 10 | 28085 | 0.655852 | 0.344148 |
| node2vec_knn | 20 | 28085 | 0.604289 | 0.395711 |

