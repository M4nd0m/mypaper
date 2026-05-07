# DBLP High-Order Diagnostics

## Run Parameters

- `cache_dir`: `/home/cx/code/mypaper/HINOS/cache`
- `cross_topk`: `10`
- `data_root`: `/home/cx/code/mypaper/HINOS/dataset`
- `dataset`: `dblp`
- `knn_metric`: `cosine`
- `max_pairs_per_middle`: `5000`
- `output_dir`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02`
- `pretrain_emb_dir`: `/home/cx/code/mypaper/HINOS/pretrain`
- `seed`: `42`
- `taps_T_cap`: `10`
- `taps_alpha`: `0.2`
- `taps_budget_beta`: `0.2`
- `taps_budget_mode`: `nlogn`
- `taps_rng_seed`: `42`
- `taps_tau_eps`: `1.0`
- `topk_values`: `5,10,20`
- `tppr_K`: `3`
- `tppr_alpha`: `0.2`

## Output Files

- `edge_label_purity_by_multiplicity.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/edge_label_purity_by_multiplicity.csv`
- `second_order_path_purity_by_middle_degree.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/second_order_path_purity_by_middle_degree.csv`
- `tppr_directed_vs_symmetric_purity.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/tppr_directed_vs_symmetric_purity.csv`
- `cross_label_top_tppr_pairs.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/cross_label_top_tppr_pairs.csv`
- `cross_label_common_neighbor_frequency.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/cross_label_common_neighbor_frequency.csv`
- `cross_label_common_neighbor_degree_summary.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/cross_label_common_neighbor_degree_summary.csv`
- `neighborhood_purity_comparison.csv`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/neighborhood_purity_comparison.csv`
- `summary.md`: `/home/cx/code/mypaper/HINOS/diagnostics/dblp_high_order/K3_beta02/summary.md`

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
| directed_pi | 5 | 24681 | 0.597196 | 0.402804 |
| directed_pi | 10 | 24681 | 0.560181 | 0.439819 |
| directed_pi | 20 | 24681 | 0.520431 | 0.479569 |
| symmetrized_pi | 5 | 24681 | 0.604375 | 0.395625 |
| symmetrized_pi | 10 | 24681 | 0.564954 | 0.435046 |
| symmetrized_pi | 20 | 24681 | 0.524682 | 0.475318 |

### Cross-Label Common Neighbor Degree Summary

| degree_bucket | pair_count_covered | total_common_neighbor_occurrences |
| --- | --- | --- |
| degree<=5 | 6659 | 8137 |
| degree=6-10 | 24525 | 48604 |
| degree=11-20 | 33167 | 57297 |
| degree=21-50 | 39982 | 68389 |
| degree>50 | 22979 | 31411 |
| no_common_neighbor | 30447 | 0 |

### Top Cross-Label Common Neighbor Frequency

| common_neighbor | label | degree | frequency_in_cross_label_pairs |
| --- | --- | --- | --- |
| 992 | 2 | 312 | 595 |
| 3566 | 2 | 276 | 370 |
| 5388 | 2 | 231 | 299 |
| 13 | 0 | 182 | 269 |
| 10202 | 5 | 153 | 265 |
| 485 | 0 | 98 | 260 |
| 2792 | 0 | 220 | 260 |
| 3493 | 0 | 152 | 234 |
| 2691 | 0 | 130 | 232 |
| 456 | 0 | 195 | 221 |
| 1069 | 0 | 112 | 215 |
| 4878 | 2 | 127 | 210 |
| 20855 | 6 | 171 | 203 |
| 1228 | 2 | 118 | 187 |
| 5236 | 5 | 154 | 185 |
| 1531 | 0 | 103 | 172 |
| 6311 | 2 | 98 | 168 |
| 806 | 0 | 129 | 161 |
| 3246 | 0 | 142 | 159 |
| 5322 | 0 | 111 | 159 |

_Showing first 20 of 16861 rows._

### Cross-Label Top TPPR Pairs

| source | target | source_label | target_label | tppr_weight | num_common_neighbors |
| --- | --- | --- | --- | --- | --- |
| 0 | 1322 | 0 | 2 | 0.073125 | 2 |
| 0 | 27342 | 0 | 3 | 0.042881 | 1 |
| 0 | 269 | 0 | 3 | 0.033225 | 1 |
| 0 | 22921 | 0 | 6 | 0.024268 | 1 |
| 0 | 1236 | 0 | 4 | 0.013089 | 7 |
| 0 | 23875 | 0 | 2 | 0.011948 | 4 |
| 1 | 22921 | 0 | 6 | 0.083888 | 1 |
| 1 | 8809 | 0 | 2 | 0.019753 | 0 |
| 1 | 1322 | 0 | 2 | 0.018809 | 3 |
| 1 | 14508 | 0 | 4 | 0.015065 | 1 |
| 1 | 20970 | 0 | 6 | 0.015011 | 1 |
| 1 | 20901 | 0 | 6 | 0.014600 | 1 |
| 1 | 269 | 0 | 3 | 0.012430 | 0 |
| 2 | 13148 | 0 | 4 | 0.057183 | 2 |
| 2 | 8340 | 0 | 2 | 0.034149 | 3 |
| 2 | 14521 | 0 | 1 | 0.032946 | 2 |
| 3 | 10615 | 0 | 1 | 0.072397 | 1 |
| 3 | 5941 | 0 | 1 | 0.068730 | 8 |
| 3 | 6087 | 0 | 1 | 0.056909 | 1 |
| 3 | 5943 | 0 | 1 | 0.053391 | 8 |

_Showing first 20 of 105455 rows._

### Neighborhood Purity Comparison

| neighborhood_type | K | valid_node_count | purity_at_k | leakage_at_k |
| --- | --- | --- | --- | --- |
| static_onehop | 5 | 28085 | 0.638720 | 0.361280 |
| static_onehop | 10 | 28085 | 0.664238 | 0.335762 |
| static_onehop | 20 | 28085 | 0.671600 | 0.328400 |
| temporal_tppr | 5 | 24681 | 0.604375 | 0.395625 |
| temporal_tppr | 10 | 24681 | 0.564954 | 0.435046 |
| temporal_tppr | 20 | 24681 | 0.524682 | 0.475318 |
| node2vec_knn | 5 | 28085 | 0.705558 | 0.294442 |
| node2vec_knn | 10 | 28085 | 0.655852 | 0.344148 |
| node2vec_knn | 20 | 28085 | 0.604289 | 0.395711 |

