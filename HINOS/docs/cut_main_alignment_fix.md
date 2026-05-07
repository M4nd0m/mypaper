# cut_main alignment notes

This note records the evolution of `cut_main`.

## Current Main Method

The current main method is no longer the fixed-prior KL variant. It uses:

\[
\mathcal{L}_{\mathrm{com}}
=
\rho_{\mathrm{cut}}\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
+
\rho_{\mathrm{KL}}\mathcal{L}_{\mathrm{TGC\text{-}KL}}
+
\rho_{\mathrm{bal}}\mathcal{L}_{\mathrm{HINOS\text{-}Bal}}.
\]

The assignment matrix is the TGC-style Student-t assignment:

\[
S := Q = \operatorname{StudentT}(Z,C).
\]

The final prediction for `cut_main` remains:

\[
\hat y_i=\arg\max_k S_{ik}.
\]

## KL Target Path

The fixed-prior KL path has been removed. The main path uses dynamic TGC target refresh:

```bash
--kl_target_mode dynamic_tgc
--target_update_interval 5
```

The target is:

\[
P_{ik}
=
\frac{
S_{ik}^{2}/(f_k+\epsilon)
}{
\sum_{\ell=1}^{K}S_{i\ell}^{2}/(f_\ell+\epsilon)+\epsilon
},
\quad
f_k=\sum_i S_{ik}.
\]

The target is detached before KL is computed.

## HINOS Balance

The current method also restores the original HINOS TPPR-volume balance penalty:

\[
\mathcal{L}_{p}
=
\frac{1}{\sqrt{K}-1}
\left(
\sqrt{K}
-
\frac{1}{\sqrt{2m_\Pi}}
\sum_{j=1}^{K}
\left\|
s_j \odot d_\Pi^{1/2}
\right\|_2
\right).
\]

This penalty is query-agnostic and works for full-graph community discovery because it depends only on \(S\), \(d_\Pi\), and \(m_\Pi\).

## Scale Alignment

The full-graph community loss is still normalized by the number of mini-batches in each epoch:

\[
\mathcal{L}^{(b,e)}
=
\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}^{(b)}
+
\lambda_{\mathrm{batch}}\mathcal{L}_{\mathrm{batch}}^{(b)}
+
\frac{\lambda_{\mathrm{com}}^{(e)}}{B}\mathcal{L}_{\mathrm{com}}.
\]

This avoids applying the same full-graph term \(B\) times per epoch.

## Validation Notes

Local workstation checks should stay static:

```bash
python -m py_compile main.py trainer.py data_load.py sparsification.py clustering_utils.py evaluate.py
python main.py --help
```

Run smoke tests and experiments on the GPU server.

## DBLP Low-Cost Run

The DBLP low-cost run is available through:

```bash
bash run_dblp_fixed_prior_lowcost.sh
```

That script uses `--kl_target_mode dynamic_tgc`.
