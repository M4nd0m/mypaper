# Archived Degree-Aware TPPR-Cut Attempts

The executable training path currently uses only raw TPPR normalized cut:

\[
\mathcal{L}_{TPPR\text{-}Cut}
=
\operatorname{Tr}
\left[
(Q^\top D_\Pi Q+\epsilon I)^{-1}
Q^\top(D_\Pi-W_\Pi)Q
\right],
\]

where:

\[
W_\Pi=\frac{1}{2}(\Pi+\Pi^\top),\quad W_{ii}=0.
\]

Student-t assignment, DTGC-KL, HINOS-Bal, temporal loss, batch reconstruction, TPPR/TAPS cache construction, and final prediction are unchanged.

## Full-Null Degree Correction

The removed full-null correction used:

\[
L_\Pi^{dc}
=
D_\Pi-W_\Pi
+
\gamma
\left(
\frac{d^\Pi(d^\Pi)^\top}{2m_\Pi}
-
D_\Pi
\right).
\]

For row-stochastic \(Q\), with:

\[
G=Q^\top D_\Pi Q,\quad
v=Q^\top d^\Pi,\quad
E=\frac{vv^\top}{2m_\Pi},
\]

we have:

\[
G\mathbf{1}=v,\quad v^\top\mathbf{1}=2m_\Pi.
\]

Therefore:

\[
\operatorname{Tr}(G^{-1}E)=1,
\quad
\operatorname{Tr}(G^{-1}G)=K,
\]

and the correction trace is:

\[
\operatorname{Tr}[G^{-1}(E-G)]=1-K.
\]

Thus:

\[
\mathcal{L}_{dc}
=
\mathcal{L}_{raw}
+
\gamma(1-K).
\]

This changes the numeric loss value but not the optimization direction.

## Edge-Level Residual Attempt

The edge-level residual experiment used:

\[
w^{res}_{ij}
=
\max\left(
w_{ij}
-
\gamma\frac{d_i^\Pi d_j^\Pi}{2m_\Pi+\epsilon},
0
\right),
\]

then recomputed a normalized cut on \(W_{res}\).

DBLP ablations with \(K=3\), \(\beta=0.3\), and \(\gamma\in\{0.025,0.05\}\) showed that the residual variant lowered the cut value but did not materially change `nmi_argmax_s`, assignment entropy, KL, or HINOS balance compared with raw NCut. It mainly removed many low-weight edges while leaving the dominant TPPR mass structure nearly unchanged.

This variant is therefore removed from the executable training path.

## Current Status

- Raw TPPR-Cut is the only training cut objective.
- `loss_tppr_cut_raw` remains in metrics and equals `loss_tppr_cut`.
- Degree-aware alternatives should be treated as archived ablations unless a new design changes the actual assignment trajectory, not just the cut value.
