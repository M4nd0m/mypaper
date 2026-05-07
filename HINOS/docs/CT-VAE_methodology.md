# CT-VAE / CT-CAVAE Methodology

## 1. Problem Formulation

A continuous dynamic network is represented as an event-stream graph:

$$
G=\{V,E,T\},
$$

where $V$ is the node set, $E$ is the interaction set, and $T$ is the timestamp domain. The interactions between two nodes $u$ and $v$ across different time periods are stored as timestamped triples:

$$
T_{u,v}:=\{(u,v,t_1),(u,v,t_2),\ldots,(u,v,t_T)\}.
$$

Given current time $t$, the historical neighbor-event sequence of node $u$ is denoted as:

$$
N_t(u):=\{(v,t_1),(s,t_2),\ldots,(v,t)\}.
$$

The community detection objective is to partition all nodes into $k$ disjoint communities:

$$
C=\{c_1,c_2,\ldots,c_k\}.
$$

The task is unsupervised. The model learns node representations from the continuous-time event stream and uses them to infer community structure.

## 2. Overall Method Structure

CT-VAE extends the variational autoencoder framework to continuous dynamic networks. The method keeps the VAE-style structure of inference, generation, and ELBO, but changes how each part is defined.

For continuous dynamic networks, the input is not an adjacency matrix. It is an ordered event stream. Therefore, CT-VAE does not use a GCN encoder to infer a Gaussian posterior from an adjacency matrix. Instead, it uses a Hawkes-process-based inference component to directly optimize node representations:

$$
Z\in\mathbb{R}^{|V|\times d}.
$$

The generation component also does not reconstruct the full adjacency matrix. It reconstructs a community-compact relation structure through pseudo labels and cosine similarity constraints.

CT-CAVAE is the scalable community-aware variant of CT-VAE. It keeps the same generation component as CT-VAE, but introduces a Gaussian mixture distribution over communities so that community detection can be integrated into the VAE objective.

## 3. Inference Component of CT-VAE and CT-CAVAE

The inference component models the probability of temporal interactions through a Hawkes process. For two nodes $u$ and $v$ interacting at time $t$, the conditional intensity is defined as:

$$
\lambda_{u,v}(t)
=
\mu_{u,v}
+
\sum_{v',t'\in N_t(v)}
\alpha_{u,v'}(t)\mu_{u,v'}\kappa(t-t')
+
\sum_{u',t'\in N_t(u)}
\alpha_{v,u'}(t)\mu_{v,u'}\kappa(t-t').
$$

The base intensity is defined by the negative squared distance between node representations:

$$
\mu_{u,v}=-\lVert z_u-z_v\rVert^2.
$$

The temporal decay kernel is:

$$
\kappa(t-t')=\exp(-\delta(t-t')),
$$

where $\delta$ is a learnable temporal decay parameter. A larger time interval produces a smaller historical influence.

The normalized causal weight is:

$$
\alpha_{u,v'}(t)
=
\frac{\exp(\mu_{u,v'})}{\sum_{v\in N_t(u)}\exp(\mu_{u,v})}.
$$

This term measures the relative influence of each historical interaction on the current interaction.

The model learns $Z$ by maximizing the log-likelihood of observed temporal interactions and suppressing sampled negative interactions:

$$
q(Z\mid E,T)
:=
\sum_{u\in V}\log q(Z\mid u,N_u(t))
=
\sum_{u\in V}\sum_{v\in N_t(u)}
\left[
\log\phi(\lambda_{u,v}(t))
-
\sum_{v'\in P(u)}\log\phi(\lambda_{u,v'}(t))
\right],
$$

where $P(u)\propto |N(u)|$ is the negative sampling distribution and $\phi(\cdot)$ is the sigmoid function.

In this design, the Hawkes process replaces the adjacency-matrix-based GCN inference component. The posterior is not assumed to be a Gaussian distribution parameterized by an encoder. The optimization target shifts from learning Gaussian posterior parameters to directly optimizing the node representation matrix $Z$.

## 4. Generation Component of CT-VAE and CT-CAVAE

The generation component reconstructs a relation structure from learned node representations. Since the input is an event stream rather than an adjacency matrix, CT-VAE reconstructs local positive and negative relations at the batch/event level.

Given an interaction triplet $(u,v,t)$, historical neighbor node $h\in N_t(u)$, and negative sampled node $v'\in P(u)$, a relaxed reconstruction objective can be written as:

$$
\min
\left\{
|1-\cos(Z_u,Z_v)|
+
|1-\cos(Z_u,Z_h)|
+
|0-\cos(Z_u,Z_{v'})|
\right\}.
$$

The cosine similarity is:

$$
\cos(Z_u,Z_v)=
\frac{Z_uZ_v^{\top}}{\lVert Z_u\rVert\cdot\lVert Z_v\rVert}.
$$

CT-VAE further introduces a compact community constraint through a pseudo-label matrix $X$. The generation objective becomes:

$$
\min
\left\{
|X_{u,v}-\cos(Z_u,Z_v)|
+
|X_{u,h}-\cos(Z_u,Z_h)|
+
|0-\cos(Z_u,Z_{v'})|
\right\}.
$$

For nodes $u$ and $v$, let $c(u)$ and $c(v)$ be their community membership vectors at the current epoch. The pseudo label is defined as:

$$
X_{u,v}=1
\quad\Longleftrightarrow\quad
c(u)c(v)^{\top}=1,
$$

and otherwise:

$$
X_{u,v}=0.
$$

Thus, CT-VAE reconstructs an edge between two nodes only when two conditions are jointly satisfied:

1. their representations are similar;
2. they are assigned to the same pseudo community.

This compact constraint makes the generation component community-aware. It avoids reconstructing every observed interaction and reduces the effect of hub-mediated or boundary-crossing interactions on the reconstructed community structure.

## 5. ELBO of CT-VAE

CT-VAE combines the compact generation objective and the Hawkes-based inference objective in an ELBO-style objective:

$$
\mathcal{L}
=
-
\sum_{u,v,t\in T\atop h\in N_t(u),\ v'\in P(u)}
\left\{
|X_{u,v}-\cos(Z_u,Z_v)|
+
|X_{u,h}-\cos(Z_u,Z_h)|
+
|0-\cos(Z_u,Z_{v'})|
\right\}
-
KL(q(Z\mid E,T)\Vert p(Z)).
$$

The prior $p(Z)$ is not chosen as a standard Gaussian prior. Since $q(Z\mid E,T)$ is not modeled as a Gaussian posterior, CT-VAE introduces a Student’s $t$-distribution prior over community centers.

For node $u$ and the $k$-th community, the prior is:

$$
p(Z_u)
=
\frac{
\left(1+\lVert Z_u-\theta_k\rVert^2/\varepsilon\right)^{-\frac{\varepsilon+1}{2}}
}{
\sum_{c_i\in C}
\left(1+\lVert Z_u-\theta_{c_i}\rVert^2/\varepsilon\right)^{-\frac{\varepsilon+1}{2}}
}.
$$

Here, $\theta_k$ is the cluster center of the $k$-th community initialized by K-means, and $\varepsilon$ is the degree of freedom. The default value is:

$$
\varepsilon=1.
$$

The KL term encourages the learned node representation to remain close to the corresponding Student’s $t$-distribution center. This prior is used to support community structure discovery and to reduce the sensitivity of the representation space to outliers.

## 6. Scalable Variant: CT-CAVAE

CT-CAVAE extends CT-VAE by introducing a Gaussian mixture distribution over communities. The generation component remains unchanged. The inference component is modified by adding a community variable $C$:

$$
q(Z,C\mid E,T)=q(Z\mid E,T)p(C\mid Z).
$$

The community posterior is defined through a Gaussian mixture form:

$$
p(C\mid Z)=
\frac{p(Z\mid C)p(C)}{\sum_k p(Z\mid C_k)p(C_k)}.
$$

The conditional distribution of representations under a community component is:

$$
p(Z\mid C)\sim\mathcal{N}(Z\mid\mu_c,\sigma_c^2),
$$

and the community prior is categorical:

$$
p(C)\sim Cat(\pi),
\qquad
\sum_k\pi_k=1.
$$

The CT-CAVAE objective is:

$$
\mathcal{L}
=
\mathbb{E}_{(Z,C)\sim q(Z,C\mid E,T)}
\log
\frac{p(Z,C,T)}{q(Z,C\mid E,T)}.
$$

Expanded with the CT-VAE reconstruction term, the objective is:

$$
\mathcal{L}
=
-
\sum_{u,v,t\in T\atop h\in N_t(u),\ v'\in P(u)}
\left\{
|X_{u,v}-\cos(Z_u,Z_v)|
+
|X_{u,h}-\cos(Z_u,Z_h)|
+
|0-\cos(Z_u,Z_{v'})|
\right\}
-
KL(q(Z\mid E,T)\Vert p(Z))
-
KL(q(C\mid Z)\Vert p(C\mid Z)).
$$

The additional KL divergence aligns two community distributions:

- $p(C\mid Z)$: the community distribution induced by the Gaussian mixture model;
- $q(C\mid Z)$: the downstream-style community assignment distribution.

The downstream-style community assignment distribution is defined as:

$$
q(C\mid Z)
=
\frac{(1+\lVert Z-W\rVert^2)^{-1}}
{\sum_{c_i\in C}(1+\lVert Z-W_{c_i}\rVert^2)^{-1}}.
$$

Here, $W$ is a learnable cluster center matrix initialized by K-means.

By maximizing the CT-CAVAE objective, the term

$$
KL(q(C\mid Z)\Vert p(C\mid Z))
$$

is minimized, so the community structure obtained by the downstream assignment distribution becomes consistent with the community structure induced by the end-to-end Gaussian mixture distribution.

## 7. Method Procedure

1. Represent the continuous dynamic network as an event stream of triples $(u,v,t)$.
2. Initialize node representations $Z$ and cluster centers using K-means-based initialization.
3. For each observed temporal interaction, compute the Hawkes conditional intensity $\lambda_{u,v}(t)$ from the current node representations and historical neighbor events.
4. Optimize the Hawkes-based inference objective with negative sampling.
5. Construct pseudo labels $X$ from current community assignments.
6. Optimize the compact generation objective using cosine similarity between positive pairs, historical neighbor pairs, and negative sampled pairs.
7. For CT-VAE, optimize the ELBO with the Student’s $t$-distribution prior and obtain final communities through K-means.
8. For CT-CAVAE, add the Gaussian mixture community distribution and optimize the extra community-alignment KL term to support end-to-end community detection.
