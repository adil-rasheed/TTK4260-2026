"""
t-SNE Interactive Explorer
===========================
Comprehensive interactive t-SNE teaching tool following the lecture structure.
Covers motivation, algorithm internals, pitfalls, validation, and method comparison.

Based on tSNE_Comprehensive_Lecture.ipynb
"""

import streamlit as st
import numpy as np
from time import perf_counter

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.manifold import TSNE, MDS, trustworthiness
from sklearn.decomposition import PCA
from sklearn.datasets import (make_swiss_roll, make_blobs, make_moons,
                               make_s_curve, load_digits)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

st.set_page_config(page_title="t-SNE Explorer", page_icon="🗺️", layout="wide")

# ===== STYLING =====
st.markdown("""
<style>
    .tsne-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;
    }
    .tsne-header h1 { color: white; margin: 0; }
    .tsne-header p  { color: #e0d4ff; margin: 0.5rem 0 0 0; }
    .info-panel {
        background-color: #eef2ff; border-left: 4px solid #4f46e5;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
    .warn-panel {
        background-color: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
    .math-panel {
        background-color: #f5f3ff; border-left: 4px solid #7c3aed;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;
    }
    .success-panel {
        background-color: #ecfdf5; border-left: 4px solid #10b981;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tsne-header">
    <h1>🗺️ t-SNE Interactive Explorer</h1>
    <p>t-Distributed Stochastic Neighbor Embedding — Visualize high-dimensional data by preserving local neighborhoods</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════
COLORS = {"A": "#E74C3C", "B": "#2E86C1", "C": "#27AE60",
          "D": "#F39C12", "ntnu": "#00509E", "gray": "#AAAAAA"}
PLOTLY_DISCRETE = px.colors.qualitative.G10

DATASET_OPTIONS = {
    "Swiss Roll (3D)": "swiss_roll",
    "S-Curve (3D)": "s_curve",
    "4 Gaussian Blobs (10D)": "blobs",
    "Two Moons (10D)": "moons",
    "Concentric Spheres (3D)": "concentric",
}


@st.cache_data(show_spinner=False)
def make_dataset(name="swiss_roll", n=500, noise=0.5, seed=42):
    """Generate a synthetic dataset. Returns (X, color, title)."""
    rng = np.random.RandomState(seed)
    if name == "swiss_roll":
        X, c = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
        return X, c, "Swiss Roll (3D)"
    elif name == "blobs":
        X, c = make_blobs(n_samples=n, centers=4, n_features=10,
                          cluster_std=1.5, random_state=seed)
        return X, c, "4 Gaussian Blobs (10D)"
    elif name == "moons":
        X, c = make_moons(n_samples=n, noise=0.08, random_state=seed)
        return np.hstack([X, rng.randn(n, 8) * 0.3]), c, "Two Moons (10D)"
    elif name == "s_curve":
        X, c = make_s_curve(n_samples=n, noise=noise, random_state=seed)
        return X, c, "S-Curve (3D)"
    elif name == "concentric":
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = rng.uniform(0, np.pi, n)
        label = (np.arange(n) > n // 2).astype(int)
        r = np.where(label == 0, 1.0, 3.0) + rng.randn(n) * 0.15
        X = np.column_stack([r * np.sin(phi) * np.cos(theta),
                             r * np.sin(phi) * np.sin(theta),
                             r * np.cos(phi)])
        return X, label, "Concentric Spheres (3D)"
    return np.random.randn(n, 3), np.zeros(n), "Random"


@st.cache_data(show_spinner="Running t-SNE...")
def run_tsne(X, perplexity=30, lr=200, n_iter=1000,
             init="pca", seed=42, early_exag=12.0):
    """Run t-SNE and return 2-D embedding."""
    tsne = TSNE(n_components=2,
                perplexity=min(perplexity, len(X) - 1),
                learning_rate=lr if lr != "auto" else "auto",
                max_iter=n_iter, init=init,
                random_state=seed, early_exaggeration=early_exag)
    return tsne.fit_transform(X)


def plotly_embedding(Y, color, title="", continuous=False, size=4, height=500):
    """Create a Plotly scatter of a 2-D embedding."""
    if continuous:
        fig = px.scatter(x=Y[:, 0], y=Y[:, 1], color=color,
                         color_continuous_scale="Spectral", title=title,
                         opacity=0.75)
    else:
        fig = px.scatter(x=Y[:, 0], y=Y[:, 1],
                         color=[str(c) for c in color],
                         title=title, opacity=0.75,
                         color_discrete_sequence=PLOTLY_DISCRETE)
    fig.update_layout(xaxis_title="Dim 1", yaxis_title="Dim 2",
                      height=height, template="plotly_white",
                      margin=dict(t=40, b=30))
    fig.update_traces(marker=dict(size=size))
    return fig


def is_continuous(color):
    """Check if color array is continuous (not integer labels)."""
    return not np.array_equal(color, color.astype(int))


# P-matrix helpers
def p_conditional_for_point(X, i, sigma):
    """Compute p_{j|i} for a single point with bandwidth sigma."""
    dist2 = np.sum((X - X[i]) ** 2, axis=1)
    dist2[i] = np.inf
    num = np.exp(-dist2 / (2.0 * sigma ** 2))
    denom = np.sum(num)
    if denom < 1e-30:
        return np.zeros(len(X))
    p = num / denom
    p[i] = 0.0
    return p


def perplexity_from_p(p):
    """Compute perplexity = 2^H(P) from a conditional distribution."""
    pf = p[p > 0]
    H = -np.sum(pf * np.log2(pf))
    return 2 ** H


def sigma_for_target_perplexity(X, i, target_perp, tol=1e-3, max_iter=60):
    """Binary search on log(sigma) to achieve target perplexity."""
    lo, hi = -10.0, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sigma = np.exp(mid)
        p = p_conditional_for_point(X, i, sigma)
        perp = perplexity_from_p(p)
        if abs(perp - target_perp) < tol:
            return sigma, perp
        if perp > target_perp:
            hi = mid
        else:
            lo = mid
    sigma = np.exp(0.5 * (lo + hi))
    return sigma, perplexity_from_p(p_conditional_for_point(X, i, sigma))


@st.cache_data(show_spinner="Computing P matrix...")
def compute_P_matrix(X, perplexity=30, tol=1e-3):
    """Compute the full symmetric P matrix with per-point bandwidth."""
    n = X.shape[0]
    Pcond = np.zeros((n, n), dtype=np.float64)
    sigmas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sig, _ = sigma_for_target_perplexity(X, i, target_perp=perplexity, tol=tol)
        sigmas[i] = sig
        Pcond[i] = p_conditional_for_point(X, i, sig)
    P = (Pcond + Pcond.T) / (2.0 * n)
    np.fill_diagonal(P, 0.0)
    P = np.maximum(P, 1e-12)
    P /= P.sum()
    return P, sigmas


def compute_Q_matrix(Y):
    """Student-t kernel: q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Z."""
    dist2 = pairwise_distances(Y, squared=True)
    np.fill_diagonal(dist2, np.inf)
    W = 1.0 / (1.0 + dist2)
    W[~np.isfinite(W)] = 0.0
    Q = W / np.sum(W)
    return np.maximum(Q, 1e-12)


def knn_jaccard_overlap(X_high, Y_low, k=15):
    """Compute per-point Jaccard overlap of k-NN sets."""
    nn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k).fit(Y_low)
    idx_high = nn_high.kneighbors(return_distance=False)
    idx_low = nn_low.kneighbors(return_distance=False)
    jaccard = np.zeros(len(X_high))
    for i in range(len(X_high)):
        s_h = set(idx_high[i])
        s_l = set(idx_low[i])
        jaccard[i] = len(s_h & s_l) / len(s_h | s_l)
    return jaccard


# ═══════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════
tab_motivation, tab_algorithm, tab_perplexity, tab_optimization, tab_pitfalls, \
    tab_validation, tab_comparison, tab_mnist, tab_dashboard = st.tabs([
        "🎯 Motivation",
        "🧮 Algorithm",
        "🎛️ Perplexity",
        "⚡ Optimization",
        "⚠️ Pitfalls",
        "✅ Validation",
        "🔀 Comparison",
        "🔢 MNIST",
        "🎮 Dashboard",
    ])

# ══════════════════════════════════════════════════════
# TAB 1: MOTIVATION
# ══════════════════════════════════════════════════════
with tab_motivation:
    st.markdown("## Why Do We Need Nonlinear Embeddings?")

    st.markdown("""
    <div class="info-panel">
    <b>The Problem:</b> Real datasets often live on <b>curved manifolds</b> (spirals, S-curves).
    Linear methods like PCA squash and mix local neighborhoods when projecting.<br><br>
    <b>The Goal:</b> Find a nonlinear map that preserves <b>who is near whom</b> — local neighborhood structure.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.markdown("### Settings")
        ds_name_mot = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()),
                                   index=0, key="mot_ds")
        n_mot = st.slider("Samples", 200, 800, 500, 50, key="mot_n")
        seed_mot = st.number_input("Seed", 0, 999, 42, key="mot_seed")

    ds_key = DATASET_OPTIONS[ds_name_mot]
    X_mot, color_mot, title_mot = make_dataset(ds_key, n=n_mot, seed=seed_mot)
    X_mot_s = StandardScaler().fit_transform(X_mot)
    cont = is_continuous(color_mot)

    Y_pca_mot = PCA(n_components=2).fit_transform(X_mot_s)
    Y_tsne_mot = run_tsne(X_mot_s, perplexity=30, seed=seed_mot)

    with col_right:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["PCA (linear)", "t-SNE (nonlinear)"])
        if cont:
            fig.add_trace(go.Scatter(x=Y_pca_mot[:, 0], y=Y_pca_mot[:, 1], mode="markers",
                                     marker=dict(size=4, color=color_mot,
                                                 colorscale="Spectral", opacity=0.7),
                                     showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=Y_tsne_mot[:, 0], y=Y_tsne_mot[:, 1], mode="markers",
                                     marker=dict(size=4, color=color_mot,
                                                 colorscale="Spectral", opacity=0.7),
                                     showlegend=False), row=1, col=2)
        else:
            for c_val in np.unique(color_mot):
                mask = color_mot == c_val
                clr = PLOTLY_DISCRETE[int(c_val) % len(PLOTLY_DISCRETE)]
                fig.add_trace(go.Scatter(x=Y_pca_mot[mask, 0], y=Y_pca_mot[mask, 1],
                                         mode="markers",
                                         marker=dict(size=4, color=clr, opacity=0.7),
                                         name=f"Class {int(c_val)}"),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=Y_tsne_mot[mask, 0], y=Y_tsne_mot[mask, 1],
                                         mode="markers",
                                         marker=dict(size=4, color=clr, opacity=0.7),
                                         showlegend=False),
                              row=1, col=2)
        fig.update_layout(title_text=f"Motivation: {title_mot}",
                          height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Distance Concentration
    with st.expander("📊 Distance Concentration in High Dimensions", expanded=False):
        st.markdown("""
        As dimensionality $D$ grows, all pairwise distances **concentrate** around the same value.
        The nearest neighbor is barely closer than a random point.

        **t-SNE's solution:** Abandon global distances. Give every point the same effective number
        of neighbors (perplexity) by adapting a local bandwidth $\\sigma_i$.
        """)

        dims_dc = [2, 5, 10, 20, 50, 100, 200]
        dim_selected = st.select_slider("Dimensions", options=dims_dc, value=10, key="dc_dim")
        rng_dc = np.random.RandomState(0)
        Xd = rng_dc.rand(800, dim_selected)
        a = rng_dc.randint(0, 800, size=6000)
        b = rng_dc.randint(0, 800, size=6000)
        dists = np.linalg.norm(Xd[a] - Xd[b], axis=1)

        fig_dc = go.Figure()
        fig_dc.add_trace(go.Histogram(x=dists, nbinsx=50, marker_color=COLORS["B"],
                                       histnorm="probability density"))
        fig_dc.update_layout(title=f"Pairwise Distance Distribution (D = {dim_selected})",
                             xaxis_title="Euclidean distance", yaxis_title="Density",
                             height=350, template="plotly_white")
        st.plotly_chart(fig_dc, use_container_width=True)

    # When to use
    with st.expander("✅ When to Use / ❌ When NOT to Use", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **✅ Use t-SNE for:**
            - Exploring clusters in high-D data
            - Spotting outliers
            - Inspecting learned representations
            - Hypothesis generation
            """)
        with c2:
            st.markdown("""
            **❌ Do NOT use t-SNE for:**
            - Measuring distances between clusters
            - Comparing cluster sizes
            - Quantitative global geometry
            - As final evidence (it's exploratory!)
            """)


# ══════════════════════════════════════════════════════
# TAB 2: ALGORITHM
# ══════════════════════════════════════════════════════
with tab_algorithm:
    st.markdown("## The t-SNE Algorithm: Three Steps")

    st.markdown("""
    | Step | What | Kernel |
    |------|------|--------|
    | **Step 1** | High-D distances → similarities $p_{ij}$ | Gaussian (per-point $\\sigma_i$) |
    | **Step 2** | Low-D distances → similarities $q_{ij}$ | Student-t (heavy tail) |
    | **Step 3** | Minimize mismatch $\\text{KL}(P \\| Q)$ | Gradient descent on $\\{y_i\\}$ |
    """)

    algo_sub = st.radio("Select topic:", [
        "Step 1: High-D Similarities (P matrix)",
        "Step 2: Student-t Kernel",
        "Step 3: KL Divergence & Forces",
    ], horizontal=True, key="algo_sub")

    if algo_sub.startswith("Step 1"):
        st.markdown("""
        <div class="math-panel">
        <b>Conditional probability:</b>
        $$p_{j|i} = \\frac{\\exp(-\\|x_i - x_j\\|^2 / 2\\sigma_i^2)}{\\sum_{k \\neq i} \\exp(-\\|x_i - x_k\\|^2 / 2\\sigma_i^2)}$$
        <b>Symmetrization:</b> $p_{ij} = (p_{j|i} + p_{i|j}) / 2n$
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### P Matrix Visualization")
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            n_tiny = st.slider("Points", 30, 80, 60, 10, key="p_n")
            perp_tiny = st.slider("Perplexity", 3, 20, 10, 1, key="p_perp")

        X_tiny, c_tiny, _ = make_dataset("blobs", n=n_tiny, seed=42)
        X_tiny = StandardScaler().fit_transform(X_tiny)
        order = np.argsort(c_tiny)
        X_tiny = X_tiny[order]

        P_tiny, sigmas_tiny = compute_P_matrix(X_tiny, perplexity=perp_tiny)

        with col_p2:
            fig_p = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Distance Matrix D",
                                                  f"P Matrix (perplexity={perp_tiny})"])
            D_mat = squareform(pdist(X_tiny))
            fig_p.add_trace(go.Heatmap(z=D_mat, colorscale="Blues", showscale=False),
                            row=1, col=1)
            fig_p.add_trace(go.Heatmap(z=P_tiny, colorscale="Hot", showscale=False),
                            row=1, col=2)
            fig_p.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown(f"""
            <div class="success-panel">
            P matrix sum: <b>{P_tiny.sum():.6f}</b> (should ≈ 1) &nbsp;|&nbsp;
            σ range: <b>{sigmas_tiny.min():.4f}</b> to <b>{sigmas_tiny.max():.4f}</b><br>
            Notice the <b>block-diagonal structure</b> — within-cluster pairs have high p_ij.
            </div>
            """, unsafe_allow_html=True)

    elif algo_sub.startswith("Step 2"):
        st.markdown("""
        <div class="math-panel">
        <b>Low-D similarity (Student-t, 1 d.o.f.):</b>
        $$q_{ij} = \\frac{(1 + \\|y_i - y_j\\|^2)^{-1}}{\\sum_{k \\neq l}(1 + \\|y_k - y_l\\|^2)^{-1}}$$
        The heavy tail lets moderately-close points spread out in 2D without a huge probability penalty.
        </div>
        """, unsafe_allow_html=True)

        r_vals = np.linspace(0, 6, 400)
        gauss = np.exp(-r_vals ** 2 / 2.0)
        student = 1.0 / (1.0 + r_vals ** 2)

        fig_kernel = go.Figure()
        fig_kernel.add_trace(go.Scatter(x=r_vals, y=gauss, mode="lines",
                                         name="Gaussian exp(−r²/2)",
                                         line=dict(color=COLORS["A"], width=3)))
        fig_kernel.add_trace(go.Scatter(x=r_vals, y=student, mode="lines",
                                         name="Student-t 1/(1+r²)",
                                         line=dict(color=COLORS["B"], width=3)))
        mask_tail = r_vals > 2
        fig_kernel.add_trace(go.Scatter(x=r_vals[mask_tail], y=student[mask_tail],
                                         fill="tozeroy",
                                         fillcolor="rgba(46,134,193,0.15)",
                                         mode="none", name="Heavy tail region"))
        fig_kernel.update_layout(title="Gaussian vs Student-t Kernel",
                                  xaxis_title="Distance r",
                                  yaxis_title="Unnormalized Similarity",
                                  height=450, template="plotly_white")
        st.plotly_chart(fig_kernel, use_container_width=True)

        # Crowding demo
        with st.expander("🔍 Crowding Problem: MDS vs t-SNE", expanded=True):
            X_cr, c_cr, _ = make_dataset("concentric", n=400, seed=42)
            X_cr = StandardScaler().fit_transform(X_cr)
            Y_mds_cr = MDS(n_components=2, random_state=42,
                           normalized_stress="auto").fit_transform(X_cr)
            Y_tsne_cr = run_tsne(X_cr, perplexity=30, seed=42)

            fig_cr = make_subplots(rows=1, cols=2,
                                   subplot_titles=["MDS (crowded)", "t-SNE (separated)"])
            for c_val in [0, 1]:
                m = c_cr == c_val
                clr = [COLORS["A"], COLORS["B"]][c_val]
                nm = ["Inner", "Outer"][c_val]
                fig_cr.add_trace(go.Scatter(x=Y_mds_cr[m, 0], y=Y_mds_cr[m, 1],
                                             mode="markers",
                                             marker=dict(size=5, color=clr, opacity=0.7),
                                             name=nm), row=1, col=1)
                fig_cr.add_trace(go.Scatter(x=Y_tsne_cr[m, 0], y=Y_tsne_cr[m, 1],
                                             mode="markers",
                                             marker=dict(size=5, color=clr, opacity=0.7),
                                             showlegend=False), row=1, col=2)
            fig_cr.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_cr, use_container_width=True)

    else:  # Step 3: KL
        st.markdown("""
        <div class="math-panel">
        <b>Cost function:</b>
        $$\\mathcal{L} = \\text{KL}(P \\| Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}$$
        <b>Gradient:</b>
        $$\\frac{\\partial \\mathcal{L}}{\\partial y_i} = 4 \\sum_{j \\neq i}
          (p_{ij} - q_{ij}) \\cdot \\frac{y_i - y_j}{1 + \\|y_i - y_j\\|^2}$$
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### KL Asymmetry")
        st.markdown("""
        | Scenario | $p_{ij}$ | $q_{ij}$ | Penalty |
        |----------|----------|----------|---------|
        | Drop a true neighbor | LARGE | small | **HUGE** |
        | Keep a false neighbor | small | LARGE | **~Zero** |
        """)

        def KL(P, Q):
            P = np.clip(np.asarray(P, float), 1e-12, 1)
            Q = np.clip(np.asarray(Q, float), 1e-12, 1)
            return float(np.sum(P * np.log(P / Q)))

        P1 = np.array([0.70, 0.20, 0.10])
        Q_good = np.array([0.65, 0.25, 0.10])
        Q_bad_true = np.array([0.10, 0.45, 0.45])
        Q_bad_false = np.array([0.80, 0.10, 0.10])

        kls = [KL(P1, Q_good), KL(P1, Q_bad_true), KL(P1, Q_bad_false)]
        labels_kl = ["Q ≈ P (good)", "Q drops true neighbor", "Q adds false neighbor"]
        colors_kl = [COLORS["C"], COLORS["A"], COLORS["D"]]

        fig_kl = go.Figure(go.Bar(x=labels_kl, y=kls, marker_color=colors_kl,
                                   text=[f"{v:.4f}" for v in kls],
                                   textposition="outside"))
        fig_kl.update_layout(title="KL Asymmetry: Dropping True Neighbors Is Expensive",
                              yaxis_title="KL(P || Q)", height=400,
                              template="plotly_white")
        st.plotly_chart(fig_kl, use_container_width=True)

        st.info("**Key insight:** t-SNE preserves local neighborhoods aggressively. "
                "It doesn't care if unrelated points end up close, but it *hates* "
                "separating true neighbors.")


# ══════════════════════════════════════════════════════
# TAB 3: PERPLEXITY
# ══════════════════════════════════════════════════════
with tab_perplexity:
    st.markdown("## Perplexity: The Key Hyperparameter")

    st.markdown("""
    <div class="info-panel">
    <b>Perplexity</b> = $2^{H(P_i)}$ where $H$ is the Shannon entropy of the conditional distribution
    around point $x_i$.<br>
    Think of it as a <b>smooth k-nearest neighbors</b>. t-SNE runs a binary search for each point
    to find $\\sigma_i$ matching the target perplexity.
    </div>
    """, unsafe_allow_html=True)

    col_perp_l, col_perp_r = st.columns([1, 3])
    with col_perp_l:
        ds_perp_name = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()),
                                     index=0, key="perp_ds")
        n_perp = st.slider("Samples", 200, 600, 400, 50, key="perp_n")
        perp_val = st.slider("Perplexity", 2, 80, 30, 1, key="perp_val")

    ds_perp_key = DATASET_OPTIONS[ds_perp_name]
    X_perp, color_perp, title_perp = make_dataset(ds_perp_key, n=n_perp, seed=42)
    X_perp_s = StandardScaler().fit_transform(X_perp)
    cont_perp = is_continuous(color_perp)
    Y_perp = run_tsne(X_perp_s, perplexity=perp_val, seed=42)

    with col_perp_r:
        fig_perp = plotly_embedding(Y_perp, color_perp,
                                     title=f"{title_perp} — Perplexity = {perp_val}",
                                     continuous=cont_perp, height=500)
        st.plotly_chart(fig_perp, use_container_width=True)

    if perp_val <= 5:
        st.warning("⚠️ Very low perplexity → micro-clusters, may fragment continuous data.")
    elif perp_val >= 50:
        st.warning("⚠️ High perplexity → macro-structure dominates, local detail lost.")
    else:
        st.success("✅ Moderate perplexity — good balance of local and global structure.")

    with st.expander("📈 Perplexity Sweep (pre-computed)", expanded=False):
        perps_sweep = [2, 5, 10, 20, 30, 50]
        fig_sweep = make_subplots(rows=2, cols=3,
                                   subplot_titles=[f"Perp = {p}" for p in perps_sweep])
        for idx, p in enumerate(perps_sweep):
            Y_sw = run_tsne(X_perp_s, perplexity=p, seed=42)
            row, col = idx // 3 + 1, idx % 3 + 1
            if cont_perp:
                fig_sweep.add_trace(go.Scatter(x=Y_sw[:, 0], y=Y_sw[:, 1],
                                                mode="markers",
                                                marker=dict(size=3, color=color_perp,
                                                            colorscale="Spectral",
                                                            opacity=0.7),
                                                showlegend=False), row=row, col=col)
            else:
                for c_val in np.unique(color_perp):
                    m = color_perp == c_val
                    clr = PLOTLY_DISCRETE[int(c_val) % len(PLOTLY_DISCRETE)]
                    fig_sweep.add_trace(go.Scatter(x=Y_sw[m, 0], y=Y_sw[m, 1],
                                                    mode="markers",
                                                    marker=dict(size=3, color=clr,
                                                                opacity=0.7),
                                                    showlegend=False),
                                        row=row, col=col)
        fig_sweep.update_layout(height=600, template="plotly_white",
                                 title_text="Perplexity Sweep")
        for r in range(1, 3):
            for c in range(1, 4):
                fig_sweep.update_xaxes(showticklabels=False, row=r, col=c)
                fig_sweep.update_yaxes(showticklabels=False, row=r, col=c)
        st.plotly_chart(fig_sweep, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4: OPTIMIZATION
# ══════════════════════════════════════════════════════
with tab_optimization:
    st.markdown("## Optimization: Learning Rate & Convergence")

    st.markdown("""
    | Parameter | Typical Range | Effect |
    |-----------|--------------|--------|
    | **Learning rate (η)** | 200–1000 | Too small → blob; too large → explosion |
    | **Early exaggeration** | ×12 for first 250 iters | Forces tight clusters early |
    | **Initialization** | `"pca"` recommended | Stable, faster convergence |
    | **Iterations** | 500–1000 | Too few → incomplete |
    """)

    opt_sub = st.radio("Topic:", ["Learning Rate", "Iteration Snapshots"], horizontal=True,
                        key="opt_sub")

    if opt_sub == "Learning Rate":
        X_lr, c_lr, _ = make_dataset("blobs", n=400, seed=42)
        X_lr = StandardScaler().fit_transform(X_lr)

        rates = {"10 (too small)": 10, "200 (default)": 200,
                 "500 (moderate)": 500, "5000 (too large)": 5000}
        fig_lr = make_subplots(rows=1, cols=4, subplot_titles=list(rates.keys()))
        for idx, (label, lr) in enumerate(rates.items()):
            Y_lr = run_tsne(X_lr, perplexity=30, lr=lr, seed=42, n_iter=800)
            for c_val in np.unique(c_lr):
                m = c_lr == c_val
                clr = PLOTLY_DISCRETE[int(c_val) % len(PLOTLY_DISCRETE)]
                fig_lr.add_trace(go.Scatter(x=Y_lr[m, 0], y=Y_lr[m, 1], mode="markers",
                                             marker=dict(size=4, color=clr, opacity=0.7),
                                             showlegend=False),
                                  row=1, col=idx + 1)
        fig_lr.update_layout(title="Effect of Learning Rate", height=380,
                              template="plotly_white")
        for i in range(1, 5):
            fig_lr.update_xaxes(showticklabels=False, row=1, col=i)
            fig_lr.update_yaxes(showticklabels=False, row=1, col=i)
        st.plotly_chart(fig_lr, use_container_width=True)

        st.markdown("""
        <div class="warn-panel">
        <b>Too small →</b> blob (stuck in initial state). &nbsp;
        <b>Too large →</b> scattered chaos. &nbsp;
        <b>Moderate →</b> clear, separated clusters.
        </div>
        """, unsafe_allow_html=True)

    else:  # Iteration Snapshots
        col_snap_l, col_snap_r = st.columns([1, 4])
        with col_snap_l:
            ds_snap = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()),
                                    index=2, key="snap_ds")
            early_exag_on = st.checkbox("Early Exaggeration", value=True, key="snap_exag")

        ds_snap_key = DATASET_OPTIONS[ds_snap]
        X_snap, c_snap, title_snap = make_dataset(ds_snap_key, n=400, seed=42)
        X_snap_s = StandardScaler().fit_transform(X_snap)
        cont_snap = is_continuous(c_snap)
        exag = 12.0 if early_exag_on else 1.0
        iter_stops = [250, 350, 500, 700, 850, 1000]

        with col_snap_r:
            fig_snap = make_subplots(rows=2, cols=3,
                                      subplot_titles=[f"Iter {it}" for it in iter_stops])
            for idx, n_it in enumerate(iter_stops):
                Y_snap = run_tsne(X_snap_s, perplexity=30, lr=200, n_iter=n_it,
                                   seed=42, early_exag=exag)
                row, col = idx // 3 + 1, idx % 3 + 1
                if cont_snap:
                    fig_snap.add_trace(
                        go.Scatter(x=Y_snap[:, 0], y=Y_snap[:, 1], mode="markers",
                                   marker=dict(size=3, color=c_snap,
                                               colorscale="Spectral", opacity=0.7),
                                   showlegend=False), row=row, col=col)
                else:
                    for c_val in np.unique(c_snap):
                        m = c_snap == c_val
                        clr = PLOTLY_DISCRETE[int(c_val) % len(PLOTLY_DISCRETE)]
                        fig_snap.add_trace(
                            go.Scatter(x=Y_snap[m, 0], y=Y_snap[m, 1], mode="markers",
                                       marker=dict(size=3, color=clr, opacity=0.7),
                                       showlegend=False), row=row, col=col)

            exag_lbl = "ON" if early_exag_on else "OFF"
            fig_snap.update_layout(
                title_text=f"Optimization: {title_snap} (Early Exagg. {exag_lbl})",
                height=650, template="plotly_white")
            for r in range(1, 3):
                for c in range(1, 4):
                    fig_snap.update_xaxes(showticklabels=False, row=r, col=c)
                    fig_snap.update_yaxes(showticklabels=False, row=r, col=c)
            st.plotly_chart(fig_snap, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 5: PITFALLS
# ══════════════════════════════════════════════════════
with tab_pitfalls:
    st.markdown("## Common Pitfalls — What You Can and Cannot Trust")

    st.markdown("""
    <div class="warn-panel">
    <b>The Golden Rule:</b> t-SNE is a <b>topology map</b> (who is next to whom),
    <b>not a metric map</b>. Global distances, axes, and cluster sizes are all distorted.
    </div>
    """, unsafe_allow_html=True)

    pitfall_choice = st.radio("Select pitfall:", [
        "1: Axes Are Meaningless",
        "2: Global Distances Are Illusions",
        "3: Sizes & Densities Distorted",
        "4: Tearing Continuous Data",
    ], horizontal=True, key="pitfall_choice")

    if pitfall_choice.startswith("1"):
        st.markdown("### Pitfall 1: Axes Are Meaningless")
        st.markdown("The KL objective is invariant to rotation, translation, and scale. "
                     "**Never say** 'Feature X increases as you move right.'")

        X_pit, c_pit, _ = make_dataset("blobs", n=400, seed=42)
        X_pit = StandardScaler().fit_transform(X_pit)

        fig_seeds = make_subplots(rows=1, cols=4,
                                   subplot_titles=[f"Seed {s}" for s in [0, 1, 2, 3]])
        for idx, seed in enumerate([0, 1, 2, 3]):
            Y_pit = run_tsne(X_pit, perplexity=30, seed=seed)
            for c_val in np.unique(c_pit):
                m = c_pit == c_val
                clr = PLOTLY_DISCRETE[int(c_val) % len(PLOTLY_DISCRETE)]
                fig_seeds.add_trace(
                    go.Scatter(x=Y_pit[m, 0], y=Y_pit[m, 1], mode="markers",
                               marker=dict(size=4, color=clr, opacity=0.7),
                               showlegend=(idx == 0), name=f"C{int(c_val)}"),
                    row=1, col=idx + 1)
        fig_seeds.update_layout(title="Same Data, Different Seeds → Different Orientations",
                                 height=380, template="plotly_white")
        for i in range(1, 5):
            fig_seeds.update_xaxes(showticklabels=False, row=1, col=i)
            fig_seeds.update_yaxes(showticklabels=False, row=1, col=i)
        st.plotly_chart(fig_seeds, use_container_width=True)

    elif pitfall_choice.startswith("2"):
        st.markdown("### Pitfall 2: Global Distances Are Illusions")
        st.markdown("The empty space between clusters is arbitrary. "
                     "You **cannot** conclude that cluster A is closer to B than to C.")

        X_gd, c_gd, _ = make_dataset("blobs", n=400, seed=42)
        X_gd = StandardScaler().fit_transform(X_gd)
        Y_gd = run_tsne(X_gd, perplexity=30, seed=42)

        labels_u = np.unique(c_gd)
        cent_orig = np.array([X_gd[c_gd == c].mean(axis=0) for c in labels_u])
        cent_emb = np.array([Y_gd[c_gd == c].mean(axis=0) for c in labels_u])
        d_orig = squareform(pdist(cent_orig))
        d_emb = squareform(pdist(cent_emb))
        d_orig_n = d_orig / d_orig.max()
        d_emb_n = d_emb / d_emb.max()
        cn = [f"C{int(c)}" for c in labels_u]

        fig_gd = make_subplots(rows=1, cols=2,
                                subplot_titles=["Original Distances (normalized)",
                                                "t-SNE Distances (normalized)"])
        fig_gd.add_trace(go.Heatmap(z=d_orig_n, x=cn, y=cn, colorscale="Blues",
                                     showscale=False, text=np.round(d_orig_n, 2),
                                     texttemplate="%{text}"), row=1, col=1)
        fig_gd.add_trace(go.Heatmap(z=d_emb_n, x=cn, y=cn, colorscale="Reds",
                                     showscale=False, text=np.round(d_emb_n, 2),
                                     texttemplate="%{text}"), row=1, col=2)
        fig_gd.update_layout(title="Inter-Cluster Distances Are NOT Preserved",
                              height=400, template="plotly_white")
        st.plotly_chart(fig_gd, use_container_width=True)

    elif pitfall_choice.startswith("3"):
        st.markdown("### Pitfall 3: Sizes & Densities Are Distorted")
        st.markdown("The local bandwidth σ_i expands sparse regions and shrinks dense ones. "
                     "All clusters end up looking roughly the same size.")

        np.random.seed(42)
        dense = np.random.randn(200, 10) * 0.5 + 3
        sparse = np.random.randn(30, 10) * 2.0 - 3
        X_dd = StandardScaler().fit_transform(np.vstack([dense, sparse]))
        labels_dd = np.array([0] * 200 + [1] * 30)

        Y_dd = run_tsne(X_dd, perplexity=20, seed=42)
        X_dd_pca = PCA(n_components=2).fit_transform(X_dd)

        fig_dd = make_subplots(rows=1, cols=2,
                                subplot_titles=["PCA View (true sizes)",
                                                "t-SNE (distorted sizes)"])
        for c_val, nm, clr in [(0, "Dense (n=200)", COLORS["A"]),
                                (1, "Sparse (n=30)", COLORS["C"])]:
            m = labels_dd == c_val
            fig_dd.add_trace(go.Scatter(x=X_dd_pca[m, 0], y=X_dd_pca[m, 1],
                                         mode="markers",
                                         marker=dict(size=5, color=clr, opacity=0.7),
                                         name=nm), row=1, col=1)
            fig_dd.add_trace(go.Scatter(x=Y_dd[m, 0], y=Y_dd[m, 1], mode="markers",
                                         marker=dict(size=5, color=clr, opacity=0.7),
                                         showlegend=False), row=1, col=2)
        fig_dd.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_dd, use_container_width=True)

    else:  # Pitfall 4: Tearing
        st.markdown("### Pitfall 4: Low Perplexity Tears Continuous Data")
        st.markdown("A 1-D gradient in 20D — low perplexity creates artificial islands.")

        rng_frag = np.random.RandomState(42)
        t_frag = np.linspace(0, 1, 300)
        X_frag = np.column_stack(
            [t_frag] + [rng_frag.randn(300) * 0.05 for _ in range(19)])
        X_frag = StandardScaler().fit_transform(X_frag)

        perp_frag = st.slider("Perplexity", 2, 60, 5, 2, key="frag_perp")
        Y_frag = run_tsne(X_frag, perplexity=perp_frag, seed=42, n_iter=800)

        fig_frag = px.scatter(x=Y_frag[:, 0], y=Y_frag[:, 1], color=t_frag,
                              color_continuous_scale="Spectral",
                              title=f"Continuous 1D Gradient — Perplexity = {perp_frag}")
        fig_frag.update_layout(height=450, template="plotly_white")
        fig_frag.update_traces(marker=dict(size=6))
        st.plotly_chart(fig_frag, use_container_width=True)

        if perp_frag < 10:
            st.error("⚠️ Low perplexity tore the continuous gradient into artificial islands!")
        else:
            st.success("✅ Moderate perplexity preserves the continuous nature.")


# ══════════════════════════════════════════════════════
# TAB 6: VALIDATION
# ══════════════════════════════════════════════════════
with tab_validation:
    st.markdown("## Validating Your Embedding")

    st.markdown("""
    | Check | How | What it tells you |
    |-------|-----|-------------------|
    | **kNN Overlap (Jaccard)** | Compare k-NN in high-D vs low-D | Local structure preservation |
    | **Trustworthiness** | `sklearn.manifold.trustworthiness` | How well low-D respects high-D neighbors |
    | **Stability** | Multiple seeds | Whether clusters are real or artifacts |
    """)

    val_sub = st.radio("Validation method:", ["kNN Jaccard Overlap", "Trustworthiness Sweep"],
                        horizontal=True, key="val_sub")

    if val_sub == "kNN Jaccard Overlap":
        col_v1, col_v2 = st.columns([1, 3])
        with col_v1:
            k_val = st.slider("k (neighbors)", 5, 50, 15, 1, key="knn_k")
            ds_val_name = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()),
                                        index=2, key="val_ds")

        ds_val_key = DATASET_OPTIONS[ds_val_name]
        X_val, c_val, _ = make_dataset(ds_val_key, n=400, seed=42)
        X_val = StandardScaler().fit_transform(X_val)
        Y_val = run_tsne(X_val, perplexity=30, seed=42)
        jacc = knn_jaccard_overlap(X_val, Y_val, k=k_val)

        with col_v2:
            fig_jacc = make_subplots(rows=1, cols=2,
                                      subplot_titles=[
                                          f"Embedding (color = Jaccard, k={k_val})",
                                          f"Distribution (mean = {jacc.mean():.3f})"])
            fig_jacc.add_trace(go.Scatter(
                x=Y_val[:, 0], y=Y_val[:, 1], mode="markers",
                marker=dict(size=5, color=jacc, colorscale="RdYlGn",
                            colorbar=dict(title="Jaccard"), opacity=0.8)),
                row=1, col=1)
            fig_jacc.add_trace(go.Histogram(x=jacc, nbinsx=30,
                                             marker_color=COLORS["B"]),
                                row=1, col=2)
            fig_jacc.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_jacc, use_container_width=True)

    else:  # Trustworthiness
        st.markdown("Computing trustworthiness across perplexities on MNIST digits...")

        @st.cache_data(show_spinner="Computing trustworthiness sweep...")
        def trust_sweep():
            digits = load_digits()
            X_d = StandardScaler().fit_transform(digits.data)
            X_d_pca = PCA(n_components=40).fit_transform(X_d)
            perps_ = [5, 10, 15, 20, 30, 40, 50]
            scores_, times_ = [], []
            for p in perps_:
                t0 = perf_counter()
                Y_ = run_tsne.__wrapped__(X_d_pca, perplexity=p, seed=0, lr=200, n_iter=800)
                dt_ = perf_counter() - t0
                tw_ = trustworthiness(X_d_pca, Y_, n_neighbors=10)
                scores_.append(tw_)
                times_.append(dt_)
            return perps_, scores_, times_

        perps_tw, scores_tw, times_tw = trust_sweep()

        fig_tw = make_subplots(rows=1, cols=2,
                                subplot_titles=["Trustworthiness@10 vs Perplexity",
                                                "Runtime vs Perplexity"])
        fig_tw.add_trace(go.Scatter(x=perps_tw, y=scores_tw, mode="lines+markers",
                                     marker=dict(color=COLORS["B"])), row=1, col=1)
        fig_tw.add_trace(go.Bar(x=[str(p) for p in perps_tw], y=times_tw,
                                 marker_color=COLORS["D"]), row=1, col=2)
        fig_tw.update_layout(height=400, template="plotly_white", showlegend=False)
        fig_tw.update_yaxes(title_text="Trustworthiness", row=1, col=1)
        fig_tw.update_yaxes(title_text="Seconds", row=1, col=2)
        st.plotly_chart(fig_tw, use_container_width=True)

        # Show values
        df_tw = {"Perplexity": perps_tw,
                 "Trustworthiness@10": [f"{s:.4f}" for s in scores_tw],
                 "Runtime (s)": [f"{t:.1f}" for t in times_tw]}
        st.dataframe(df_tw, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 7: COMPARISON
# ══════════════════════════════════════════════════════
with tab_comparison:
    st.markdown("## Method Comparison: PCA / MDS / t-SNE / UMAP")

    st.markdown("""
    | Method | Type | Preserves | Speed |
    |--------|------|-----------|-------|
    | **PCA** | Linear | Global variance | Fast |
    | **MDS** | Nonlinear | Global distances | Moderate |
    | **t-SNE** | Nonlinear | Local neighborhoods | Moderate |
    | **UMAP** | Nonlinear | Local + some global | Fast |
    """)

    col_cmp1, col_cmp2 = st.columns([1, 4])
    with col_cmp1:
        ds_cmp_name = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()),
                                    index=0, key="cmp_ds")
        n_cmp = st.slider("Samples", 200, 600, 400, 50, key="cmp_n")

    ds_cmp_key = DATASET_OPTIONS[ds_cmp_name]
    X_cmp, c_cmp, title_cmp = make_dataset(ds_cmp_key, n=n_cmp, seed=42)
    X_cmp_s = StandardScaler().fit_transform(X_cmp)
    cont_cmp = is_continuous(c_cmp)

    t0 = perf_counter()
    Y_pca_cmp = PCA(n_components=2).fit_transform(X_cmp_s)
    t_pca = perf_counter() - t0

    t0 = perf_counter()
    Y_mds_cmp = MDS(n_components=2, random_state=42,
                     normalized_stress="auto").fit_transform(X_cmp_s)
    t_mds = perf_counter() - t0

    t0 = perf_counter()
    Y_tsne_cmp = run_tsne(X_cmp_s, perplexity=30, seed=42)
    t_tsne = perf_counter() - t0

    # Try UMAP
    Y_umap_cmp = None
    t_umap = None
    try:
        import umap
        t0 = perf_counter()
        Y_umap_cmp = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                 random_state=42).fit_transform(X_cmp_s)
        t_umap = perf_counter() - t0
    except ImportError:
        pass

    with col_cmp2:
        methods = {"PCA": (Y_pca_cmp, t_pca), "MDS": (Y_mds_cmp, t_mds),
                   "t-SNE": (Y_tsne_cmp, t_tsne)}
        if Y_umap_cmp is not None:
            methods["UMAP"] = (Y_umap_cmp, t_umap)

        n_methods = len(methods)
        fig_cmp = make_subplots(rows=1, cols=n_methods,
                                 subplot_titles=[f"{nm} ({t:.2f}s)"
                                                 for nm, (_, t) in methods.items()])
        for idx, (nm, (Y_m, _)) in enumerate(methods.items()):
            if cont_cmp:
                fig_cmp.add_trace(go.Scatter(
                    x=Y_m[:, 0], y=Y_m[:, 1], mode="markers",
                    marker=dict(size=4, color=c_cmp, colorscale="Spectral", opacity=0.7),
                    showlegend=False), row=1, col=idx + 1)
            else:
                for c_val in np.unique(c_cmp):
                    m = c_cmp == c_val
                    clr = PLOTLY_DISCRETE[int(c_val) % len(PLOTLY_DISCRETE)]
                    fig_cmp.add_trace(go.Scatter(
                        x=Y_m[m, 0], y=Y_m[m, 1], mode="markers",
                        marker=dict(size=4, color=clr, opacity=0.7),
                        showlegend=False), row=1, col=idx + 1)
        fig_cmp.update_layout(title_text=f"Method Comparison: {title_cmp}",
                               height=420, template="plotly_white")
        for i in range(1, n_methods + 1):
            fig_cmp.update_xaxes(showticklabels=False, row=1, col=i)
            fig_cmp.update_yaxes(showticklabels=False, row=1, col=i)
        st.plotly_chart(fig_cmp, use_container_width=True)

    if Y_umap_cmp is None:
        st.info("💡 Install `umap-learn` to include UMAP in the comparison.")


# ══════════════════════════════════════════════════════
# TAB 8: MNIST
# ══════════════════════════════════════════════════════
with tab_mnist:
    st.markdown("## Real-World Application: Handwritten Digits (MNIST)")

    st.markdown("""
    <div class="info-panel">
    <b>Pipeline:</b> Load digits (8×8, 64 features) → StandardScaler → PCA (30D) → t-SNE → Interactive Plotly
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 4])
    with col_m1:
        pca_dims_m = st.slider("PCA dims", 5, 64, 30, 5, key="mnist_pca")
        perp_m = st.slider("Perplexity", 5, 80, 30, 5, key="mnist_perp")
        seed_m = st.number_input("Seed", 0, 999, 42, key="mnist_seed")

    @st.cache_data(show_spinner="Running t-SNE on digits...")
    def mnist_pipeline(pca_dims, perp, seed):
        digits = load_digits()
        X = StandardScaler().fit_transform(digits.data)
        X_pca = PCA(n_components=pca_dims).fit_transform(X)
        var_ret = PCA(n_components=pca_dims).fit(X).explained_variance_ratio_.sum()
        t0 = perf_counter()
        Y = TSNE(n_components=2, perplexity=perp, learning_rate=200,
                 max_iter=1000, init="pca", random_state=seed).fit_transform(X_pca)
        dt = perf_counter() - t0
        tw = trustworthiness(X_pca, Y, n_neighbors=10)
        return Y, digits.target, var_ret, dt, tw

    Y_m, targets_m, var_m, dt_m, tw_m = mnist_pipeline(pca_dims_m, perp_m, seed_m)

    with col_m2:
        # Metrics row
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Variance Retained", f"{var_m:.1%}")
        mc2.metric("Trustworthiness@10", f"{tw_m:.4f}")
        mc3.metric("Runtime", f"{dt_m:.1f}s")

        fig_m = go.Figure()
        for d in range(10):
            mask = targets_m == d
            fig_m.add_trace(go.Scatter(
                x=Y_m[mask, 0], y=Y_m[mask, 1], mode="markers",
                name=f"Digit {d}",
                marker=dict(size=5, color=PLOTLY_DISCRETE[d], opacity=0.7),
                text=[f"Digit {d}"] * mask.sum(), hoverinfo="text"))
        fig_m.update_layout(
            title=f"t-SNE of Digits (PCA→{pca_dims_m}D, perp={perp_m})",
            xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
            height=600, template="plotly_white",
            legend=dict(title="Digit"))
        st.plotly_chart(fig_m, use_container_width=True)

    st.info("💡 Click legend entries to toggle digits. Zoom into boundary regions "
            "(e.g., 4 vs 9, 3 vs 8) to explore confusion zones.")


# ══════════════════════════════════════════════════════
# TAB 9: DASHBOARD
# ══════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("## Full Parameter Dashboard")
    st.markdown("Explore all t-SNE parameters at once on your choice of dataset.")

    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.markdown("### Dataset")
        ds_dash_name = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()),
                                     index=2, key="dash_ds")
        n_dash = st.slider("Samples", 200, 800, 400, 50, key="dash_n")

        st.markdown("### t-SNE Parameters")
        perp_dash = st.slider("Perplexity", 2, 100, 30, 1, key="dash_perp")
        lr_dash = st.slider("Learning Rate", 10, 2000, 200, 10, key="dash_lr")
        init_dash = st.selectbox("Initialization", ["pca", "random"], key="dash_init")
        iter_dash = st.slider("Iterations", 250, 2000, 1000, 50, key="dash_iter")
        seed_dash = st.number_input("Seed", 0, 999, 42, key="dash_seed")
        exag_dash = st.slider("Early Exaggeration", 1.0, 30.0, 12.0, 1.0, key="dash_exag")

    ds_dash_key = DATASET_OPTIONS[ds_dash_name]
    X_dash, c_dash, title_dash = make_dataset(ds_dash_key, n=n_dash, seed=seed_dash)
    X_dash_s = StandardScaler().fit_transform(X_dash)
    cont_dash = is_continuous(c_dash)

    t0 = perf_counter()
    Y_dash = run_tsne(X_dash_s, perplexity=perp_dash, lr=lr_dash, n_iter=iter_dash,
                       init=init_dash, seed=seed_dash, early_exag=exag_dash)
    dt_dash = perf_counter() - t0

    tw_dash = trustworthiness(X_dash_s, Y_dash, n_neighbors=min(10, n_dash - 1))
    jacc_dash = knn_jaccard_overlap(X_dash_s, Y_dash, k=min(15, n_dash - 1))

    with col_d2:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Trustworthiness@10", f"{tw_dash:.4f}")
        mc2.metric("Mean Jaccard@15", f"{jacc_dash.mean():.4f}")
        mc3.metric("Runtime", f"{dt_dash:.1f}s")

        fig_dash = plotly_embedding(Y_dash, c_dash,
                                     title=f"{title_dash} — perp={perp_dash}, lr={lr_dash}",
                                     continuous=cont_dash, height=550)
        st.plotly_chart(fig_dash, use_container_width=True)

    # Preprocessing note
    with st.expander("📋 Preprocessing & Reproducibility Checklist", expanded=False):
        st.markdown("""
        **Always report these when publishing a t-SNE plot:**

        | Parameter | Your Value |
        |-----------|-----------|
        | Scaling | StandardScaler |
        | PCA dims | — (synthetic data) |
        | Perplexity | {perp} |
        | Learning rate | {lr} |
        | Iterations | {iters} |
        | Initialization | {init} |
        | Early exaggeration | {exag} |
        | Random seed | {seed} |
        | scikit-learn version | `sklearn.__version__` |
        """.format(perp=perp_dash, lr=lr_dash, iters=iter_dash,
                   init=init_dash, exag=exag_dash, seed=seed_dash))


# ══════════════════════════════════════════════════════
# SIDEBAR SUMMARY
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.markdown("## 📖 t-SNE Quick Reference")
    st.markdown("""
    **5 Rules of Thumb:**
    1. **Pre-process** — Scale features
    2. **Denoise** — PCA to 30–50 dims first
    3. **Sweep** — Try perplexity {5, 10, 30, 50}
    4. **Verify** — Multiple random seeds
    5. **Interpret Locally** — Ignore global distances

    **Key Math:**
    - $p_{ij}$ : Gaussian kernel (high-D)
    - $q_{ij}$ : Student-t kernel (low-D)
    - Cost: $\\text{KL}(P \\| Q)$
    - Gradient: attraction + repulsion

    **References:**
    - van der Maaten & Hinton (2008)
    - [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
    """)
