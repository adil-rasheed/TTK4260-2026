"""
ICA Prerequisite — Central Limit Theorem & Independence vs Correlation
=======================================================================
Interactive demonstrations of the two key prerequisites for understanding ICA:
1. The Central Limit Theorem (CLT) — why ICA needs non-Gaussianity
2. Uncorrelated ≠ Independent — correlation vs mutual information

Based on ICA-Prerequisite.ipynb
"""

import streamlit as st
import numpy as np
import math
from collections import Counter

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

st.set_page_config(page_title="ICA Prerequisite", page_icon="🔑", layout="wide")

# ===== STYLING =====
st.markdown("""
<style>
    .ica-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;
    }
    .ica-header h1 { color: white; margin: 0; }
    .ica-header p  { color: #e0d7ff; margin: 0.5rem 0 0 0; }
    .info-panel {
        background-color: #f0f4ff; border-left: 4px solid #6366f1;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ica-header">
    <h1>🔑 ICA Prerequisites</h1>
    <p>Central Limit Theorem &amp; Understanding Statistical Independence</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────  HELPER FUNCTIONS  ──────────────────────────
def correlation(x, y):
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    return num / den if den != 0 else 0

def mutual_information(x, y, bins=20):
    n = len(x)
    x_bins = np.linspace(x.min(), x.max(), bins + 1)
    y_bins = np.linspace(y.min(), y.max(), bins + 1)
    x_d = np.digitize(x, x_bins) - 1
    y_d = np.digitize(y, y_bins) - 1
    joint_counts = Counter(zip(x_d.tolist(), y_d.tolist()))
    px, py = Counter(x_d.tolist()), Counter(y_d.tolist())
    mi = 0.0
    for (xi, yi), c in joint_counts.items():
        p_xy = c / n
        p_x = px[xi] / n
        p_y = py[yi] / n
        mi += p_xy * math.log2(p_xy / (p_x * p_y + 1e-12) + 1e-12)
    return mi

def generate_data(dist, n_samples, n_vars, rng):
    if dist == "Uniform":
        return rng.uniform(-1, 1, (n_samples, n_vars))
    elif dist == "Exponential":
        return rng.exponential(1.0, (n_samples, n_vars))
    elif dist == "Laplace":
        return rng.laplace(0, 1, (n_samples, n_vars))
    elif dist == "Bernoulli":
        return rng.binomial(1, 0.5, (n_samples, n_vars))
    elif dist == "Poisson":
        return rng.poisson(3.0, (n_samples, n_vars))
    elif dist == "Mixed (1000)":
        X1 = rng.uniform(-1, 1, (n_samples, max(n_vars // 5, 1)))
        X2 = rng.exponential(1, (n_samples, max(n_vars // 5, 1)))
        X3 = rng.laplace(0, 1, (n_samples, max(n_vars // 5, 1)))
        X4 = rng.binomial(1, 0.5, (n_samples, max(n_vars // 5, 1)))
        X5 = rng.poisson(3, (n_samples, max(n_vars // 5, 1)))
        return np.hstack([X1, X2, X3, X4, X5])
    return rng.normal(0, 1, (n_samples, n_vars))


# ═══════════════════════  TAB LAYOUT  ═══════════════════════
tab_clt, tab_indep = st.tabs([
    "📊 Central Limit Theorem",
    "🔗 Independence vs Correlation"
])

# ──────────────────────────────────────────────────────────────
# TAB 1: Central Limit Theorem
# ──────────────────────────────────────────────────────────────
with tab_clt:
    st.markdown("## Central Limit Theorem & Non-Gaussianity")

    st.markdown("""
<div class="info-panel">

**Why does ICA need non-Gaussianity?**

The **Central Limit Theorem** says that sums of independent random variables tend toward a Gaussian distribution.  
When signals are **mixed** (linearly combined), the mixtures are *more Gaussian* than the original sources.

➡ **ICA reverses this**: it searches for directions of **maximum non-Gaussianity** to recover the original sources.

</div>
""", unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### ⚙️ Settings")
        n_samples = st.slider("Number of samples", 1000, 50000, 10000, 1000, key="clt_n")
        selected_dists = st.multiselect(
            "Distributions",
            ["Uniform", "Exponential", "Laplace", "Bernoulli", "Poisson", "Mixed (1000)"],
            default=["Uniform", "Exponential", "Laplace"],
            key="clt_dists"
        )
        k_values = st.multiselect(
            "Number of summed variables (k)",
            [1, 2, 5, 10, 50, 100, 500, 1000],
            default=[1, 5, 10, 50, 100],
            key="clt_k"
        )
        k_values = sorted(k_values)

    with col_plot:
        if selected_dists and k_values:
            rng = np.random.default_rng(42)
            n_rows, n_cols = len(selected_dists), len(k_values)
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f"k={k}" for k in k_values] * n_rows,
                vertical_spacing=0.06, horizontal_spacing=0.04
            )

            for row_i, dist in enumerate(selected_dists):
                max_k = max(k_values)
                X = generate_data(dist, n_samples, max_k, rng)

                for col_j, k in enumerate(k_values):
                    sums = X[:, :k].sum(axis=1) / np.sqrt(k)
                    mu, sigma = np.mean(sums), np.std(sums)
                    hist_vals, bin_edges = np.histogram(sums, bins=40, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    x_gauss = np.linspace(sums.min(), sums.max(), 200)
                    y_gauss = norm.pdf(x_gauss, mu, sigma)

                    fig.add_trace(go.Bar(
                        x=bin_centers, y=hist_vals, marker_color='rgba(100,149,237,0.6)',
                        showlegend=False, name=f"{dist} k={k}"
                    ), row=row_i + 1, col=col_j + 1)
                    fig.add_trace(go.Scatter(
                        x=x_gauss, y=y_gauss, mode='lines',
                        line=dict(color='black', width=2), showlegend=False
                    ), row=row_i + 1, col=col_j + 1)

                    # Row label on first column
                    if col_j == 0:
                        fig.update_yaxes(title_text=dist, row=row_i + 1, col=1)

            height = max(250 * n_rows, 400)
            fig.update_layout(
                height=height, title_text="Central Limit Theorem: Sums → Gaussian",
                title_font_size=16, margin=dict(t=60)
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one distribution and one k value.")

    with st.expander("📖 Mathematical Background"):
        st.markdown(r"""
**Central Limit Theorem (CLT):**

Let $X_1, X_2, \ldots, X_k$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$. Then:

$$S_k = \frac{1}{\sqrt{k}} \sum_{i=1}^k (X_i - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2) \quad \text{as } k \to \infty$$

**Implications for ICA:**
- **Mixing = summing** → mixtures are *more Gaussian* than sources
- **Non-Gaussianity decreases** with more mixing
- **ICA strategy**: find directions where non-Gaussianity is **maximized** → those are the original sources
        """)

# ──────────────────────────────────────────────────────────────
# TAB 2: Independence vs Correlation
# ──────────────────────────────────────────────────────────────
with tab_indep:
    st.markdown("## Uncorrelated ≠ Independent")

    st.markdown("""
<div class="info-panel">

**Key insight for ICA:** Correlation ($\\rho$) only measures *linear* dependence.  
Mutual Information ($I$) measures *any* statistical dependence.

ICA must go **beyond** decorrelation (PCA) and minimize mutual information to achieve true independence.

</div>
""", unsafe_allow_html=True)

    col_settings, col_examples = st.columns([1, 3])

    with col_settings:
        st.markdown("### ⚙️ Settings")
        n_points = st.slider("Number of points", 100, 2000, 500, 100, key="indep_n")
        mi_bins = st.slider("MI histogram bins", 10, 50, 20, 5, key="indep_bins")
        noise_level = st.slider("Noise (Example A)", 0.1, 5.0, 1.0, 0.1, key="indep_noise")

    with col_examples:
        rng = np.random.default_rng(42)

        # Example A: Correlated
        X_a = rng.uniform(-5, 5, n_points)
        Y_a = 2 * X_a + rng.normal(0, noise_level, n_points)
        rho_a = correlation(X_a, Y_a)
        mi_a = mutual_information(X_a, Y_a, bins=mi_bins)

        # Example B: Independent
        X_b = rng.normal(0, 1, n_points)
        Y_b = rng.laplace(0, 1, n_points)
        rho_b = correlation(X_b, Y_b)
        mi_b = mutual_information(X_b, Y_b, bins=mi_bins)

        # Example C: Uncorrelated but Dependent
        X_c = rng.uniform(-1, 1, n_points)
        Y_c = X_c ** 2
        rho_c = correlation(X_c, Y_c)
        mi_c = mutual_information(X_c, Y_c, bins=mi_bins)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"A: Correlated (ρ={rho_a:.3f}, MI={mi_a:.3f})",
                f"B: Independent (ρ={rho_b:.3f}, MI={mi_b:.3f})",
                f"C: Uncorrelated but Dependent (ρ={rho_c:.3f}, MI={mi_c:.3f})"
            ],
            horizontal_spacing=0.08
        )

        fig.add_trace(go.Scatter(x=X_a, y=Y_a, mode='markers',
            marker=dict(size=3, color='blue', opacity=0.5), showlegend=False,
            name="Y = 2X + noise"), row=1, col=1)
        fig.add_trace(go.Scatter(x=X_b, y=Y_b, mode='markers',
            marker=dict(size=3, color='green', opacity=0.5), showlegend=False,
            name="X~N, Y~Laplace"), row=1, col=2)
        fig.add_trace(go.Scatter(x=X_c, y=Y_c, mode='markers',
            marker=dict(size=3, color='red', opacity=0.5), showlegend=False,
            name="Y = X²"), row=1, col=3)

        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y = 2X + noise", row=1, col=1)
        fig.update_xaxes(title_text="X ~ N(0,1)", row=1, col=2)
        fig.update_yaxes(title_text="Y ~ Laplace(0,1)", row=1, col=2)
        fig.update_xaxes(title_text="X ~ U(-1,1)", row=1, col=3)
        fig.update_yaxes(title_text="Y = X²", row=1, col=3)

        fig.update_layout(height=450, title_text="Three Cases: Correlation vs Independence")
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("### 📋 Summary")
        summary_data = {
            "Example": ["A: Y = 2X + noise", "B: X~N, Y~Laplace", "C: Y = X²"],
            "Correlation (ρ)": [f"{rho_a:.3f}", f"{rho_b:.3f}", f"{rho_c:.3f}"],
            "Mutual Info (bits)": [f"{mi_a:.3f}", f"{mi_b:.3f}", f"{mi_c:.3f}"],
            "Correlated?": ["✅ Yes", "❌ No", "❌ No"],
            "Dependent?": ["✅ Yes", "❌ No", "✅ Yes"],
        }
        st.table(summary_data)

    with st.expander("📖 Mathematical Definitions"):
        st.markdown(r"""
**Pearson Correlation:**
$$\rho(X, Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \cdot \sum(y_i - \bar{y})^2}}$$

- Measures **linear** dependence only
- $\rho = 0$ means uncorrelated, but NOT necessarily independent

**Mutual Information:**
$$I(X; Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)\,p(y)}$$

- Measures **any** statistical dependence (linear or nonlinear)
- $I = 0 \Leftrightarrow X$ and $Y$ are independent
- Always $I \geq 0$

**Why this matters for ICA:**
- PCA only decorrelates → sets $\rho = 0$ → misses nonlinear dependencies
- ICA minimizes mutual information → achieves true independence
        """)
