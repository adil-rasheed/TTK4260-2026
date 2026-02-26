"""
ICA By Hand — Step-by-Step Tutorial
====================================
Build ICA from scratch: generate sources, mix them, preprocess,
explore kurtosis interactively, and recover independent components.

Based on ICA_ByHand.ipynb
"""

import streamlit as st
import numpy as np
import math
from collections import Counter

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, norm

st.set_page_config(page_title="ICA By Hand", page_icon="🛠️", layout="wide")

# ===== STYLING =====
st.markdown("""
<style>
    .ica-header {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;
    }
    .ica-header h1 { color: white; margin: 0; }
    .ica-header p  { color: #fef3c7; margin: 0.5rem 0 0 0; }
    .info-panel {
        background-color: #fff7ed; border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
    .success-panel {
        background-color: #f0fff4; border-left: 4px solid #38a169;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ica-header">
    <h1>🛠️ ICA By Hand</h1>
    <p>Build Independent Component Analysis step by step from scratch</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────  HELPERS  ──────────────────────────
def correlation_np(x, y):
    return np.corrcoef(x, y)[0, 1]

def mutual_information(x, y, bins=30):
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


# ═══════════════════════  TABS  ═══════════════════════
tab_gen, tab_mix, tab_pre, tab_kurt, tab_recover = st.tabs([
    "🎵 Generate Sources",
    "🔀 Mix Signals",
    "⚙️ Preprocessing",
    "📐 Kurtosis Exploration",
    "✅ Recover Components",
])

# ──────────────────────────────────────────────────────
# Sidebar: global parameters
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ ICA Parameters")
    n_samples = st.slider("Number of samples", 500, 10000, 2000, 500, key="ica_n")
    seed = st.number_input("Random seed", 0, 9999, 0, key="ica_seed")
    st.markdown("---")
    st.markdown("### Mixing Matrix A")
    a11 = st.number_input("A[1,1]", value=1.0, step=0.1, key="a11")
    a12 = st.number_input("A[1,2]", value=0.5, step=0.1, key="a12")
    a21 = st.number_input("A[2,1]", value=0.0, step=0.1, key="a21")
    a22 = st.number_input("A[2,2]", value=1.0, step=0.1, key="a22")
    A = np.array([[a11, a12], [a21, a22]])
    st.markdown("---")
    src1_dist = st.selectbox("Source 1 distribution", ["Gaussian", "Laplace", "Uniform"], index=0, key="src1")
    src2_dist = st.selectbox("Source 2 distribution", ["Laplace", "Gaussian", "Uniform"], index=0, key="src2")

# ──────────────────────────────────────────────────────
# Generate data (shared across tabs via session state)
# ──────────────────────────────────────────────────────
rng = np.random.default_rng(seed)

def make_source(dist, n, rng):
    if dist == "Gaussian":
        return rng.normal(0, 1, n)
    elif dist == "Laplace":
        return rng.laplace(0, 1, n)
    elif dist == "Uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    return rng.normal(0, 1, n)

s1 = make_source(src1_dist, n_samples, rng)
s2 = make_source(src2_dist, n_samples, rng)
S = np.c_[s1, s2]

# Mix
X = S @ A.T
x1, x2 = X[:, 0], X[:, 1]

# Preprocess
X_centered = X - X.mean(axis=0)
cov = np.cov(X_centered.T)
E, D, _ = np.linalg.svd(cov)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
X_white = X_centered @ E @ D_inv_sqrt @ E.T

# Kurtosis landscape
angles_fine = np.linspace(0, 180, 361)
kurt_vals_fine = []
for a in angles_fine:
    theta = np.deg2rad(a)
    w = np.array([np.cos(theta), np.sin(theta)])
    proj = X_white @ w
    kurt_vals_fine.append(kurtosis(proj, fisher=True))
kurt_vals_fine = np.array(kurt_vals_fine)

best_idx = np.argmax(np.abs(kurt_vals_fine))
best_angle = angles_fine[best_idx]

# ══════════════════════════════════════════════════════
# TAB 1: Generate Sources
# ══════════════════════════════════════════════════════
with tab_gen:
    st.markdown("## Part I: Generate Independent Source Signals")
    st.markdown(f"""
<div class="info-panel">
Two independent source signals: <b>s₁ ~ {src1_dist}</b> and <b>s₂ ~ {src2_dist}</b>, with {n_samples:,} samples each.
</div>
""", unsafe_allow_html=True)

    # Time series
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Source 1 (s₁)", "Source 2 (s₂)"],
                        vertical_spacing=0.12)
    show_pts = min(500, n_samples)
    fig.add_trace(go.Scatter(y=s1[:show_pts], mode='lines', line=dict(color='#3b82f6', width=1),
                             name='s₁'), row=1, col=1)
    fig.add_trace(go.Scatter(y=s2[:show_pts], mode='lines', line=dict(color='#ef4444', width=1),
                             name='s₂'), row=2, col=1)
    fig.update_layout(height=350, margin=dict(t=40, b=30))
    fig.update_xaxes(title_text="Sample index", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter + stats
    col_sc, col_st = st.columns([2, 1])
    with col_sc:
        fig2 = go.Figure(go.Scatter(x=s1, y=s2, mode='markers',
            marker=dict(size=2, color='#8b5cf6', opacity=0.3)))
        rho_src = correlation_np(s1, s2)
        mi_src = mutual_information(s1, s2, bins=30)
        fig2.update_layout(title=f"Source Scatter (ρ={rho_src:.4f}, MI={mi_src:.4f} bits)",
                           xaxis_title="s₁", yaxis_title="s₂", height=400,
                           xaxis=dict(scaleanchor="y"))
        st.plotly_chart(fig2, use_container_width=True)
    with col_st:
        st.markdown("### 📊 Statistics")
        st.metric("Correlation ρ", f"{rho_src:.4f}")
        st.metric("Mutual Information", f"{mi_src:.4f} bits")
        st.metric("Kurtosis s₁", f"{kurtosis(s1, fisher=True):.3f}")
        st.metric("Kurtosis s₂", f"{kurtosis(s2, fisher=True):.3f}")
        st.info("**Independent sources**: low ρ and low MI. Kurtosis ≠ 0 for non-Gaussian distributions.")


# ══════════════════════════════════════════════════════
# TAB 2: Mixing
# ══════════════════════════════════════════════════════
with tab_mix:
    st.markdown("## Part II: Mix the Signals")

    st.markdown(r"""
<div class="info-panel">
The mixing model: $\mathbf{X} = \mathbf{S}\,\mathbf{A}^T$. The mixing matrix <b>A</b> is set in the sidebar.
</div>
""", unsafe_allow_html=True)

    col_m, col_ms = st.columns([1, 1])
    with col_m:
        st.markdown("### Mixing Matrix A")
        st.latex(rf"A = \begin{{bmatrix}} {a11:.1f} & {a12:.1f} \\ {a21:.1f} & {a22:.1f} \end{{bmatrix}}")

    with col_ms:
        rho_mix = correlation_np(x1, x2)
        mi_mix = mutual_information(x1, x2, bins=30)
        st.markdown("### Mixed Signal Statistics")
        st.metric("Correlation ρ (mixed)", f"{rho_mix:.4f}")
        st.metric("Mutual Information (mixed)", f"{mi_mix:.4f} bits")

    # Time series
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Mixed Signal x₁", "Mixed Signal x₂"],
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(y=x1[:show_pts], mode='lines', line=dict(color='purple', width=1),
                             name='x₁'), row=1, col=1)
    fig.add_trace(go.Scatter(y=x2[:show_pts], mode='lines', line=dict(color='orange', width=1),
                             name='x₂'), row=2, col=1)
    fig.update_layout(height=350, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Scatter comparison
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Original Sources (ρ={rho_src:.3f})",
        f"Mixed Signals (ρ={rho_mix:.3f})"
    ])
    fig3.add_trace(go.Scatter(x=s1, y=s2, mode='markers',
        marker=dict(size=2, color='#8b5cf6', opacity=0.3), showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=x1, y=x2, mode='markers',
        marker=dict(size=2, color='#f97316', opacity=0.3), showlegend=False), row=1, col=2)
    fig3.update_layout(height=400)
    fig3.update_xaxes(title_text="s₁", row=1, col=1)
    fig3.update_yaxes(title_text="s₂", row=1, col=1)
    fig3.update_xaxes(title_text="x₁", row=1, col=2)
    fig3.update_yaxes(title_text="x₂", row=1, col=2)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3: Preprocessing
# ══════════════════════════════════════════════════════
with tab_pre:
    st.markdown("## Part III: Centering & Whitening")

    st.markdown(r"""
<div class="info-panel">

**Step 1 — Centering:** $\mathbf{X}_c = \mathbf{X} - \mathbb{E}[\mathbf{X}]$  
**Step 2 — Whitening:** $\mathbf{X}_w = \mathbf{X}_c \, \mathbf{E} \, \mathbf{D}^{-1/2} \, \mathbf{E}^T$ (via SVD of covariance)

Whitening decorrelates the data and normalizes variance → reduces ICA to finding an **orthogonal rotation**.

</div>
""", unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        f"Original (ρ={correlation_np(x1, x2):.3f})",
        f"Centered (ρ={np.corrcoef(X_centered.T)[0,1]:.3f})",
        f"Whitened (ρ={np.corrcoef(X_white.T)[0,1]:.2e})"
    ], horizontal_spacing=0.08)

    for idx, (data, color) in enumerate([
        (X, '#f97316'), (X_centered, '#3b82f6'), (X_white, '#10b981')
    ]):
        fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers',
            marker=dict(size=2, color=color, opacity=0.3), showlegend=False),
            row=1, col=idx + 1)

    fig.update_layout(height=450, title_text="Preprocessing Pipeline: Original → Centered → Whitened")
    for i in range(1, 4):
        fig.update_xaxes(title_text="Comp 1", row=1, col=i)
        fig.update_yaxes(title_text="Comp 2", row=1, col=i, scaleanchor=f"x{i}" if i > 1 else "x")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Covariance Matrices"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before whitening:**")
            cov_before = np.cov(X_centered.T)
            st.write(np.round(cov_before, 4))
        with col2:
            st.markdown("**After whitening:**")
            cov_after = np.cov(X_white.T)
            st.write(np.round(cov_after, 4))


# ══════════════════════════════════════════════════════
# TAB 4: Kurtosis Exploration
# ══════════════════════════════════════════════════════
with tab_kurt:
    st.markdown("## Part IV: Interactive Kurtosis Exploration")

    st.markdown(r"""
<div class="info-panel">

**Kurtosis** measures non-Gaussianity: $\text{Kurt}(X) = E\!\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3$

- Kurt = 0 → Gaussian | Kurt > 0 → super-Gaussian (heavy tails) | Kurt < 0 → sub-Gaussian (light tails)

**ICA principle:** Independent components lie at angles of **maximum |kurtosis|**!

</div>
""", unsafe_allow_html=True)

    angle_deg = st.slider("Projection angle θ (degrees)", 0, 180, 0, 1, key="kurt_angle")

    theta = np.deg2rad(angle_deg)
    w_vec = np.array([np.cos(theta), np.sin(theta)])
    proj = X_white @ w_vec
    kurt_val = kurtosis(proj, fisher=True)

    # 3-panel plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "Whitened Data + Projection Direction",
        f"Projection Histogram (θ={angle_deg}°, Kurt={kurt_val:.3f})",
        "Kurtosis vs Projection Angle"
    ], horizontal_spacing=0.08)

    # Panel 1: scatter + axis
    fig.add_trace(go.Scatter(x=X_white[:, 0], y=X_white[:, 1], mode='markers',
        marker=dict(size=2, color='gray', opacity=0.2), showlegend=False), row=1, col=1)
    line_len = 3
    fig.add_trace(go.Scatter(
        x=[-line_len * w_vec[0], line_len * w_vec[0]],
        y=[-line_len * w_vec[1], line_len * w_vec[1]],
        mode='lines', line=dict(color='red', width=3), name='Projection axis', showlegend=False
    ), row=1, col=1)
    fig.update_xaxes(range=[-4, 4], row=1, col=1, scaleanchor="y")
    fig.update_yaxes(range=[-4, 4], row=1, col=1)

    # Panel 2: histogram + Gaussian
    hist_vals, bin_edges = np.histogram(proj, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mu_p, sigma_p = np.mean(proj), np.std(proj)
    x_g = np.linspace(proj.min(), proj.max(), 200)
    fig.add_trace(go.Bar(x=bin_centers, y=hist_vals, marker_color='rgba(147,51,234,0.6)',
                         showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_g, y=norm.pdf(x_g, mu_p, sigma_p), mode='lines',
        line=dict(color='black', width=2, dash='dash'), showlegend=False), row=1, col=2)

    # Panel 3: kurtosis curve + marker
    fig.add_trace(go.Scatter(x=angles_fine, y=kurt_vals_fine, mode='lines',
        line=dict(color='#3b82f6', width=2), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[angle_deg], y=[kurt_val], mode='markers',
        marker=dict(size=12, color='red', symbol='circle'), showlegend=False), row=1, col=3)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=3)

    fig.update_layout(height=420, margin=dict(t=50, b=30))
    fig.update_xaxes(title_text="Angle (°)", row=1, col=3)
    fig.update_yaxes(title_text="Kurtosis", row=1, col=3)
    st.plotly_chart(fig, use_container_width=True)

    # Best angle info
    st.markdown(f"""
<div class="success-panel">
🎯 <b>Best angle (max |kurtosis|):</b> θ = {best_angle:.1f}° — Kurtosis = {kurt_vals_fine[best_idx]:.3f}<br>
🔄 Orthogonal component at θ = {(best_angle + 90) % 180:.1f}°
</div>
""", unsafe_allow_html=True)

    with st.expander("📖 Why maximize kurtosis?"):
        st.markdown(r"""
By the **Central Limit Theorem**, linear mixtures of independent signals are *more Gaussian* than the original signals.

Therefore, the direction of **maximum non-Gaussianity** (maximum |kurtosis|) corresponds to one of the independent source signals.

The second source is at the **orthogonal** direction (90° away), because whitening ensures the problem reduces to finding a rotation.
        """)


# ══════════════════════════════════════════════════════
# TAB 5: Recovery
# ══════════════════════════════════════════════════════
with tab_recover:
    st.markdown("## Part V: Recover Independent Components")

    theta_best = np.deg2rad(best_angle)
    w1 = np.array([np.cos(theta_best), np.sin(theta_best)])
    w2 = np.array([-np.sin(theta_best), np.cos(theta_best)])
    W = np.vstack([w1, w2])
    S_est = X_white @ W.T

    st.markdown(f"""
<div class="info-panel">
Using best angle θ = {best_angle:.1f}°, the unmixing matrix is:
</div>
""", unsafe_allow_html=True)

    st.latex(rf"W = \begin{{bmatrix}} \cos({best_angle:.1f}°) & \sin({best_angle:.1f}°) \\ -\sin({best_angle:.1f}°) & \cos({best_angle:.1f}°) \end{{bmatrix}} = \begin{{bmatrix}} {w1[0]:.4f} & {w1[1]:.4f} \\ {w2[0]:.4f} & {w2[1]:.4f} \end{{bmatrix}}")

    # Comparison: original vs recovered (time series)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Original s₁", "Recovered ŝ₁",
        "Original s₂", "Recovered ŝ₂"
    ], vertical_spacing=0.12, horizontal_spacing=0.08)

    fig.add_trace(go.Scatter(y=s1[:show_pts], mode='lines', line=dict(color='#3b82f6', width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=S_est[:show_pts, 0], mode='lines', line=dict(color='#10b981', width=1),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=s2[:show_pts], mode='lines', line=dict(color='#ef4444', width=1),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=S_est[:show_pts, 1], mode='lines', line=dict(color='#10b981', width=1),
                             showlegend=False), row=2, col=2)
    fig.update_layout(height=400, title_text="Original Sources vs Recovered Components")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter comparison
    rho_rec = correlation_np(S_est[:, 0], S_est[:, 1])
    mi_rec = mutual_information(S_est[:, 0], S_est[:, 1], bins=30)

    fig2 = make_subplots(rows=1, cols=3, subplot_titles=[
        f"Original Sources (ρ={rho_src:.3f})",
        f"Mixed Signals (ρ={rho_mix:.3f})",
        f"Recovered (ρ={rho_rec:.3f})"
    ], horizontal_spacing=0.08)

    fig2.add_trace(go.Scatter(x=s1, y=s2, mode='markers',
        marker=dict(size=2, color='#8b5cf6', opacity=0.3), showlegend=False), row=1, col=1)
    fig2.add_trace(go.Scatter(x=x1, y=x2, mode='markers',
        marker=dict(size=2, color='#f97316', opacity=0.3), showlegend=False), row=1, col=2)
    fig2.add_trace(go.Scatter(x=S_est[:, 0], y=S_est[:, 1], mode='markers',
        marker=dict(size=2, color='#10b981', opacity=0.3), showlegend=False), row=1, col=3)
    fig2.update_layout(height=400, title_text="Scatter Comparison: Sources → Mixed → Recovered")
    st.plotly_chart(fig2, use_container_width=True)

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Original Sources")
        st.metric("ρ", f"{rho_src:.4f}")
        st.metric("MI", f"{mi_src:.4f} bits")
    with c2:
        st.markdown("#### Mixed Signals")
        st.metric("ρ", f"{rho_mix:.4f}")
        st.metric("MI", f"{mi_mix:.4f} bits")
    with c3:
        st.markdown("#### Recovered Components")
        st.metric("ρ", f"{rho_rec:.4f}")
        st.metric("MI", f"{mi_rec:.4f} bits")

    st.markdown("""
<div class="success-panel">
✅ <b>ICA successfully recovered the independent components!</b><br>
Correlation and mutual information decreased from the mixed signals back to near-zero levels.
</div>
""", unsafe_allow_html=True)

    with st.expander("⚠️ ICA Ambiguities"):
        st.markdown("""
| Ambiguity | Explanation |
|-----------|-------------|
| **Permutation** | The order of recovered components is arbitrary |
| **Sign** | Components may be negated (multiplied by −1) |
| **Scale** | The amplitude may differ from the original |

These are **inherent limitations** of ICA — the algorithm can only recover sources up to these ambiguities.
        """)
