"""
ICA Geometric Understanding
============================
Visualize ICA as a 3-step geometric transformation on images:
1. Rotation (PCA-like) to align principal axes
2. Scaling to equalize variances (whitening)
3. Kurtosis-based rotation to find independent components

Based on ICA_Geometric.ipynb and FastICA-ImageSeparation.ipynb
"""

import streamlit as st
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

st.set_page_config(page_title="ICA Geometric", page_icon="📐", layout="wide")

# ===== STYLING =====
st.markdown("""
<style>
    .ica-header {
        background: linear-gradient(135deg, #059669 0%, #0891b2 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;
    }
    .ica-header h1 { color: white; margin: 0; }
    .ica-header p  { color: #d1fae5; margin: 0.5rem 0 0 0; }
    .info-panel {
        background-color: #ecfdf5; border-left: 4px solid #059669;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
    .step-panel {
        background-color: #f0f9ff; border-left: 4px solid #0891b2;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;
    }
    .success-panel {
        background-color: #f0fff4; border-left: 4px solid #38a169;
        padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ica-header">
    <h1>📐 Geometric Understanding of ICA</h1>
    <p>Visualize ICA as three geometric transformations: Rotation → Scaling → Kurtosis Rotation</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────  DATA LOADING  ──────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_data
def load_mixed_images():
    """Load and preprocess the mixed grayscale images."""
    img1_path = DATA_DIR / "mixed1.jpg"
    img2_path = DATA_DIR / "mixed2.jpg"

    if not img1_path.exists() or not img2_path.exists():
        return None, None, "Mixed image files not found in data/"

    img1 = np.array(Image.open(img1_path).convert('L')).astype(float)
    img2 = np.array(Image.open(img2_path).convert('L')).astype(float)

    # Ensure same size
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    return img1, img2, None

img1, img2, err = load_mixed_images()

if err:
    st.error(err)
    st.stop()

M, N = img1.shape

# ═══════════════════════  TABS  ═══════════════════════
tab_overview, tab_step1, tab_step2, tab_step3, tab_result = st.tabs([
    "🔍 Overview",
    "🔄 Step 1: Rotation (PCA)",
    "📏 Step 2: Scaling",
    "📐 Step 3: Kurtosis Rotation",
    "✅ Recovered Images"
])


# ══════════════════════════════════════════════════════
# TAB: Overview
# ══════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## The Geometric ICA Pipeline")

    st.markdown("""
<div class="info-panel">

**Problem:** We observe two mixed images. Each pixel is a linear combination of two unknown source images.

**Goal:** Recover the original source images using only the mixtures — <b>Blind Source Separation</b>.

**Geometric insight:** The joint scatter plot of pixel intensities forms a parallelogram.  
ICA transforms this parallelogram back to a rectangle aligned with independent axes.

</div>
""", unsafe_allow_html=True)

    # Show mixed images
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1 / 255.0, caption="Mixed Image 1", use_container_width=True, clamp=True)
    with col2:
        st.image(img2 / 255.0, caption="Mixed Image 2", use_container_width=True, clamp=True)

    # Scatter plot of mixed pixel values
    x1_flat = img1.flatten()
    x2_flat = img2.flatten()

    # Subsample for plotting performance
    n_pixels = len(x1_flat)
    subsample = min(20000, n_pixels)
    idx = np.random.default_rng(42).choice(n_pixels, subsample, replace=False)

    fig = go.Figure(go.Scatter(
        x=x1_flat[idx], y=x2_flat[idx], mode='markers',
        marker=dict(size=2, color='rgba(99,102,241,0.3)'),
    ))
    fig.update_layout(
        title="Scatter Plot of Mixed Image Pixel Intensities",
        xaxis_title="Mixed Image 1", yaxis_title="Mixed Image 2",
        height=500, xaxis=dict(scaleanchor="y")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="step-panel">

**Three Geometric Steps:**

1. **Rotation** — Rotate to align the principal axes (like PCA). Angle θ₀ from variance maximization.
2. **Scaling** — Scale each axis by 1/σ to equalize variances (whitening). Makes the cloud circular.
3. **Kurtosis Rotation** — Rotate by angle φ₀ that maximizes kurtosis → finds independent directions.

</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# Compute all steps
# ══════════════════════════════════════════════════════
x1v = (x1_flat - np.mean(x1_flat)).reshape(-1, 1)
x2v = (x2_flat - np.mean(x2_flat)).reshape(-1, 1)

# Step 1: Rotation angle
theta0 = 0.5 * np.arctan2(-2 * np.sum(x1v * x2v), np.sum(x1v**2 - x2v**2))
Ustar = np.array([[np.cos(theta0), np.sin(theta0)],
                   [-np.sin(theta0), np.cos(theta0)]])

# Apply rotation to images
X1_rot = Ustar[0, 0] * img1 + Ustar[0, 1] * img2
X2_rot = Ustar[1, 0] * img1 + Ustar[1, 1] * img2

# Step 2: Compute variances along rotated axes
sigma1 = np.sum((x1v * np.cos(theta0) + x2v * np.sin(theta0))**2)
sigma2 = np.sum((x1v * np.cos(theta0 - np.pi / 2) + x2v * np.sin(theta0 - np.pi / 2))**2)

Sigma_inv = np.array([[1 / np.sqrt(sigma1), 0],
                       [0, 1 / np.sqrt(sigma2)]])

# Apply scaling
X1_scaled = Sigma_inv[0, 0] * X1_rot
X2_scaled = Sigma_inv[1, 1] * X2_rot

# Step 3: Kurtosis rotation
x1vbar = X1_scaled.flatten().reshape(-1, 1)
x2vbar = X2_scaled.flatten().reshape(-1, 1)

numerator = -np.sum(2 * (x1vbar**3) * x2vbar - 2 * x1vbar * (x2vbar**3))
denominator = np.sum(3 * (x1vbar**2) * (x2vbar**2) - 0.5 * (x1vbar**4) - 0.5 * (x2vbar**4))
phi0 = 0.25 * np.arctan2(numerator, denominator)

V = np.array([[np.cos(phi0), np.sin(phi0)],
              [-np.sin(phi0), np.cos(phi0)]])

m1 = V[0, 0] * X1_scaled + V[0, 1] * X2_scaled
m2 = V[1, 0] * X1_scaled + V[1, 1] * X2_scaled

# Normalize to [0, 1]
S1_hat = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))
S2_hat = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2))


# ══════════════════════════════════════════════════════
# TAB: Step 1 — Rotation
# ══════════════════════════════════════════════════════
with tab_step1:
    st.markdown("## Step 1: Rotation — Align Principal Axes")

    st.markdown(rf"""
<div class="step-panel">

**Objective:** Find the angle θ₀ that aligns the data cloud with the coordinate axes (removes correlation between axes).

**Formula:**
$$\theta_0 = \frac{{1}}{{2}} \arctan\left(\frac{{-2 \sum x_1 x_2}}{{\sum(x_1^2 - x_2^2)}}\right)$$

**Computed:** θ₀ = {np.degrees(theta0):.2f}°

</div>
""", unsafe_allow_html=True)

    st.latex(rf"U^* = \begin{{bmatrix}} \cos(\theta_0) & \sin(\theta_0) \\ -\sin(\theta_0) & \cos(\theta_0) \end{{bmatrix}} = \begin{{bmatrix}} {Ustar[0,0]:.4f} & {Ustar[0,1]:.4f} \\ {Ustar[1,0]:.4f} & {Ustar[1,1]:.4f} \end{{bmatrix}}")

    # Scatter: before vs after rotation
    x1r = X1_rot.flatten()
    x2r = X2_rot.flatten()

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Before Rotation (ρ={np.corrcoef(x1_flat, x2_flat)[0,1]:.3f})",
        f"After Rotation (ρ={np.corrcoef(x1r, x2r)[0,1]:.4f})"
    ])
    fig.add_trace(go.Scatter(x=x1_flat[idx], y=x2_flat[idx], mode='markers',
        marker=dict(size=2, color='rgba(99,102,241,0.3)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x1r[idx], y=x2r[idx], mode='markers',
        marker=dict(size=2, color='rgba(14,165,233,0.3)'), showlegend=False), row=1, col=2)
    fig.update_layout(height=450, title_text="Effect of Rotation")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB: Step 2 — Scaling
# ══════════════════════════════════════════════════════
with tab_step2:
    st.markdown("## Step 2: Scaling — Equalize Variances (Whitening)")

    st.markdown(rf"""
<div class="step-panel">

**Objective:** Scale each axis so that variance = 1 in all directions. This makes the data cloud circular.

**Variance ratio:** σ₁/σ₂ = {sigma1/sigma2:.4f}

$$\Sigma^{{-1}} = \begin{{bmatrix}} 1/\sqrt{{\sigma_1}} & 0 \\ 0 & 1/\sqrt{{\sigma_2}} \end{{bmatrix}}$$

</div>
""", unsafe_allow_html=True)

    x1s = X1_scaled.flatten()
    x2s = X2_scaled.flatten()

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "After Rotation (elliptical)",
        "After Scaling (circular = whitened)"
    ])
    fig.add_trace(go.Scatter(x=x1r[idx], y=x2r[idx], mode='markers',
        marker=dict(size=2, color='rgba(14,165,233,0.3)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x1s[idx], y=x2s[idx], mode='markers',
        marker=dict(size=2, color='rgba(16,185,129,0.3)'), showlegend=False), row=1, col=2)
    fig.update_layout(height=450, title_text="Effect of Scaling (Whitening)")
    for i in range(1, 3):
        fig.update_xaxes(scaleanchor=f"y{i}" if i > 1 else "y", row=1, col=i)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB: Step 3 — Kurtosis Rotation
# ══════════════════════════════════════════════════════
with tab_step3:
    st.markdown("## Step 3: Kurtosis-Based Rotation — Find Independent Directions")

    st.markdown(rf"""
<div class="step-panel">

**Objective:** After whitening, find the rotation angle φ₀ that maximizes kurtosis of the projected data.

**Formula (normalized kurtosis extremum):**
$$\phi_0 = \frac{{1}}{{4}} \arctan\left(\frac{{-\sum[2\bar{{x}}_1^3 \bar{{x}}_2 - 2\bar{{x}}_1 \bar{{x}}_2^3]}}{{\sum[3\bar{{x}}_1^2 \bar{{x}}_2^2 - 0.5\bar{{x}}_1^4 - 0.5\bar{{x}}_2^4]}}\right)$$

**Computed:** φ₀ = {np.degrees(phi0):.2f}°

</div>
""", unsafe_allow_html=True)

    # Interactive: let user adjust phi manually and compare
    phi_manual = st.slider(
        "Adjust rotation angle φ (degrees)",
        -90.0, 90.0, float(np.degrees(phi0)), 0.5,
        key="phi_slider",
        help="The optimal value is pre-computed. Move the slider to see what happens with other angles."
    )

    phi_rad = np.deg2rad(phi_manual)
    V_manual = np.array([[np.cos(phi_rad), np.sin(phi_rad)],
                          [-np.sin(phi_rad), np.cos(phi_rad)]])
    m1_man = V_manual[0, 0] * X1_scaled + V_manual[0, 1] * X2_scaled
    m2_man = V_manual[1, 0] * X1_scaled + V_manual[1, 1] * X2_scaled

    # Kurtosis landscape for whitened data
    phi_range = np.linspace(-90, 90, 361)
    kurt_landscape = []
    for p in phi_range:
        pr = np.deg2rad(p)
        proj = np.cos(pr) * x1s + np.sin(pr) * x2s
        kurt_landscape.append(kurtosis(proj, fisher=True))
    kurt_landscape = np.array(kurt_landscape)

    # Current kurtosis
    proj_curr = np.cos(phi_rad) * x1s + np.sin(phi_rad) * x2s
    kurt_curr = kurtosis(proj_curr, fisher=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Kurtosis Landscape (current φ={phi_manual:.1f}°, Kurt={kurt_curr:.3f})",
        "Recovered Image Preview"
    ])
    fig.add_trace(go.Scatter(x=phi_range, y=kurt_landscape, mode='lines',
        line=dict(color='#6366f1', width=2), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[phi_manual], y=[kurt_curr], mode='markers',
        marker=dict(size=12, color='red', symbol='circle'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[np.degrees(phi0)], y=[kurtosis(np.cos(phi0)*x1s + np.sin(phi0)*x2s, fisher=True)],
        mode='markers', marker=dict(size=10, color='green', symbol='star'),
        name=f'Optimal φ₀={np.degrees(phi0):.1f}°', showlegend=True), row=1, col=1)
    fig.update_xaxes(title_text="φ (degrees)", row=1, col=1)
    fig.update_yaxes(title_text="Kurtosis", row=1, col=1)

    # Image preview
    m1_norm = (m1_man - np.min(m1_man)) / (np.max(m1_man) - np.min(m1_man) + 1e-10)
    fig.add_trace(go.Heatmap(z=np.flipud(m1_norm), colorscale='gray', showscale=False), row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB: Recovered Images
# ══════════════════════════════════════════════════════
with tab_result:
    st.markdown("## Recovered Source Images")

    st.markdown("""
<div class="success-panel">
✅ <b>Geometric ICA complete!</b> The three transformations (rotation + scaling + kurtosis rotation) recovered the independent source images from their mixtures.
</div>
""", unsafe_allow_html=True)

    # Side-by-side: mixed vs recovered
    st.markdown("### Mixed Images (Observed)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1 / 255.0, caption="Mixed Image 1", use_container_width=True, clamp=True)
    with col2:
        st.image(img2 / 255.0, caption="Mixed Image 2", use_container_width=True, clamp=True)

    st.markdown("### Recovered Source Images")
    col3, col4 = st.columns(2)
    with col3:
        st.image(S1_hat, caption="Recovered Source 1", use_container_width=True, clamp=True)
    with col4:
        st.image(S2_hat, caption="Recovered Source 2", use_container_width=True, clamp=True)

    # Statistics
    st.markdown("### 📊 Separation Quality")
    rho_mixed = np.corrcoef(x1_flat, x2_flat)[0, 1]
    rho_rec = np.corrcoef(S1_hat.flatten(), S2_hat.flatten())[0, 1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Correlation (mixed)", f"{rho_mixed:.4f}")
    with c2:
        st.metric("Correlation (recovered)", f"{rho_rec:.4f}")
    with c3:
        st.metric("Improvement", f"{abs(rho_mixed) - abs(rho_rec):.4f}", delta=f"{(1 - abs(rho_rec)/abs(rho_mixed))*100:.1f}%")

    with st.expander("📖 Full Transformation Pipeline"):
        st.markdown(rf"""
**Step 1 — Rotation:** θ₀ = {np.degrees(theta0):.2f}° → decorrelates axes  
**Step 2 — Scaling:** σ₁/σ₂ = {sigma1/sigma2:.2f} → equalizes variances (whitening)  
**Step 3 — Kurtosis rotation:** φ₀ = {np.degrees(phi0):.2f}° → finds independent directions

**Combined transformation:**
$$\mathbf{{S}} = \mathbf{{V}} \cdot \Sigma^{{-1}} \cdot \mathbf{{U}}^* \cdot \mathbf{{X}}$$

This factorization gives **geometric meaning** to each step of ICA:
- $\mathbf{{U}}^*$: PCA rotation (align principal axes)
- $\Sigma^{{-1}}$: variance normalization (sphering)
- $\mathbf{{V}}$: kurtosis-based rotation (find independence)
        """)

    with st.expander("⚠️ ICA Ambiguities"):
        st.markdown("""
| Ambiguity | Explanation |
|-----------|-------------|
| **Permutation** | Which recovered image maps to which original source is unknown |
| **Sign** | Images may be inverted (negative) |
| **Scale** | Brightness/contrast may differ |

These are **fundamental** to ICA and cannot be resolved without additional information.
        """)
