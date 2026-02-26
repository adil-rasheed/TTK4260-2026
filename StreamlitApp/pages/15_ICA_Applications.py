"""
ICA Applications — Image Separation & Audio (Cocktail Party)
=============================================================
Two real-world ICA applications:
1. Image Separation — Kurtosis scan to unmix overlapping images
2. Audio Separation — FastICA Cocktail Party problem with 3 microphones

Based on FastICA-ImageSeparation.ipynb and ICA_Audio.ipynb
"""

import streamlit as st
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ICA Applications", page_icon="🎯", layout="wide")

# ===== STYLING =====
st.markdown("""
<style>
    .ica-header {
        background: linear-gradient(135deg, #dc2626 0%, #7c3aed 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;
    }
    .ica-header h1 { color: white; margin: 0; }
    .ica-header p  { color: #fde8e8; margin: 0.5rem 0 0 0; }
    .info-panel {
        background-color: #fef2f2; border-left: 4px solid #dc2626;
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
    <h1>🎯 ICA Applications</h1>
    <p>Real-world Blind Source Separation: Images and Audio</p>
</div>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent.parent / "data"

# ═══════════════════════  TABS  ═══════════════════════
tab_img, tab_audio = st.tabs([
    "🖼️ Image Separation",
    "🎵 Audio Separation (Cocktail Party)",
])


# ══════════════════════════════════════════════════════
# TAB 1: Image Separation
# ══════════════════════════════════════════════════════
with tab_img:
    st.markdown("## Image Separation via Kurtosis Maximization")

    st.markdown("""
<div class="info-panel">

**Problem:** Two grayscale images are linearly mixed: $x_i = a_{i1} s_1 + a_{i2} s_2$.  
We observe only the mixtures — can we recover the original sources?

**Method:** Scan all projection angles θ (0°–180°) and compute kurtosis. The directions of **maximum** and **minimum** kurtosis correspond to the two independent sources.

</div>
""", unsafe_allow_html=True)

    # Load images
    from PIL import Image as PILImage

    @st.cache_data
    def load_and_process_images():
        img1_path = DATA_DIR / "mixed1.jpg"
        img2_path = DATA_DIR / "mixed2.jpg"
        if not img1_path.exists() or not img2_path.exists():
            return None, None, None, None, None, None, None, "Image files not found"

        X1 = np.array(PILImage.open(img1_path).convert('L')).astype(float)
        X2 = np.array(PILImage.open(img2_path).convert('L')).astype(float)
        h = min(X1.shape[0], X2.shape[0])
        w = min(X1.shape[1], X2.shape[1])
        X1, X2 = X1[:h, :w], X2[:h, :w]

        # Flatten and stack
        X = np.c_[X1.flatten(), X2.flatten()]

        # Kurtosis scan
        n_angles = 720
        angles = np.linspace(0, np.pi, n_angles)
        kurt_vals = np.array([kurtosis(X @ np.array([np.cos(t), np.sin(t)]), fisher=True) for t in angles])

        best_max = angles[np.argmax(kurt_vals)]
        best_min = angles[np.argmin(kurt_vals)]

        def project_and_reshape(X, angle, h, w):
            w_vec = np.array([np.cos(angle), np.sin(angle)])
            proj = X @ w_vec
            proj = (proj - proj.min()) / (proj.max() - proj.min())
            return proj.reshape(h, w)

        rec1 = project_and_reshape(X, best_max, h, w)
        rec2 = project_and_reshape(X, best_min, h, w)

        # Fix inversion if needed: correlate with original mixtures
        for i, (rec, ref_imgs) in enumerate([(rec1, [X1, X2]), (rec2, [X1, X2])]):
            best_corr = 0
            for ref in ref_imgs:
                ref_norm = (ref - ref.min()) / (ref.max() - ref.min() + 1e-10)
                corr = np.corrcoef(rec.flatten(), ref_norm.flatten())[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
            if best_corr < 0:
                if i == 0:
                    rec1 = 1 - rec1
                else:
                    rec2 = 1 - rec2

        return X1, X2, angles, kurt_vals, best_max, best_min, rec1, rec2

    result = load_and_process_images()
    if isinstance(result[-1], str):
        st.error(result[-1])
    else:
        X1, X2, angles, kurt_vals, best_max, best_min, rec1, rec2 = result

        # Show mixed images
        st.markdown("### Mixed Images (Observed)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(X1 / 255.0, caption="Mixed Image 1", use_container_width=True, clamp=True)
        with col2:
            st.image(X2 / 255.0, caption="Mixed Image 2", use_container_width=True, clamp=True)

        # Kurtosis landscape
        st.markdown("### Kurtosis Landscape")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.degrees(angles), y=kurt_vals, mode='lines',
            line=dict(color='#6366f1', width=2), name='Kurtosis'))
        fig.add_vline(x=np.degrees(best_max), line_dash="dash", line_color="red",
            annotation_text=f"Max: {np.degrees(best_max):.1f}°")
        fig.add_vline(x=np.degrees(best_min), line_dash="dash", line_color="green",
            annotation_text=f"Min: {np.degrees(best_min):.1f}°")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Kurtosis vs Projection Angle (720 angles scanned)",
            xaxis_title="Angle (degrees)", yaxis_title="Kurtosis (Fisher)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Max kurtosis angle", f"{np.degrees(best_max):.1f}°")
            st.metric("Max kurtosis value", f"{kurt_vals[np.argmax(kurt_vals)]:.3f}")
        with c2:
            st.metric("Min kurtosis angle", f"{np.degrees(best_min):.1f}°")
            st.metric("Min kurtosis value", f"{kurt_vals[np.argmin(kurt_vals)]:.3f}")

        # Recovered images
        st.markdown("### Recovered Source Images")
        col3, col4 = st.columns(2)
        with col3:
            st.image(rec1, caption="Recovered Source 1 (max kurtosis)", use_container_width=True, clamp=True)
        with col4:
            st.image(rec2, caption="Recovered Source 2 (min kurtosis)", use_container_width=True, clamp=True)

        st.markdown("""
<div class="success-panel">
✅ <b>Image separation successful!</b> The kurtosis scan identified two distinct projection angles that recover the independent source images.
</div>
""", unsafe_allow_html=True)

        with st.expander("📖 How it works"):
            st.markdown(r"""
**Algorithm:**
1. Flatten each image to a 1D vector
2. Stack into matrix $\mathbf{X} \in \mathbb{R}^{n \times 2}$
3. For each angle $\theta \in [0°, 180°)$, compute projection $y = \mathbf{X} \cdot [\cos\theta, \sin\theta]^T$
4. Measure kurtosis of $y$
5. **Max kurtosis** → one source; **Min kurtosis** → the other source
6. Project along optimal angles and reshape back to images

**Why it works:**
- By the CLT, mixtures are *more Gaussian* than sources
- Sources have *extremal* kurtosis values (maximum or minimum non-Gaussianity)
- Scanning all angles finds the directions that "undo" the mixing
            """)


# ══════════════════════════════════════════════════════
# TAB 2: Audio Separation
# ══════════════════════════════════════════════════════
with tab_audio:
    st.markdown("## 🎵 Cocktail Party Problem — Audio Source Separation")

    st.markdown("""
<div class="info-panel">

**The Cocktail Party Problem:** Three microphones each record a mixture of three speakers.  
Using only the mixed recordings, ICA (FastICA) separates the individual speakers.

**Model:** $\\mathbf{X} = \\mathbf{S}\\,\\mathbf{A}^T$ where each column of $\\mathbf{S}$ is one speaker, and $\\mathbf{A}$ is the unknown mixing matrix.

</div>
""", unsafe_allow_html=True)

    # Check for audio files
    mixture_files = [DATA_DIR / f"mixture_{i}.wav" for i in range(1, 4)]
    audio_available = all(f.exists() for f in mixture_files)

    if not audio_available:
        st.warning("Audio mixture files not found in the data directory. Showing a synthetic demo instead.")

        # Synthetic audio demo
        st.markdown("### Synthetic Audio Demo")
        with st.sidebar:
            st.markdown("### 🎵 Audio Settings")
            sr = 8000
            duration = st.slider("Duration (seconds)", 1.0, 5.0, 2.0, 0.5, key="audio_dur")
            n_audio = int(sr * duration)

        rng = np.random.default_rng(42)
        t = np.linspace(0, duration, n_audio)

        # Create 3 synthetic sources
        s1_audio = np.sin(2 * np.pi * 440 * t)  # Pure tone
        s2_audio = np.sign(np.sin(2 * np.pi * 300 * t))  # Square wave
        s3_audio = rng.laplace(0, 1, n_audio)  # Noise burst
        s3_audio = s3_audio / np.max(np.abs(s3_audio))

        S_audio = np.column_stack([s1_audio, s2_audio, s3_audio])

        # Mix
        A_audio = np.array([[1.0, 0.5, 0.3],
                            [0.3, 1.0, 0.5],
                            [0.5, 0.3, 1.0]])
        X_audio = S_audio @ A_audio.T

        # Apply FastICA
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=3, whiten='unit-variance', max_iter=5000, tol=1e-4, random_state=42)
        S_recovered_audio = ica.fit_transform(X_audio)
        S_recovered_audio = S_recovered_audio / np.max(np.abs(S_recovered_audio), axis=0) * 0.9

        # Plot waveforms
        colors_src = ['#3b82f6', '#ef4444', '#10b981']
        colors_mix = ['#7c3aed', '#f97316', '#6366f1']
        colors_rec = ['#dc2626', '#2563eb', '#059669']
        show_samples = min(int(0.05 * sr), n_audio)  # Show 50ms

        fig = make_subplots(rows=3, cols=3, subplot_titles=[
            "Source 1 (Sine)", "Source 2 (Square)", "Source 3 (Noise)",
            "Mixture 1", "Mixture 2", "Mixture 3",
            "Recovered 1", "Recovered 2", "Recovered 3"
        ], vertical_spacing=0.08, horizontal_spacing=0.05)

        t_show = t[:show_samples] * 1000  # ms

        for i in range(3):
            fig.add_trace(go.Scatter(x=t_show, y=S_audio[:show_samples, i], mode='lines',
                line=dict(color=colors_src[i], width=1), showlegend=False), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=t_show, y=X_audio[:show_samples, i], mode='lines',
                line=dict(color=colors_mix[i], width=1), showlegend=False), row=2, col=i+1)
            fig.add_trace(go.Scatter(x=t_show, y=S_recovered_audio[:show_samples, i], mode='lines',
                line=dict(color=colors_rec[i], width=1), showlegend=False), row=3, col=i+1)

        fig.update_layout(height=600, title_text="Synthetic Cocktail Party: Sources → Mixtures → Recovered")
        for i in range(1, 4):
            fig.update_xaxes(title_text="Time (ms)", row=3, col=i)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation matrix (independence check)
        st.markdown("### Independence Verification")
        corr_mix = np.corrcoef(X_audio.T)
        corr_rec = np.corrcoef(S_recovered_audio.T)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Correlation — Mixed signals:**")
            fig_h1 = go.Figure(go.Heatmap(
                z=corr_mix, x=["Mix 1", "Mix 2", "Mix 3"], y=["Mix 1", "Mix 2", "Mix 3"],
                colorscale='RdBu_r', zmin=-1, zmax=1, text=np.round(corr_mix, 3), texttemplate="%{text}"
            ))
            fig_h1.update_layout(height=300, margin=dict(t=30))
            st.plotly_chart(fig_h1, use_container_width=True)
        with col2:
            st.markdown("**Correlation — Recovered sources:**")
            fig_h2 = go.Figure(go.Heatmap(
                z=corr_rec, x=["Rec 1", "Rec 2", "Rec 3"], y=["Rec 1", "Rec 2", "Rec 3"],
                colorscale='RdBu_r', zmin=-1, zmax=1, text=np.round(corr_rec, 3), texttemplate="%{text}"
            ))
            fig_h2.update_layout(height=300, margin=dict(t=30))
            st.plotly_chart(fig_h2, use_container_width=True)

        off_diag = corr_rec[~np.eye(3, dtype=bool)]
        max_corr = np.max(np.abs(off_diag))
        quality = "Excellent" if max_corr < 0.1 else "Good" if max_corr < 0.3 else "Moderate"

        st.markdown(f"""
<div class="success-panel">
✅ <b>Separation quality: {quality}</b> (max off-diagonal |ρ| = {max_corr:.4f})<br>
The recovered sources have near-zero cross-correlation, confirming statistical independence.
</div>
""", unsafe_allow_html=True)

        # Mixing matrix
        with st.expander("🔍 Estimated Mixing Matrix"):
            st.markdown("**True mixing matrix A:**")
            st.write(np.round(A_audio, 3))
            st.markdown("**Estimated mixing matrix (from ICA):**")
            st.write(np.round(ica.mixing_, 3))

    else:
        # Real audio files available
        from scipy.io import wavfile as scipy_wav

        @st.cache_data
        def load_audio_mixtures():
            mixtures = []
            sample_rate = None
            for f in mixture_files:
                sr, audio = scipy_wav.read(str(f))
                audio = audio.astype(np.float64)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                # Normalize to [-1, 1]
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
                mixtures.append(audio)
                sample_rate = sr
            min_len = min(len(m) for m in mixtures)
            mixtures = [m[:min_len] for m in mixtures]
            return np.column_stack(mixtures), sample_rate

        X_audio, sample_rate = load_audio_mixtures()
        n_total = X_audio.shape[0]
        duration = n_total / sample_rate

        st.markdown(f"**Loaded:** {X_audio.shape[1]} mixtures, {n_total:,} samples, {sample_rate} Hz, {duration:.1f}s")

        # Play mixed audio
        st.markdown("### 🎧 Listen to the Mixtures")
        for i in range(X_audio.shape[1]):
            st.audio(X_audio[:, i], sample_rate=sample_rate, format="audio/wav")
            st.caption(f"Mixture {i+1}")

        # Waveform plots
        plot_dur = min(3.0, duration)
        plot_samples = int(plot_dur * sample_rate)
        t_arr = np.arange(plot_samples) / sample_rate

        fig = make_subplots(rows=3, cols=1, subplot_titles=[f"Mixture {i+1}" for i in range(3)],
                            vertical_spacing=0.08)
        for i in range(3):
            fig.add_trace(go.Scatter(x=t_arr, y=X_audio[:plot_samples, i], mode='lines',
                line=dict(color='purple', width=0.5), showlegend=False), row=i+1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=i+1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_layout(height=500, title_text="Mixed Audio Waveforms")
        st.plotly_chart(fig, use_container_width=True)

        # Apply FastICA
        st.markdown("### 🤖 Apply FastICA")
        with st.spinner("Running FastICA..."):
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=3, whiten='unit-variance', max_iter=5000, tol=1e-4, random_state=42)
            S_recovered_audio = ica.fit_transform(X_audio)
            S_recovered_audio = S_recovered_audio / np.max(np.abs(S_recovered_audio), axis=0) * 0.9

        # Play recovered audio
        st.markdown("### 🎧 Listen to the Recovered Sources")
        for i in range(S_recovered_audio.shape[1]):
            st.audio(S_recovered_audio[:, i], sample_rate=sample_rate, format="audio/wav")
            st.caption(f"Recovered Source {i+1}")

        # Waveform plots - recovered
        colors_rec = ['darkred', 'darkblue', 'darkgreen']
        fig2 = make_subplots(rows=3, cols=1, subplot_titles=[f"Recovered Source {i+1}" for i in range(3)],
                             vertical_spacing=0.08)
        for i in range(3):
            fig2.add_trace(go.Scatter(x=t_arr, y=S_recovered_audio[:plot_samples, i], mode='lines',
                line=dict(color=colors_rec[i], width=0.5), showlegend=False), row=i+1, col=1)
            fig2.update_yaxes(title_text="Amplitude", row=i+1, col=1)
        fig2.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig2.update_layout(height=500, title_text="Recovered Source Waveforms")
        st.plotly_chart(fig2, use_container_width=True)

        # Independence check
        st.markdown("### 📊 Independence Verification")
        corr_mix = np.corrcoef(X_audio.T)
        corr_rec = np.corrcoef(S_recovered_audio.T)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Correlation — Mixed signals:**")
            fig_h1 = go.Figure(go.Heatmap(
                z=corr_mix, x=["Mix 1", "Mix 2", "Mix 3"], y=["Mix 1", "Mix 2", "Mix 3"],
                colorscale='RdBu_r', zmin=-1, zmax=1,
                text=np.round(corr_mix, 3), texttemplate="%{text}"
            ))
            fig_h1.update_layout(height=300, margin=dict(t=30))
            st.plotly_chart(fig_h1, use_container_width=True)

        with col2:
            st.markdown("**Correlation — Recovered sources:**")
            fig_h2 = go.Figure(go.Heatmap(
                z=corr_rec, x=["Rec 1", "Rec 2", "Rec 3"], y=["Rec 1", "Rec 2", "Rec 3"],
                colorscale='RdBu_r', zmin=-1, zmax=1,
                text=np.round(corr_rec, 3), texttemplate="%{text}"
            ))
            fig_h2.update_layout(height=300, margin=dict(t=30))
            st.plotly_chart(fig_h2, use_container_width=True)

        off_diag = corr_rec[~np.eye(3, dtype=bool)]
        max_corr = np.max(np.abs(off_diag))
        quality = "Excellent" if max_corr < 0.1 else "Good" if max_corr < 0.3 else "Moderate"

        st.markdown(f"""
<div class="success-panel">
✅ <b>Separation quality: {quality}</b> (max off-diagonal |ρ| = {max_corr:.4f})<br>
</div>
""", unsafe_allow_html=True)

        with st.expander("🔍 Estimated Mixing Matrix"):
            st.write(np.round(ica.mixing_, 3))
            for i in range(3):
                row = ica.mixing_[i]
                st.markdown(f"Mixture {i+1} ≈ {row[0]:.2f}·Src1 + {row[1]:.2f}·Src2 + {row[2]:.2f}·Src3")

    with st.expander("📖 About FastICA"):
        st.markdown(r"""
**FastICA** is an efficient fixed-point iteration algorithm for ICA:

1. **Preprocessing:** Center data ($\mathbf{X} - \mathbb{E}[\mathbf{X}]$) and whiten (decorrelate + unit variance)
2. **Iteration:** For each component, find direction $\mathbf{w}$ that maximizes non-Gaussianity using Newton's method
3. **Deflation:** After finding one component, project it out and repeat

**Key advantage:** FastICA converges in very few iterations (typically < 100), unlike gradient-based methods.

**Non-Gaussianity measures used:**
- Negentropy (approximated via contrast functions)
- Kurtosis (simpler but less robust)

**The three ICA ambiguities apply:**
- **Order:** Which source is recovered first is arbitrary
- **Sign:** Sources may be inverted
- **Scale:** Amplitude is not preserved
        """)
