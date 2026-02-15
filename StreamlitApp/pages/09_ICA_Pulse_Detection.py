"""
ICA Application - Pulse Detection from Video
=============================================
Detect heart rate from facial video using Independent Component Analysis (ICA).

Based on PulseDetectionFromVideo.ipynb
"""

import streamlit as st
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt, find_peaks, spectrogram

st.set_page_config(page_title="ICA Pulse Detection", page_icon="💓", layout="wide")

# ===== STYLING =====
st.markdown("""
<style>
    .pulse-header {
        background: linear-gradient(135deg, #e74c3c 0%, #8e44ad 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pulse-header h1 { color: white; margin: 0; }
    .pulse-header p { color: #f0e0ff; margin: 0.5rem 0 0 0; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
    .metric-card h2 { color: white; margin: 0; font-size: 2.2rem; }
    .metric-card p { color: #e0d0ff; margin: 0; font-size: 0.9rem; }
    .info-panel {
        background-color: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
    .success-panel {
        background-color: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem 1.2rem;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .step-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        width: 28px; height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ===== COLORS =====
COMP_COLORS = ['#e74c3c', '#3498db', '#2ecc71']
RGB_COLORS = ['#e74c3c', '#2ecc71', '#3498db']
RGB_NAMES = ['Red', 'Green', 'Blue']


# ===== HELPER FUNCTIONS =====
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
    if low >= high:
        return data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def extract_rgb_traces(video_path, roi_top, roi_bottom, roi_left, roi_right,
                       use_face_detection=False, forehead_only=False):
    """Extract average RGB traces from video ROI.

    If *use_face_detection* is True, OpenCV's Haar cascade frontal-face
    detector is used to locate the face in each frame and the ROI is placed
    on the detected bounding box (optionally narrowed to the forehead region
    when *forehead_only* is True).  Falls back to the manual percentage-based
    ROI if no face is found in a given frame.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load Haar cascade for face detection
    face_cascade = None
    if use_face_detection:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            face_cascade = None  # fall back to manual ROI

    R_trace, G_trace, B_trace = [], [], []
    sample_frame = None
    sample_face_box = None  # (x, y, w, h) on the sample frame
    mid_frame_idx = total_frames // 2
    last_face_box = None  # smooth over missed detections

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # --- Determine ROI ---
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                # Pick the largest face
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                last_face_box = (fx, fy, fw, fh)
            # else keep last_face_box from previous frame

            if last_face_box is not None:
                fx, fy, fw, fh = last_face_box
                if forehead_only:
                    # Top 40% of the face box (forehead + upper cheeks)
                    roi = frame[fy:fy + int(fh * 0.4), fx:fx + fw]
                else:
                    # Add a small inward margin (10%) to avoid hair / jaw
                    margin_x = int(fw * 0.1)
                    margin_y = int(fh * 0.1)
                    roi = frame[
                        fy + margin_y : fy + fh - margin_y,
                        fx + margin_x : fx + fw - margin_x
                    ]
            else:
                # No face detected at all yet — fall back to manual ROI
                roi = frame[int(h * roi_top):int(h * roi_bottom),
                            int(w * roi_left):int(w * roi_right)]
        else:
            roi = frame[int(h * roi_top):int(h * roi_bottom),
                        int(w * roi_left):int(w * roi_right)]

        if roi.size == 0:
            # Safety: if ROI collapsed to zero use full frame
            roi = frame

        avg_color = np.mean(roi, axis=(0, 1))
        B_trace.append(avg_color[0])
        G_trace.append(avg_color[1])
        R_trace.append(avg_color[2])

        if frame_idx == mid_frame_idx:
            sample_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sample_face_box = last_face_box  # may be None

        frame_idx += 1

    cap.release()

    if len(R_trace) == 0:
        return None, None, None, None, None, None, None

    return (np.array(R_trace), np.array(G_trace), np.array(B_trace),
            fps, sample_frame, frame_idx, sample_face_box)


def run_ica_pipeline(R_trace, G_trace, B_trace, fps, lowcut, highcut, filter_order):
    """Run full ICA pipeline: normalize, ICA, filter."""
    X_video = np.column_stack([R_trace, G_trace, B_trace])

    # Normalize
    mean_vals = np.mean(X_video, axis=0)
    std_vals = np.std(X_video, axis=0)
    std_vals = np.where(std_vals == 0, 1e-10, std_vals)
    X_norm = (X_video - mean_vals) / std_vals

    # ICA
    ica = FastICA(n_components=3, max_iter=2000, random_state=42)
    S_video = ica.fit_transform(X_norm)

    # Bandpass filter
    S_filtered = np.apply_along_axis(
        lambda x: bandpass_filter(x, lowcut, highcut, fps, order=filter_order),
        0, S_video
    )

    return X_norm, S_video, S_filtered, ica


def estimate_heart_rate(S_filtered, fs, bpm_low=40, bpm_high=180):
    """Estimate heart rate from each ICA component."""
    results = []
    freqs = np.fft.rfftfreq(len(S_filtered), 1 / fs)
    freq_mask = (freqs * 60 >= bpm_low) & (freqs * 60 <= bpm_high)
    for i in range(3):
        fft_val = np.abs(np.fft.rfft(S_filtered[:, i]))
        peak_idx = np.argmax(fft_val[freq_mask])
        peak_freq = freqs[freq_mask][peak_idx] * 60
        peak_power = fft_val[freq_mask][peak_idx]
        results.append({'component': i, 'bpm': peak_freq, 'power': peak_power})
    return results


# ===== CAMERA UTILITIES =====
def detect_cameras(max_index: int = 8):
    """Probe camera indices 0..max_index and return a list of (index, label) for
    cameras that can be opened.  On macOS the iPhone Continuity Camera is
    typically exposed alongside the built-in FaceTime camera; the labels help
    the user pick the right one."""
    import cv2
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Read one frame to check it really works
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else ''
                cameras.append((idx, f"Camera {idx}  ({w}×{h})  {backend}".strip()))
            cap.release()
        else:
            cap.release()
    return cameras


def grab_preview_frame(camera_index: int):
    """Capture a single frame from the given camera and return it as an RGB
    numpy array (or None on failure).

    Reads several warm-up frames so the camera's auto-exposure / auto-white-balance
    have time to settle — otherwise the first frame is often very dark or black.
    """
    import cv2
    import time

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm-up: let the camera auto-expose for ~1.5 s (read & discard frames)
    warmup_start = time.time()
    frame = None
    while time.time() - warmup_start < 1.5:
        ret, frame = cap.read()
        if not ret:
            break

    # Read the actual frame we want to keep
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ===== RECORD VIDEO =====
def record_webcam_video(duration, fps_target, camera_index=0, live_placeholder=None):
    """Record video from webcam using OpenCV.

    Records in MJPG AVI format, then converts to MP4 (H.264) so that the
    browser-based Streamlit video player can handle playback.

    If *live_placeholder* is a Streamlit container (e.g. ``st.empty()``), each
    captured frame is displayed there so the user sees a live preview while
    recording.

    Returns (mp4_path, error_string).
    """
    import cv2
    import time

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, (f"Could not open camera index {camera_index}. "
                      "Check camera permissions or select a different camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm-up: read frames for ~1.5 s so auto-exposure / white-balance settle
    # (prevents the first seconds of the recording from being dark/black).
    warmup_start = time.time()
    frame = None
    while time.time() - warmup_start < 1.5:
        ret, frame = cap.read()
        if not ret:
            break
        # Show the warm-up frames in the live placeholder so user sees the camera waking up
        if live_placeholder is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_placeholder.image(
                frame_rgb,
                caption="\u23f3 Camera warming up \u2014 adjusting exposure\u2026",
                use_container_width=True,
            )

    if frame is None:
        cap.release()
        return None, "Could not read from webcam."

    h, w = frame.shape[:2]

    # Record to a temporary AVI first (MJPG is reliable across platforms)
    tmp_avi = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
    tmp_avi_path = tmp_avi.name
    tmp_avi.close()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(tmp_avi_path, fourcc, fps_target, (w, h))

    start = time.time()
    count = 0
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        out.write(frame)
        count += 1

        # Show live preview (every 2nd frame to reduce overhead)
        if live_placeholder is not None and count % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed = time.time() - start
            live_placeholder.image(
                frame_rgb,
                caption=f"🔴 Recording … {elapsed:.1f}s / {duration}s  |  {count} frames",
                use_container_width=True,
            )

    cap.release()
    out.release()

    # Clear the live preview
    if live_placeholder is not None:
        live_placeholder.empty()

    if count < 30:
        os.unlink(tmp_avi_path)
        return None, f"Only {count} frames recorded. Need at least 30."

    # Convert AVI → MP4 so browsers can play it in the Streamlit video widget
    mp4_path = _convert_to_mp4(tmp_avi_path, fps_target)
    return mp4_path, None


def _convert_to_mp4(avi_path: str, fps: float) -> str:
    """Re-encode an AVI file as MP4 (using mp4v codec) for browser playback.

    Falls back to returning the original AVI path if conversion fails.
    """
    import cv2

    mp4_path = avi_path.rsplit('.', 1)[0] + '.mp4'
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return avi_path

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    # Clean up the intermediate AVI
    try:
        os.unlink(avi_path)
    except OSError:
        pass

    return mp4_path


# ============================================================
# PAGE LAYOUT
# ============================================================

# Title
st.markdown("""
<div class="pulse-header">
    <h1>💓 ICA Pulse Detection from Video</h1>
    <p>Detect your heart rate using Independent Component Analysis on facial video</p>
</div>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")

    st.markdown("### 📹 Recording")
    recording_duration = st.slider("Recording duration (s)", 5, 30, 10)
    target_fps = st.slider("Target FPS", 15, 60, 30)

    st.markdown("### 📷 Camera Selection")
    # Detect available cameras (cached so we don't probe every rerun)
    if 'available_cameras' not in st.session_state:
        with st.spinner("Detecting cameras..."):
            st.session_state['available_cameras'] = detect_cameras()
    avail_cams = st.session_state['available_cameras']

    if avail_cams:
        cam_options = {label: idx for idx, label in avail_cams}
        selected_cam_label = st.selectbox(
            "Select camera",
            list(cam_options.keys()),
            index=0,
            help="On macOS the iPhone Continuity Camera often appears as Camera 0. "
                 "Select a different index to use your built-in laptop camera."
        )
        selected_cam_idx = cam_options[selected_cam_label]
    else:
        st.warning("No cameras detected.")
        selected_cam_idx = 0

    if st.button("🔄 Re-detect cameras", use_container_width=True):
        st.session_state.pop('available_cameras', None)
        st.rerun()

    st.markdown("### 🎯 Region of Interest")
    use_face_detection = st.toggle(
        "Auto-detect face",
        value=True,
        help="Use OpenCV Haar cascade to automatically locate the face in each frame. "
             "When enabled the manual ROI sliders below are used as a fallback only."
    )
    forehead_only = False
    if use_face_detection:
        forehead_only = st.toggle(
            "Forehead only",
            value=False,
            help="Restrict the analysis to the top 40 %% of the detected face "
                 "(forehead region). This can give a cleaner pulse signal."
        )
    st.caption("Manual ROI (used when face detection is off or as fallback):")
    roi_top = st.slider("ROI Top (%)", 0, 80, 30, help="Top boundary of face ROI")
    roi_bottom = st.slider("ROI Bottom (%)", 20, 100, 70, help="Bottom boundary of face ROI")
    roi_left = st.slider("ROI Left (%)", 0, 80, 40, help="Left boundary of face ROI")
    roi_right = st.slider("ROI Right (%)", 20, 100, 60, help="Right boundary of face ROI")

    st.markdown("### 🔧 Bandpass Filter")
    lowcut = st.slider("Low cutoff (Hz)", 0.5, 1.5, 0.7, 0.05,
                        help="0.7 Hz = 42 BPM")
    highcut = st.slider("High cutoff (Hz)", 2.0, 5.0, 4.0, 0.1,
                          help="4.0 Hz = 240 BPM")
    filter_order = st.slider("Filter order", 2, 8, 5)

    st.markdown("### 📊 Display")
    bpm_display_low = st.number_input("BPM range low", 30, 100, 40)
    bpm_display_high = st.number_input("BPM range high", 100, 250, 180)
    plot_seconds = st.slider("Time series display (s)", 3, 30, 10)

    st.markdown("---")
    st.markdown("### 📚 Theory")
    st.markdown(r"""
    **ICA Model:**
    $$\mathbf{x}(t)=\mathbf{A}\,\mathbf{s}(t)$$

    **Unmixing:**
    $$\mathbf{s}(t)=\mathbf{W}\,\mathbf{x}(t)$$

    One of the recovered sources $s_i$ is the blood-volume pulse.
    """)

# Convert ROI to fractions
roi_vals = (roi_top / 100, roi_bottom / 100, roi_left / 100, roi_right / 100)

# ===== VIDEO INPUT =====
st.markdown("## 📹 Step 1 — Provide a Facial Video")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.markdown("### Upload a video file")
    uploaded_file = st.file_uploader(
        "Upload a video of your face (AVI, MOV, MP4)",
        type=['avi', 'mov', 'mp4', 'mkv'],
        help="Record ~10 s of your face with good lighting and minimal movement."
    )

with input_col2:
    st.markdown("### Record from webcam")
    st.caption("Uses your computer's camera via OpenCV (local only).")
    record_btn = st.button("🔴 Start Recording", type="primary", use_container_width=True)

# ── Camera Preview (live snapshot) ────────────────────────────
st.markdown("---")
preview_col_cam, preview_col_info = st.columns([2, 1])

with preview_col_cam:
    st.markdown("### 📷 Camera Preview")
    if st.button("🔄 Refresh Preview", use_container_width=False):
        st.session_state['_refresh_preview'] = True

    # Always show a snapshot so the user can verify the right camera
    preview_frame = grab_preview_frame(selected_cam_idx)
    if preview_frame is not None:
        # Run face detection on the preview frame and draw the box
        if use_face_detection:
            import cv2 as _cv2_preview
            _gray_pv = _cv2_preview.cvtColor(
                _cv2_preview.cvtColor(preview_frame, _cv2_preview.COLOR_RGB2BGR),
                _cv2_preview.COLOR_BGR2GRAY
            )
            _cascade_pv = _cv2_preview.CascadeClassifier(
                _cv2_preview.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            _faces_pv = _cascade_pv.detectMultiScale(
                _gray_pv, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            annotated_frame = preview_frame.copy()
            if len(_faces_pv) > 0:
                fx, fy, fw, fh = max(_faces_pv, key=lambda r: r[2] * r[3])
                import cv2 as _drw
                if forehead_only:
                    _drw.rectangle(annotated_frame, (fx, fy),
                                   (fx + fw, fy + int(fh * 0.4)),
                                   (0, 255, 0), 2)
                    _drw.putText(annotated_frame, 'Forehead ROI',
                                 (fx, fy - 8), _drw.FONT_HERSHEY_SIMPLEX,
                                 0.6, (0, 255, 0), 2)
                else:
                    margin_x = int(fw * 0.1)
                    margin_y = int(fh * 0.1)
                    _drw.rectangle(annotated_frame,
                                   (fx + margin_x, fy + margin_y),
                                   (fx + fw - margin_x, fy + fh - margin_y),
                                   (0, 255, 0), 2)
                    _drw.putText(annotated_frame, 'Face ROI',
                                 (fx + margin_x, fy + margin_y - 8),
                                 _drw.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                st.image(annotated_frame,
                         caption=f"✅ Face detected — {selected_cam_label}",
                         use_container_width=True)
            else:
                st.image(preview_frame,
                         caption=f"⚠️ No face detected — {selected_cam_label}",
                         use_container_width=True)
                st.warning("No face detected. Ensure your face is visible and well-lit.")
        else:
            st.image(preview_frame, caption=f"Live snapshot from {selected_cam_label}",
                     use_container_width=True)
    else:
        st.warning(f"Could not grab a frame from camera index {selected_cam_idx}. "
                   "Try selecting a different camera in the sidebar.")

with preview_col_info:
    st.markdown("### ℹ️ Camera Tips")
    st.markdown("""
    - **macOS**: The iPhone Continuity Camera often appears as Camera 0.
      Your **built-in FaceTime / laptop camera** is usually Camera 1 or 2.
    - Use the **Camera Selection** dropdown in the sidebar to switch.
    - Click **Refresh Preview** to see a new snapshot.
    - During recording you will see a **live video feed** below.
    - **Auto-detect face** draws a green box around the detected face ROI.
    """)

# ── Handle recording (with live preview) ──────────────────────
if record_btn:
    st.markdown("### 🔴 Recording in progress…")
    live_frame_placeholder = st.empty()
    path, err = record_webcam_video(
        recording_duration, target_fps,
        camera_index=selected_cam_idx,
        live_placeholder=live_frame_placeholder,
    )
    if err:
        st.error(err)
    else:
        st.session_state['recorded_video_path'] = path
        st.success(f"✅ Recording saved ({recording_duration}s).")
        st.rerun()  # rerun so the video player picks up the new file

# Resolve video path
video_path = None
video_bytes = None   # for the player

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    tmp = tempfile.NamedTemporaryFile(suffix='.' + uploaded_file.name.split('.')[-1], delete=False)
    tmp.write(video_bytes)
    tmp.close()
    video_path = tmp.name
    # stash for the remove-all flow
    st.session_state['uploaded_video_path'] = video_path
elif 'recorded_video_path' in st.session_state:
    video_path = st.session_state['recorded_video_path']
    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
    else:
        # file was deleted
        del st.session_state['recorded_video_path']
        video_path = None

# ── Video Preview & Playback ──────────────────────────────────
if video_path is not None and video_bytes is not None:
    st.markdown("---")
    st.markdown("### 🎬 Video Preview")

    preview_col, info_col = st.columns([2, 1])

    with preview_col:
        st.video(video_bytes)

    with info_col:
        # Quick stats via OpenCV
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(video_path)
        _total = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        _fps_info = _cap.get(_cv2.CAP_PROP_FPS) or 0
        _w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        _h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()
        _dur = _total / _fps_info if _fps_info > 0 else 0

        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Duration | {_dur:.1f} s |
        | Frames   | {_total} |
        | FPS      | {_fps_info:.1f} |
        | Resolution | {_w} × {_h} |
        | File size | {len(video_bytes) / 1024:.0f} KB |
        """)

        # -- Remove / Clear button --
        if st.button("🗑️ Remove Video & Reset", type="secondary", use_container_width=True):
            # Clean up temp files
            for key in ('recorded_video_path', 'uploaded_video_path'):
                p = st.session_state.pop(key, None)
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            st.rerun()

if video_path is None:
    st.info("👆 Upload a video or record one to begin the analysis.")
    st.stop()

# ===== PROCESSING =====
st.markdown("---")
st.markdown("## ⚙️ Step 2 — Processing Pipeline")

with st.spinner("Extracting RGB traces from video..."):
    R_trace, G_trace, B_trace, fps, sample_frame, n_frames, detected_face_box = extract_rgb_traces(
        video_path, *roi_vals,
        use_face_detection=use_face_detection,
        forehead_only=forehead_only,
    )

if R_trace is None:
    st.error("Failed to extract frames from video. Please check the file and try again.")
    st.stop()

duration_s = n_frames / fps

# Pipeline progress
pipeline_cols = st.columns(5)
steps = [
    ("1", "Extract RGB", f"{n_frames} frames @ {fps:.0f} FPS"),
    ("2", "Normalize", "Z-score per channel"),
    ("3", "FastICA", "3 components"),
    ("4", "Bandpass", f"{lowcut}–{highcut} Hz"),
    ("5", "FFT / HR", "Peak detection"),
]
for col, (num, name, desc) in zip(pipeline_cols, steps):
    col.markdown(f"""
    <div style="text-align:center; padding:0.5rem; background:#f8f9fa; border-radius:0.5rem; border:1px solid #dee2e6;">
        <span class="step-badge">{num}</span><br>
        <b>{name}</b><br>
        <small style="color:#666;">{desc}</small>
    </div>
    """, unsafe_allow_html=True)

with st.spinner("Running ICA pipeline..."):
    X_norm, S_video, S_filtered, ica_model = run_ica_pipeline(
        R_trace, G_trace, B_trace, fps, lowcut, highcut, filter_order
    )
    hr_results = estimate_heart_rate(S_filtered, fps, bpm_display_low, bpm_display_high)

# Determine best component
best_comp = max(hr_results, key=lambda r: r['power'])
best_hr = best_comp['bpm']
best_idx = best_comp['component']

# ===== HEADLINE METRICS =====
st.markdown("---")
st.markdown("## 💓 Results")

m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"""
<div class="metric-card">
    <h2>{best_hr:.0f}</h2>
    <p>Estimated Heart Rate (BPM)</p>
</div>""", unsafe_allow_html=True)
m2.markdown(f"""
<div class="metric-card" style="background:linear-gradient(135deg,#38a169,#2d8659);">
    <h2>{best_idx}</h2>
    <p>Best ICA Component</p>
</div>""", unsafe_allow_html=True)
m3.markdown(f"""
<div class="metric-card" style="background:linear-gradient(135deg,#3182ce,#2b6cb0);">
    <h2>{n_frames}</h2>
    <p>Frames Processed</p>
</div>""", unsafe_allow_html=True)
m4.markdown(f"""
<div class="metric-card" style="background:linear-gradient(135deg,#d69e2e,#b7791f);">
    <h2>{duration_s:.1f}s</h2>
    <p>Video Duration</p>
</div>""", unsafe_allow_html=True)

# ===== ANALYSIS TABS =====
st.markdown("---")

tabs = st.tabs([
    "📊 Overview",
    "⏱️ HR Over Time",
    "🔬 ICA Components",
    "✅ Signal Quality",
    "🧮 ICA Matrices",
    "🌈 Spectrogram",
    "📈 PSD & Harmonics",
    "💗 HRV Analysis",
    "🏆 Component Score",
    "🎯 ROI Preview",
    "🌀 3D Phase Space",
])

plot_frames = min(int(plot_seconds * fps), len(X_norm))
time_axis = np.arange(plot_frames) / fps
freqs_full = np.fft.rfftfreq(len(S_filtered), 1 / fps)
freq_mask_full = (freqs_full * 60 >= bpm_display_low) & (freqs_full * 60 <= bpm_display_high)

# ──────────────────────────────────────────────
# TAB 0 — OVERVIEW
# ──────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Raw RGB Intensity Changes (Normalized)")
    st.markdown("""
    The three curves show the average R, G, B pixel intensities over the face ROI,
    after Z-score normalization. Subtle oscillations encode the blood-volume pulse.
    The **Green channel** is most sensitive due to hemoglobin absorption at ~540 nm.
    """)
    fig_rgb = go.Figure()
    for i, (name, color) in enumerate(zip(RGB_NAMES, RGB_COLORS)):
        fig_rgb.add_trace(go.Scatter(
            x=time_axis, y=X_norm[:plot_frames, i],
            name=name, line=dict(color=color, width=1.5)
        ))
    fig_rgb.update_layout(
        xaxis_title="Time (s)", yaxis_title="Normalized Intensity",
        height=350, template="plotly_white",
        legend=dict(orientation="h", y=1.12)
    )
    st.plotly_chart(fig_rgb, use_container_width=True)

    col_ica, col_fft = st.columns(2)

    with col_ica:
        st.markdown("### Independent Components (filtered)")
        fig_ic = go.Figure()
        for i in range(3):
            fig_ic.add_trace(go.Scatter(
                x=time_axis,
                y=S_filtered[:plot_frames, i] + i * 0.5,
                name=f"Component {i}",
                line=dict(color=COMP_COLORS[i], width=1.2)
            ))
        fig_ic.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude (offset)",
            height=350, template="plotly_white",
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    with col_fft:
        st.markdown("### Frequency Spectrum")
        fig_fft = go.Figure()
        for i in range(3):
            fft_val = np.abs(np.fft.rfft(S_filtered[:, i]))
            fig_fft.add_trace(go.Scatter(
                x=freqs_full[freq_mask_full] * 60,
                y=fft_val[freq_mask_full],
                name=f"Component {i}",
                line=dict(color=COMP_COLORS[i], width=2)
            ))
        # Mark best peak
        fft_best = np.abs(np.fft.rfft(S_filtered[:, best_idx]))
        peak_mask_idx = np.argmax(fft_best[freq_mask_full])
        fig_fft.add_trace(go.Scatter(
            x=[freqs_full[freq_mask_full][peak_mask_idx] * 60],
            y=[fft_best[freq_mask_full][peak_mask_idx]],
            mode='markers', marker=dict(size=14, symbol='star', color='gold',
                                         line=dict(width=2, color='black')),
            name=f"Detected HR: {best_hr:.0f} BPM"
        ))
        fig_fft.update_layout(
            xaxis_title="Heart Rate (BPM)", yaxis_title="Magnitude",
            height=350, template="plotly_white",
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    # Per-component HR
    st.markdown("#### Dominant Frequencies")
    hr_cols = st.columns(3)
    for i, r in enumerate(hr_results):
        hr_cols[i].metric(f"Component {i}", f"{r['bpm']:.1f} BPM",
                          "Best" if i == best_idx else "")

# ──────────────────────────────────────────────
# TAB 1 — HEART RATE OVER TIME
# ──────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Heart Rate Over Time — Sliding Window")
    st.markdown("""
    A 5-second sliding window (1-second hop) computes a local FFT at each step.
    The component with the strongest spectral peak is used. This reveals how HR
    fluctuates during the recording.
    """)

    window_s = st.slider("Window length (s)", 3, 10, 5, key="hr_window")
    hop_s = st.slider("Hop size (s)", 1, 5, 1, key="hr_hop")
    window_size = int(window_s * fps)
    hop_size = int(hop_s * fps)

    hr_over_time = []
    time_points = []
    for start in range(0, len(S_filtered) - window_size, hop_size):
        window = S_filtered[start:start + window_size]
        best_hr_w, max_pow = 0, 0
        for ci in range(3):
            fv = np.abs(np.fft.rfft(window[:, ci]))
            fr = np.fft.rfftfreq(len(window), 1 / fps)
            fm = (fr * 60 >= bpm_display_low) & (fr * 60 <= bpm_display_high)
            pp = np.max(fv[fm])
            if pp > max_pow:
                max_pow = pp
                best_hr_w = fr[fm][np.argmax(fv[fm])] * 60
        hr_over_time.append(best_hr_w)
        time_points.append((start + window_size / 2) / fps)

    hr_arr = np.array(hr_over_time)
    mean_hr = np.mean(hr_arr)
    std_hr = np.std(hr_arr)

    fig_hrt = go.Figure()
    fig_hrt.add_trace(go.Scatter(
        x=time_points, y=hr_over_time,
        mode='lines+markers', name='Heart Rate',
        line=dict(color='#e74c3c', width=2.5),
        marker=dict(size=8, color='#e74c3c', line=dict(width=1, color='white'))
    ))
    fig_hrt.add_hline(y=mean_hr, line_dash="dash", line_color="#333",
                       annotation_text=f"Mean: {mean_hr:.1f} BPM")
    fig_hrt.add_hrect(y0=mean_hr - std_hr, y1=mean_hr + std_hr,
                       fillcolor="rgba(231,76,60,0.12)", line_width=0)
    fig_hrt.update_layout(
        xaxis_title="Time (s)", yaxis_title="Heart Rate (BPM)",
        yaxis_range=[max(30, mean_hr - 30), mean_hr + 30],
        height=450, template="plotly_white"
    )
    st.plotly_chart(fig_hrt, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Mean HR", f"{mean_hr:.1f} BPM")
    mc2.metric("Std Dev", f"±{std_hr:.1f} BPM")
    mc3.metric("Range", f"{np.min(hr_arr):.0f}–{np.max(hr_arr):.0f} BPM")

# ──────────────────────────────────────────────
# TAB 2 — ICA COMPONENTS (TIME + FREQ)
# ──────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Individual ICA Components — Time & Frequency Domain")
    st.markdown("""
    Each component is shown in both time domain (waveform) and frequency domain (FFT).
    The component with a clear, narrow spectral peak in the HR range is the pulse signal.
    """)

    for i in range(3):
        col_t, col_f = st.columns(2)

        with col_t:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                y=S_filtered[:plot_frames, i],
                x=time_axis,
                line=dict(color=COMP_COLORS[i], width=1),
                name=f"Component {i}"
            ))
            fig_t.update_layout(
                title=f"Component {i} — Time Domain",
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                height=280, template="plotly_white", showlegend=False
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with col_f:
            fft_val = np.abs(np.fft.rfft(S_filtered[:, i]))
            bpm_axis = freqs_full[freq_mask_full] * 60
            fft_masked = fft_val[freq_mask_full]
            peak_idx = np.argmax(fft_masked)
            peak_bpm = bpm_axis[peak_idx]

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=bpm_axis, y=fft_masked,
                line=dict(color=COMP_COLORS[i], width=2),
                name=f"Component {i}"
            ))
            fig_f.add_trace(go.Scatter(
                x=[peak_bpm], y=[fft_masked[peak_idx]],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=[f"{peak_bpm:.1f} BPM"],
                textposition="top right",
                name="Peak"
            ))
            fig_f.update_layout(
                title=f"Component {i} — Frequency Domain",
                xaxis_title="Heart Rate (BPM)", yaxis_title="Magnitude",
                height=280, template="plotly_white", showlegend=False
            )
            st.plotly_chart(fig_f, use_container_width=True)

        if i < 2:
            st.markdown("---")

# ──────────────────────────────────────────────
# TAB 3 — SIGNAL QUALITY
# ──────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Signal Quality Assessment")
    st.markdown("Six diagnostic views to evaluate signal and detection quality.")

    # Row 1
    sq1, sq2, sq3 = st.columns(3)

    # 3.1 Correlation heatmap
    with sq1:
        corr = np.corrcoef(X_norm.T)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr, x=RGB_NAMES, y=RGB_NAMES,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=np.round(corr, 2), texttemplate="%{text}",
            textfont=dict(size=14)
        ))
        fig_corr.update_layout(title="RGB Channel Correlation", height=350,
                                template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    # 3.2 Before/After filtering
    with sq2:
        fig_bf = go.Figure()
        ps = min(200, len(S_video))
        fig_bf.add_trace(go.Scatter(
            y=S_video[:ps, 0], name='Before Filter',
            line=dict(color='#aaa', width=1), opacity=0.7
        ))
        fig_bf.add_trace(go.Scatter(
            y=S_filtered[:ps, 0], name='After Filter',
            line=dict(color='#e74c3c', width=2)
        ))
        fig_bf.update_layout(title="Filtering Effect (Component 0)", height=350,
                              xaxis_title="Frame", yaxis_title="Amplitude",
                              template="plotly_white")
        st.plotly_chart(fig_bf, use_container_width=True)

    # 3.3 SNR
    with sq3:
        snr_vals = []
        for i in range(3):
            fv = np.abs(np.fft.rfft(S_filtered[:, i]))
            fm = freq_mask_full
            sig_p = np.max(fv[fm])
            noise_p = np.median(fv[fm])
            snr_vals.append(20 * np.log10(sig_p / (noise_p + 1e-10)))

        fig_snr = go.Figure(data=go.Bar(
            x=[f"Comp {i}" for i in range(3)],
            y=snr_vals,
            marker_color=COMP_COLORS,
            text=[f"{v:.1f} dB" for v in snr_vals],
            textposition='outside'
        ))
        fig_snr.update_layout(title="Signal-to-Noise Ratio", height=350,
                               yaxis_title="SNR (dB)", template="plotly_white")
        st.plotly_chart(fig_snr, use_container_width=True)

    # Row 2
    sq4, sq5, sq6 = st.columns(3)

    # 3.4 PSD log scale
    with sq4:
        fig_psd = go.Figure()
        for i in range(3):
            fv = np.abs(np.fft.rfft(S_filtered[:, i]))
            psd = fv ** 2
            fig_psd.add_trace(go.Scatter(
                x=freqs_full[freq_mask_full] * 60,
                y=psd[freq_mask_full],
                name=f"Comp {i}",
                line=dict(color=COMP_COLORS[i], width=1.5)
            ))
        fig_psd.update_layout(title="Power Spectral Density", height=350,
                               xaxis_title="BPM", yaxis_title="PSD",
                               yaxis_type="log", template="plotly_white")
        st.plotly_chart(fig_psd, use_container_width=True)

    # 3.5 RGB variance over time
    with sq5:
        win = 30
        rgb_var = [np.var(np.mean(X_norm[i:i + win], axis=1))
                   for i in range(len(X_norm) - win)]
        fig_var = go.Figure(data=go.Scatter(
            y=rgb_var, line=dict(color='#3498db', width=1)
        ))
        fig_var.update_layout(title="RGB Variance Over Time", height=350,
                               xaxis_title="Frame", yaxis_title="Variance",
                               template="plotly_white")
        st.plotly_chart(fig_var, use_container_width=True)

    # 3.6 Peak confidence
    with sq6:
        ratios = []
        for i in range(3):
            fv = np.abs(np.fft.rfft(S_filtered[:, i]))
            fm = freq_mask_full
            ratios.append(np.max(fv[fm]) / (np.mean(fv[fm]) + 1e-10))

        fig_pc = go.Figure(data=go.Bar(
            x=[f"Comp {i}" for i in range(3)],
            y=ratios,
            marker_color=COMP_COLORS,
            text=[f"{v:.1f}x" for v in ratios],
            textposition='outside'
        ))
        fig_pc.update_layout(title="Frequency Peak Confidence", height=350,
                              yaxis_title="Peak / Mean Ratio",
                              template="plotly_white")
        st.plotly_chart(fig_pc, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 4 — ICA MATRICES
# ──────────────────────────────────────────────
with tabs[4]:
    st.markdown("### ICA Mixing & Unmixing Matrices")
    st.markdown(r"""
    **Mixing matrix** $\mathbf{A}$: how independent sources combine into RGB.
    **Unmixing matrix** $\mathbf{W}$: how RGB is decomposed into sources.
    The pulse component typically has a strong Green channel weight.
    """)

    mc1, mc2 = st.columns(2)
    mixing = ica_model.mixing_
    unmixing = ica_model.components_

    with mc1:
        fig_mix = go.Figure(data=go.Heatmap(
            z=mixing, x=[f"Comp {i}" for i in range(3)], y=RGB_NAMES,
            colorscale='RdBu_r', zmid=0,
            text=np.round(mixing, 3), texttemplate="%{text:.3f}",
            textfont=dict(size=13)
        ))
        fig_mix.update_layout(title="Mixing Matrix A<br>(Components → RGB)",
                               height=380, template="plotly_white")
        st.plotly_chart(fig_mix, use_container_width=True)

    with mc2:
        fig_unm = go.Figure(data=go.Heatmap(
            z=unmixing, x=RGB_NAMES, y=[f"Comp {i}" for i in range(3)],
            colorscale='RdBu_r', zmid=0,
            text=np.round(unmixing, 3), texttemplate="%{text:.3f}",
            textfont=dict(size=13)
        ))
        fig_unm.update_layout(title="Unmixing Matrix W<br>(RGB → Components)",
                               height=380, template="plotly_white")
        st.plotly_chart(fig_unm, use_container_width=True)

    st.info("**Interpretation:** A column of **A** shows how one source affects all RGB channels. "
            "A row of **W** shows how one component is extracted from RGB. "
            "The pulse component usually has a strong weight on the Green channel.")

# ──────────────────────────────────────────────
# TAB 5 — SPECTROGRAM
# ──────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Spectrogram — Time-Frequency Analysis")
    st.markdown("""
    Short-Time Fourier Transform (STFT) reveals how frequency content evolves over time.
    A bright horizontal band at a stable BPM indicates a consistent heart rate.
    """)

    for i in range(3):
        nperseg = min(256, len(S_filtered) // 4)
        noverlap = nperseg // 2
        f_sp, t_sp, Sxx = spectrogram(S_filtered[:, i], fs=fps,
                                        nperseg=nperseg, noverlap=noverlap)
        f_bpm = f_sp * 60
        fm_sp = (f_bpm >= bpm_display_low) & (f_bpm <= bpm_display_high)

        power_db = 10 * np.log10(Sxx[fm_sp, :] + 1e-10)

        fig_sg = go.Figure(data=go.Heatmap(
            z=power_db, x=t_sp, y=f_bpm[fm_sp],
            colorscale='Viridis', colorbar_title="dB"
        ))
        # Dominant frequency trace
        dom_freqs = []
        for ti in range(len(t_sp)):
            sl = Sxx[fm_sp, ti]
            dom_freqs.append(f_bpm[fm_sp][np.argmax(sl)] if np.max(sl) > 0 else np.nan)
        fig_sg.add_trace(go.Scatter(
            x=t_sp, y=dom_freqs, mode='markers',
            marker=dict(size=5, color='red'), name='Dominant freq',
            showlegend=True
        ))
        fig_sg.update_layout(
            title=f"Component {i} — Spectrogram",
            xaxis_title="Time (s)", yaxis_title="Heart Rate (BPM)",
            height=320, template="plotly_white"
        )
        st.plotly_chart(fig_sg, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 6 — PSD & HARMONICS
# ──────────────────────────────────────────────
with tabs[6]:
    st.markdown("### Power Spectral Density with Harmonics")
    st.markdown(r"""
    A real heartbeat is not a pure sinusoid — it has **harmonics** at integer multiples
    of the fundamental frequency $f_0$. Strong harmonics confirm a clean biological signal.
    """)

    for i in range(3):
        fft_val = np.abs(np.fft.rfft(S_filtered[:, i]))
        psd = fft_val ** 2 / len(S_filtered)

        peak_idx = np.argmax(fft_val[freq_mask_full])
        fundamental = freqs_full[freq_mask_full][peak_idx] * 60

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=freqs_full[freq_mask_full] * 60, y=psd[freq_mask_full],
            line=dict(color='steelblue', width=2), name='PSD'
        ))

        harm_colors = ['red', 'orange', 'purple']
        harm_labels = ['Fundamental', '2nd Harmonic', '3rd Harmonic']
        for h_n, (hc, hl) in enumerate(zip(harm_colors, harm_labels), 1):
            hf = fundamental * h_n
            if hf <= bpm_display_high:
                freq_idx = np.argmin(np.abs(freqs_full * 60 - hf))
                fig_h.add_vline(x=hf, line_dash="dash", line_color=hc, opacity=0.7)
                fig_h.add_trace(go.Scatter(
                    x=[hf], y=[psd[freq_idx]], mode='markers+text',
                    marker=dict(size=10, color=hc),
                    text=[f"{hl}<br>{hf:.1f} BPM"],
                    textposition="top right",
                    name=hl, showlegend=True
                ))

        fig_h.update_layout(
            title=f"Component {i} — PSD (fundamental: {fundamental:.1f} BPM)",
            xaxis_title="Frequency (BPM)", yaxis_title="PSD",
            height=320, template="plotly_white"
        )
        st.plotly_chart(fig_h, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 7 — HRV ANALYSIS
# ──────────────────────────────────────────────
with tabs[7]:
    st.markdown("### Heart Rate Variability (HRV) Analysis")
    st.markdown("""
    HRV quantifies the variation in time between successive heartbeats —
    an important marker of autonomic nervous system health.
    """)

    signal_hrv = S_filtered[:, best_idx]

    # Expected peak distance
    dom_freq_hz = freqs_full[freq_mask_full][
        np.argmax(np.abs(np.fft.rfft(signal_hrv))[freq_mask_full])
    ]
    expected_dist = fps / dom_freq_hz * 0.7 if dom_freq_hz > 0 else fps

    peaks_idx, _ = find_peaks(signal_hrv,
                               distance=max(int(expected_dist), 1),
                               prominence=np.std(signal_hrv) * 0.5)

    if len(peaks_idx) < 3:
        st.warning("Too few peaks detected for HRV analysis. Try adjusting filter settings or ROI.")
    else:
        ibi = np.diff(peaks_idx) / fps * 1000  # ms
        time_beats = peaks_idx[1:] / fps

        hrv1, hrv2 = st.columns(2)

        # 7.1 Signal with detected peaks
        with hrv1:
            fig_pk = go.Figure()
            fig_pk.add_trace(go.Scatter(
                x=np.arange(len(signal_hrv)) / fps,
                y=signal_hrv, line=dict(color='#3498db', width=0.8),
                name='Signal', opacity=0.7
            ))
            fig_pk.add_trace(go.Scatter(
                x=peaks_idx / fps, y=signal_hrv[peaks_idx],
                mode='markers', marker=dict(size=7, color='red'),
                name=f'Beats ({len(peaks_idx)})'
            ))
            fig_pk.update_layout(
                title=f"Component {best_idx} — Detected Beats",
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                height=350, template="plotly_white"
            )
            st.plotly_chart(fig_pk, use_container_width=True)

        # 7.2 IBI over time
        with hrv2:
            fig_ibi = go.Figure()
            fig_ibi.add_trace(go.Scatter(
                x=time_beats, y=ibi, mode='lines+markers',
                marker=dict(size=5, color='#e74c3c'),
                line=dict(color='#e74c3c', width=1.5),
                name='IBI'
            ))
            mean_ibi = np.mean(ibi)
            std_ibi = np.std(ibi)
            fig_ibi.add_hline(y=mean_ibi, line_dash="dash",
                               annotation_text=f"Mean: {mean_ibi:.0f} ms")
            fig_ibi.add_hrect(y0=mean_ibi - std_ibi, y1=mean_ibi + std_ibi,
                               fillcolor="rgba(231,76,60,0.1)", line_width=0)
            fig_ibi.update_layout(
                title="Inter-Beat Interval Over Time",
                xaxis_title="Time (s)", yaxis_title="IBI (ms)",
                height=350, template="plotly_white"
            )
            st.plotly_chart(fig_ibi, use_container_width=True)

        hrv3, hrv4 = st.columns(2)

        # 7.3 IBI histogram
        with hrv3:
            fig_hist = go.Figure(data=go.Histogram(
                x=ibi, nbinsx=20, marker_color='skyblue',
                marker_line_color='black', marker_line_width=1
            ))
            fig_hist.add_vline(x=mean_ibi, line_dash="dash", line_color="red",
                                annotation_text=f"Mean: {mean_ibi:.0f} ms")
            fig_hist.update_layout(
                title="IBI Distribution",
                xaxis_title="Inter-Beat Interval (ms)", yaxis_title="Count",
                height=350, template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # 7.4 Poincare plot
        with hrv4:
            fig_poincare = go.Figure()
            fig_poincare.add_trace(go.Scatter(
                x=ibi[:-1], y=ibi[1:], mode='markers',
                marker=dict(size=8, color='#667eea', opacity=0.6),
                name='Successive IBIs'
            ))
            mn, mx = np.min(ibi) - 50, np.max(ibi) + 50
            fig_poincare.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode='lines', line=dict(color='red', dash='dash', width=2),
                name='Identity line'
            ))
            fig_poincare.update_layout(
                title="Poincaré Plot (Beat-to-Beat)",
                xaxis_title="IBI(n) ms", yaxis_title="IBI(n+1) ms",
                height=350, template="plotly_white",
                xaxis=dict(scaleanchor="y", scaleratio=1)
            )
            st.plotly_chart(fig_poincare, use_container_width=True)

        # HRV metrics
        sdnn = np.std(ibi)
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
        mean_hr_hrv = 60000 / mean_ibi

        st.markdown("#### HRV Metrics Summary")
        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Mean HR", f"{mean_hr_hrv:.1f} BPM")
        hm2.metric("Mean IBI", f"{mean_ibi:.0f} ms")
        hm3.metric("SDNN", f"{sdnn:.1f} ms", help="Total HRV — higher is healthier")
        hm4.metric("RMSSD", f"{rmssd:.1f} ms", help="Short-term variability (vagal tone)")

# ──────────────────────────────────────────────
# TAB 8 — COMPONENT SELECTION SCORE
# ──────────────────────────────────────────────
with tabs[8]:
    st.markdown("### Automated Component Selection")
    st.markdown("""
    An automated scoring system evaluates each ICA component across **five criteria**
    to determine which is most likely the heartbeat signal.
    """)

    criteria_weights = {
        'Peak Sharpness': 0.25,
        'Frequency Range': 0.20,
        'SNR': 0.25,
        'Consistency': 0.15,
        'Spectral Purity': 0.15,
    }

    scores = {}
    for i in range(3):
        sc = {}
        fv = np.abs(np.fft.rfft(S_filtered[:, i]))
        fm = freq_mask_full

        # Peak sharpness
        pk_power = np.max(fv[fm])
        mn_power = np.mean(fv[fm])
        sc['Peak Sharpness'] = min((pk_power / (mn_power + 1e-10)) / 10, 1.0) * 100

        # Frequency range
        pk_bpm = freqs_full[fm][np.argmax(fv[fm])] * 60
        if 50 <= pk_bpm <= 90:
            sc['Frequency Range'] = 100
        elif 40 <= pk_bpm <= 120:
            sc['Frequency Range'] = 70
        else:
            sc['Frequency Range'] = 30

        # SNR
        noise_floor = np.percentile(fv[fm], 25)
        snr = 20 * np.log10(pk_power / (noise_floor + 1e-10))
        sc['SNR'] = min(snr / 30, 1.0) * 100

        # Consistency
        ws = int(3 * fps)
        hop = int(fps)
        variances = [np.var(S_filtered[s:s + ws, i])
                     for s in range(0, len(S_filtered) - ws, hop)]
        if len(variances) > 1:
            consistency = 1.0 - min(np.std(variances) / (np.mean(variances) + 1e-10), 1.0)
        else:
            consistency = 0.5
        sc['Consistency'] = consistency * 100

        # Spectral Purity
        psd_norm = fv[fm] / (np.sum(fv[fm]) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1
        sc['Spectral Purity'] = (1.0 - spectral_entropy / max_entropy) * 100

        scores[i] = sc

    total_scores = {
        i: sum(scores[i][c] * criteria_weights[c] for c in criteria_weights)
        for i in range(3)
    }
    winner = max(total_scores, key=total_scores.get)

    cs1, cs2 = st.columns(2)

    # Radar chart
    with cs1:
        cats = list(criteria_weights.keys())
        fig_radar = go.Figure()
        for i in range(3):
            vals = [scores[i][c] for c in cats] + [scores[i][cats[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats + [cats[0]],
                fill='toself', opacity=0.2,
                name=f"Component {i}",
                line=dict(color=COMP_COLORS[i])
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Multi-Criteria Comparison",
            height=420, template="plotly_white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart
    with cs2:
        colors_bar = ['gold' if i == winner else 'skyblue' for i in range(3)]
        fig_bar = go.Figure(data=go.Bar(
            x=[total_scores[i] for i in range(3)],
            y=[f"Component {i}" for i in range(3)],
            orientation='h',
            marker_color=colors_bar,
            text=[f"{total_scores[i]:.1f}" for i in range(3)],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title="Total Weighted Score",
            xaxis_title="Score (0–100)", xaxis_range=[0, 105],
            height=420, template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Stacked breakdown
    fig_stack = go.Figure()
    for criterion in cats:
        fig_stack.add_trace(go.Bar(
            x=[f"Comp {i}" for i in range(3)],
            y=[scores[i][criterion] * criteria_weights[criterion] for i in range(3)],
            name=criterion
        ))
    fig_stack.update_layout(
        barmode='stack', title="Score Breakdown by Criterion",
        yaxis_title="Weighted Score", height=380, template="plotly_white"
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Winner announcement
    winner_hr = freqs_full[freq_mask_full][
        np.argmax(np.abs(np.fft.rfft(S_filtered[:, winner]))[freq_mask_full])
    ] * 60
    st.markdown(f"""
    <div class="success-panel">
        <h3>🏆 Recommended: Component {winner}</h3>
        <p><b>Score:</b> {total_scores[winner]:.1f} / 100 &nbsp;|&nbsp;
           <b>Estimated Heart Rate:</b> {winner_hr:.1f} BPM</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed table
    with st.expander("📋 Detailed Score Report"):
        rows = []
        for i in range(3):
            for c in cats:
                rows.append({
                    'Component': f"Comp {i}",
                    'Criterion': c,
                    'Score': f"{scores[i][c]:.1f}",
                    'Weight': f"{criteria_weights[c]:.2f}",
                    'Contribution': f"{scores[i][c] * criteria_weights[c]:.1f}"
                })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# TAB 9 — ROI PREVIEW
# ──────────────────────────────────────────────
with tabs[9]:
    st.markdown("### Region of Interest (ROI) Visualization")

    if use_face_detection and detected_face_box is not None:
        fx, fy, fw, fh = detected_face_box
        mode_str = "Forehead region" if forehead_only else "Full face (with margins)"
        st.markdown(f"""
        **Mode:** Auto-detected face &nbsp;→&nbsp; {mode_str}
        **Face bounding box:** x={fx}, y={fy}, w={fw}, h={fh}
        """)
    else:
        st.markdown(f"""
        **Mode:** Manual ROI &nbsp;→&nbsp; rows **{roi_top}%–{roi_bottom}%**, columns **{roi_left}%–{roi_right}%**.
        """)

    if sample_frame is not None:
        h, w, _ = sample_frame.shape

        # Compute the actual ROI rectangle to display
        if use_face_detection and detected_face_box is not None:
            fx, fy, fw, fh = detected_face_box
            if forehead_only:
                roi_x1, roi_y1 = fx, fy
                roi_x2, roi_y2 = fx + fw, fy + int(fh * 0.4)
            else:
                margin_x = int(fw * 0.1)
                margin_y = int(fh * 0.1)
                roi_x1 = fx + margin_x
                roi_y1 = fy + margin_y
                roi_x2 = fx + fw - margin_x
                roi_y2 = fy + fh - margin_y
        else:
            roi_y1, roi_y2 = int(h * roi_vals[0]), int(h * roi_vals[1])
            roi_x1, roi_x2 = int(w * roi_vals[2]), int(w * roi_vals[3])

        # Clamp
        roi_x1 = max(roi_x1, 0)
        roi_y1 = max(roi_y1, 0)
        roi_x2 = min(roi_x2, w)
        roi_y2 = min(roi_y2, h)

        rc1, rc2 = st.columns(2)
        with rc1:
            fig_roi = px.imshow(sample_frame)
            box_color = 'lime' if (use_face_detection and detected_face_box is not None) else 'red'
            fig_roi.add_shape(type="rect",
                               x0=roi_x1, y0=roi_y1, x1=roi_x2, y1=roi_y2,
                               line=dict(color=box_color, width=3))
            title_roi = ("Full Frame — Face Detected (green box)"
                         if box_color == 'lime'
                         else "Full Frame with Manual ROI (red box)")
            fig_roi.update_layout(title=title_roi,
                                   height=400, template="plotly_white")
            st.plotly_chart(fig_roi, use_container_width=True)

        with rc2:
            roi_crop = sample_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            fig_zoom = px.imshow(roi_crop)
            fig_zoom.update_layout(title="Zoomed ROI", height=400,
                                    template="plotly_white")
            st.plotly_chart(fig_zoom, use_container_width=True)

        avg_roi = np.mean(sample_frame[roi_y1:roi_y2, roi_x1:roi_x2], axis=(0, 1))
        st.info(f"**ROI Size:** {roi_x2 - roi_x1} × {roi_y2 - roi_y1} px &nbsp;|&nbsp; "
                f"**Avg RGB:** ({avg_roi[0]:.0f}, {avg_roi[1]:.0f}, {avg_roi[2]:.0f})")
    else:
        st.warning("Sample frame not available from video.")

# ──────────────────────────────────────────────
# TAB 10 — 3D PHASE SPACE
# ──────────────────────────────────────────────
with tabs[10]:
    st.markdown("### 3D Phase Space Trajectory")
    st.markdown("""
    Each point is one time sample in the 3-channel space, coloured by frame number.
    Periodic (circular/helical) patterns indicate a rhythmic signal like the heartbeat.
    """)

    subset = min(500, len(X_norm))
    step = max(1, subset // 250)
    idx = np.arange(0, subset, step)

    ps1, ps2, ps3 = st.columns(3)

    with ps1:
        fig_3d1 = go.Figure(data=go.Scatter3d(
            x=X_norm[idx, 0], y=X_norm[idx, 1], z=X_norm[idx, 2],
            mode='markers+lines',
            marker=dict(size=3, color=idx, colorscale='Viridis',
                        colorbar=dict(title="Frame", x=1.0)),
            line=dict(color='gray', width=1),
            name='RGB'
        ))
        fig_3d1.update_layout(
            title="RGB Phase Space",
            scene=dict(xaxis_title='Red', yaxis_title='Green', zaxis_title='Blue'),
            height=450, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_3d1, use_container_width=True)

    with ps2:
        fig_3d2 = go.Figure(data=go.Scatter3d(
            x=S_video[idx, 0], y=S_video[idx, 1], z=S_video[idx, 2],
            mode='markers+lines',
            marker=dict(size=3, color=idx, colorscale='Plasma',
                        colorbar=dict(title="Frame", x=1.0)),
            line=dict(color='gray', width=1),
            name='ICA'
        ))
        fig_3d2.update_layout(
            title="ICA Components",
            scene=dict(xaxis_title='Comp 0', yaxis_title='Comp 1', zaxis_title='Comp 2'),
            height=450, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_3d2, use_container_width=True)

    with ps3:
        fig_3d3 = go.Figure(data=go.Scatter3d(
            x=S_filtered[idx, 0], y=S_filtered[idx, 1], z=S_filtered[idx, 2],
            mode='markers+lines',
            marker=dict(size=3, color=idx, colorscale='RdBu',
                        colorbar=dict(title="Frame", x=1.0)),
            line=dict(color='gray', width=1),
            name='Filtered'
        ))
        fig_3d3.update_layout(
            title="Filtered Components",
            scene=dict(xaxis_title='Comp 0', yaxis_title='Comp 1', zaxis_title='Comp 2'),
            height=450, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_3d3, use_container_width=True)

    st.info("**Tip:** Click and drag to rotate the 3D plots. Scroll to zoom. "
            "Circular patterns in the filtered space indicate a periodic pulse signal.")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.9rem;'>
    <p>ICA Pulse Detection | Based on the PulseDetectionFromVideo notebook</p>
    <p>TTK4260-2026 — Independent Component Analysis</p>
</div>
""", unsafe_allow_html=True)
