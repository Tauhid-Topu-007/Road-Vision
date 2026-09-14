import streamlit as st
import cv2
import numpy as np
import av
import time
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# streamlit-webrtc is optional
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="RoadVision · Pothole Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="auto",   # collapsed on mobile, expanded on desktop
)

# ============================================================
# Mobile-First Responsive CSS — Premium Dark + Champagne Gold Theme
# ============================================================
st.markdown("""
<style>
    /* ---------- Base / Root Variables ---------- */
    :root {
        --accent:        #d4af6a;   /* champagne gold */
        --accent-2:      #f0d9a8;   /* light gold highlight */
        --accent-dark:   #8a6a2f;   /* deep bronze */
        --bg-top:        #08090b;
        --bg-bottom:     #101216;
        --card-bg:       #14161b;
        --card-bg-2:     #1b1e25;
        --border:        #262a33;
        --border-hi:     #3a3f4c;
        --text:          #f2eee6;
        --text-dim:      #a3a7b0;
        --text-muted:    #6f7480;
        --safe:          #6fcf97;
        --warn:          #e8b654;
        --danger:        #e0645f;
        --radius:        16px;
    }

    /* ---------- Global ---------- */
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse 1200px 600px at 20% -10%, rgba(212,175,106,0.06), transparent 60%),
            radial-gradient(ellipse 900px 500px at 100% 0%, rgba(212,175,106,0.04), transparent 55%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }
    .main .block-container {
        padding: clamp(0.75rem, 4vw, 2rem) clamp(0.75rem, 4vw, 2.25rem);
        max-width: 1300px;
    }

    /* ---------- Hero Header ---------- */
    .hero {
        position: relative;
        background: linear-gradient(135deg, #1a1c22 0%, #101216 100%);
        border: 1px solid var(--border-hi);
        border-radius: var(--radius);
        padding: clamp(1.25rem, 4vw, 2.25rem) clamp(1.25rem, 5vw, 2.5rem);
        margin-bottom: clamp(0.75rem, 3vw, 1.75rem);
        box-shadow: 0 20px 50px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), var(--accent), transparent);
        opacity: 0.9;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60%; right: -10%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(212,175,106,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        color: var(--text);
        font-size: clamp(1.35rem, 5vw, 2.15rem);
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        line-height: 1.15;
        background: linear-gradient(90deg, var(--text) 30%, var(--accent-2) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        color: var(--text-dim);
        font-size: clamp(0.82rem, 2.5vw, 1.02rem);
        margin: 0;
        line-height: 1.5;
        max-width: 520px;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(212, 175, 106, 0.1);
        border: 1px solid rgba(212, 175, 106, 0.35);
        color: var(--accent-2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: clamp(0.65rem, 2vw, 0.75rem);
        font-weight: 500;
        letter-spacing: 0.3px;
        margin-right: 8px;
        margin-top: 12px;
        backdrop-filter: blur(8px);
    }

    /* ---------- Section Titles ---------- */
    .section-title {
        font-size: clamp(0.95rem, 3vw, 1.15rem);
        font-weight: 600;
        color: var(--text);
        margin: clamp(0.75rem, 3vw, 1.4rem) 0 0.7rem 0;
        padding-left: 12px;
        border-left: 3px solid var(--accent);
        letter-spacing: 0.2px;
    }

    /* ---------- Metric Grid (Responsive) ---------- */
    /* auto-fit so the grid adapts to whatever width the Analysis column
       actually gets (narrow side-column on landscape/tablet included),
       instead of forcing a fixed column count that can cramp or clip. */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
        gap: clamp(0.5rem, 2.5vw, 1rem);
        margin: 0.75rem 0 clamp(1rem, 3vw, 1.5rem) 0;
    }
    .metric-card {
        position: relative;
        background: linear-gradient(150deg, var(--card-bg) 0%, var(--card-bg-2) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(0.75rem, 2.5vw, 1.25rem) clamp(0.85rem, 3vw, 1.4rem);
        transition: all 0.25s ease;
        min-width: 0;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .metric-card:hover {
        border-color: var(--accent);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(212, 175, 106, 0.14);
    }
    .metric-card .label {
        font-size: clamp(0.6rem, 1.8vw, 0.76rem);
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 6px;
        white-space: normal;
        overflow-wrap: break-word;
        line-height: 1.25;
    }
    .metric-card .value {
        font-size: clamp(1.05rem, 4vw, 1.85rem);
        font-weight: 700;
        color: var(--text);
        line-height: 1.15;
        overflow-wrap: break-word;
        letter-spacing: -0.3px;
    }
    .metric-card .value.accent { color: var(--accent-2); }
    .metric-card .value.warn   { color: var(--warn); }
    .metric-card .value.danger { color: var(--danger); }
    .metric-card .value.safe   { color: var(--safe); }
    .metric-card .sub {
        font-size: clamp(0.58rem, 1.6vw, 0.72rem);
        color: var(--text-muted);
        margin-top: 4px;
        white-space: normal;
        overflow-wrap: break-word;
        line-height: 1.2;
    }

    /* ---------- Severity Badges ---------- */
    .badge-small, .badge-medium, .badge-large {
        padding: 4px 11px;
        border-radius: 7px;
        font-size: clamp(0.65rem, 2vw, 0.75rem);
        font-weight: 600;
        white-space: nowrap;
        border: 1px solid transparent;
    }
    .badge-small  { background: rgba(111, 207, 151, 0.12); color: var(--safe);   border-color: rgba(111, 207, 151, 0.3); }
    .badge-medium { background: rgba(232, 182, 84, 0.12);  color: var(--warn);   border-color: rgba(232, 182, 84, 0.3); }
    .badge-large  { background: rgba(224, 100, 95, 0.12);  color: var(--danger); border-color: rgba(224, 100, 95, 0.3); }

    /* ---------- Detection Rows ---------- */
    .det-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        padding: clamp(0.55rem, 2vw, 0.75rem) clamp(0.7rem, 3vw, 1rem);
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 11px;
        margin-bottom: 9px;
        gap: 8px;
        min-width: 0;
        width: 100%;
        box-sizing: border-box;
        transition: border-color 0.2s ease;
    }
    .det-row:hover {
        border-color: var(--border-hi);
    }
    .det-row .left {
        display: flex;
        align-items: center;
        gap: clamp(6px, 2vw, 12px);
        min-width: 0;
        flex: 1;
    }
    .det-row .idx {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: #10120f;
        width: clamp(20px, 6vw, 25px);
        height: clamp(20px, 6vw, 25px);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: clamp(0.65rem, 2vw, 0.72rem);
        font-weight: 700;
        flex-shrink: 0;
    }
    .det-row .conf {
        color: var(--text-dim);
        font-size: clamp(0.7rem, 2.2vw, 0.85rem);
        white-space: nowrap;
        flex-shrink: 0;
        font-variant-numeric: tabular-nums;
    }

    /* ---------- Empty State ---------- */
    .empty-state {
        background: var(--card-bg);
        border: 1px dashed var(--border-hi);
        border-radius: var(--radius);
        padding: clamp(1.6rem, 6vw, 2.75rem) clamp(1rem, 4vw, 1.5rem);
        text-align: center;
        color: var(--text-dim);
    }
    .empty-state .icon {
        font-size: clamp(1.8rem, 8vw, 2.5rem);
        margin-bottom: 10px;
        opacity: 0.85;
    }
    .empty-state .text {
        font-size: clamp(0.8rem, 2.5vw, 0.95rem);
        line-height: 1.5;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0a0b0e;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--accent-2);
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border);
    }

    /* ---------- Tabs (mobile-friendly) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--card-bg);
        padding: 5px;
        border-radius: 13px;
        border: 1px solid var(--border);
        overflow-x: auto;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
        height: clamp(36px, 10vw, 42px);
        border-radius: 9px;
        color: var(--text-dim);
        font-weight: 500;
        padding: 0 clamp(8px, 3vw, 18px);
        font-size: clamp(0.78rem, 2.4vw, 0.92rem);
        white-space: nowrap;
        flex-shrink: 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
        color: #10120f !important;
        font-weight: 600;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: #10120f;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 14px rgba(212, 175, 106, 0.2);
    }
    @media (min-width: 700px) {
        .stButton > button { width: auto; }
    }
    .stButton > button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(212, 175, 106, 0.3);
    }

    /* ---------- Images: cap width, auto height ---------- */
    .stImage img {
        max-width: 100%;
        height: auto;
        border-radius: 13px;
        border: 1px solid var(--border);
    }

    /* ---------- File Uploader / Camera Input ---------- */
    [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
        background: var(--card-bg) !important;
        border: 1px dashed var(--border-hi) !important;
        border-radius: var(--radius) !important;
    }

    /* ---------- Slider ---------- */
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stTickBar"] { display: none; }

    /* ---------- Alerts ---------- */
    .stAlert {
        border-radius: 12px;
        font-size: clamp(0.78rem, 2.4vw, 0.9rem);
        border: 1px solid var(--border);
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: clamp(0.7rem, 2vw, 0.8rem);
        padding: clamp(1rem, 4vw, 1.5rem) 0 8px 0;
        border-top: 1px solid var(--border);
        margin-top: clamp(1.5rem, 5vw, 2.25rem);
        line-height: 1.6;
        letter-spacing: 0.2px;
    }

    /* ---------- Column gap on mobile / landscape ---------- */
    /* Stack the image + analysis columns earlier (1100px) so landscape
       phones and small tablets don't squeeze the Analysis panel into an
       unreadable sliver before the columns actually wrap. */
    @media (max-width: 1100px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.9rem !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best.pt"
    return YOLO(str(model_path))

model = load_model()

# ============================================================
# Constants
# ============================================================
SEVERITY_COLOR  = {"Small": (111, 207, 151), "Medium": (232, 182, 84), "Large": (224, 100, 95)}
SEVERITY_WEIGHT = {"Small": 1, "Medium": 3, "Large": 7}
SEVERITY_BADGE  = {
    "Small":  '<span class="badge-small">● Small</span>',
    "Medium": '<span class="badge-medium">● Medium</span>',
    "Large":  '<span class="badge-large">● Large</span>',
}

# ============================================================
# Helper Functions
# ============================================================
def severity_from_area(area, w, h, ref=300):
    scale = (ref * ref) / (w * h)
    scaled = area * scale
    if scaled <= 1024:  return "Small"
    if scaled <= 9216:  return "Medium"
    return "Large"

def road_risk_score(detections):
    if not detections:
        return 0.0, "Safe"
    score = sum(SEVERITY_WEIGHT[d["severity"]] * d["confidence"] for d in detections)
    if score < 3:    level = "Low"
    elif score < 8:  level = "Moderate"
    elif score < 15: level = "High"
    else:            level = "Critical"
    return round(score, 2), level

def detect(image_rgb, conf_threshold=0.25):
    h, w = image_rgb.shape[:2]
    result = model.predict(image_rgb, conf=conf_threshold, verbose=False)[0]

    annotated = image_rgb.copy()
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        area = (x2 - x1) * (y2 - y1)
        severity = severity_from_area(area, w, h)
        color = SEVERITY_COLOR[severity]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{severity} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            annotated,
            (x1, max(y1 - text_h - 8, 0)),
            (x1 + text_w + 6, y1),
            color, -1,
        )
        cv2.putText(
            annotated, label,
            (x1 + 3, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 18, 15), 2,
        )

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "severity": severity,
            "area": area,
        })

    score, level = road_risk_score(detections)
    return annotated, detections, score, level

# ============================================================
# UI Building Blocks
# ============================================================
def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🛣️ RoadVision</h1>
        <p>Real-time pothole detection with severity and risk scoring.</p>
        <span class="badge">YOLOv8n</span>
        <span class="badge">665 images</span>
        <span class="badge">Real-time</span>
    </div>
    """, unsafe_allow_html=True)

def risk_class(level):
    return {
        "Safe": "safe", "Low": "safe",
        "Moderate": "warn",
        "High": "danger", "Critical": "danger",
    }.get(level, "")

def render_metrics(detections, score, level, inference_ms=None):
    inf_str = f"{inference_ms:.0f} ms" if inference_ms is not None else "—"
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">Potholes</div>
            <div class="value accent">{len(detections)}</div>
            <div class="sub">detected</div>
        </div>
        <div class="metric-card">
            <div class="label">Risk Score</div>
            <div class="value {risk_class(level)}">{score}</div>
            <div class="sub">weighted</div>
        </div>
        <div class="metric-card">
            <div class="label">Risk Level</div>
            <div class="value {risk_class(level)}">{level}</div>
            <div class="sub">condition</div>
        </div>
        <div class="metric-card">
            <div class="label">Inference</div>
            <div class="value">{inf_str}</div>
            <div class="sub">latency</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_detections(detections):
    if not detections:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">✅</div>
            <div class="text">No potholes detected.<br>The road looks clean.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="section-title">Detection Breakdown</div>', unsafe_allow_html=True)
    for i, d in enumerate(detections, 1):
        st.markdown(f"""
        <div class="det-row">
            <div class="left">
                <div class="idx">{i}</div>
                {SEVERITY_BADGE[d['severity']]}
            </div>
            <div class="conf">conf&nbsp;{d['confidence']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

def render_result(image_rgb, conf_threshold):
    with st.spinner("Analyzing image..."):
        t0 = time.time()
        annotated, dets, score, level = detect(image_rgb, conf_threshold)
        inference_ms = (time.time() - t0) * 1000

    # Wide desktop: side-by-side. Narrower / landscape: stacked (CSS handles the breakpoint).
    col_img, col_side = st.columns([1.4, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-title">Annotated Output</div>', unsafe_allow_html=True)
        st.image(annotated, use_container_width=True)

    with col_side:
        st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
        render_metrics(dets, score, level, inference_ms)
        render_detections(dets)

# ============================================================
# WebRTC Processor
# ============================================================
if WEBRTC_AVAILABLE:
    class PotholeProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf_threshold = 0.25
            self.last_detections = []
            self.frame_count = 0

        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % 2 == 0:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                annotated, dets, score, level = detect(img_rgb, self.conf_threshold)
                self.last_detections = dets
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            else:
                annotated_bgr = img_bgr

            if self.last_detections:
                score, level = road_risk_score(self.last_detections)
                overlay = f"Risk: {score} ({level}) | {len(self.last_detections)} potholes"
                cv2.rectangle(annotated_bgr, (10, 10), (520, 48), (15, 18, 22), -1)
                cv2.rectangle(annotated_bgr, (10, 10), (520, 48), (106, 175, 212), 2)
                cv2.putText(
                    annotated_bgr, overlay, (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (168, 217, 240), 2,
                )

            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

# ============================================================
# Main Layout
# ============================================================
render_hero()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf = st.slider(
        "Confidence threshold",
        0.05, 0.95, 0.25, 0.05,
        help="Higher = fewer false positives, lower = more detections",
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Architecture**: YOLOv8n
    - **Training images**: 665
    - **Classes**: 1 (pothole)
    - **Size categories**: Small / Medium / Large
    """)

    st.markdown("---")
    st.markdown("### 📏 Severity Scale")
    st.markdown("""
    - 🟢 **Small** — area ≤ 1024 px
    - 🟠 **Medium** — area ≤ 9216 px
    - 🔴 **Large** — area > 9216 px
    """)

    if not WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc not installed — Real-time unavailable.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📁  Upload",
    "📷  Camera",
    "🎥  Live",
])

# Tab 1: Upload
with tab1:
    st.markdown('<div class="section-title">Upload a road image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        image = np.array(Image.open(uploaded).convert("RGB"))
        render_result(image, conf)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📤</div>
            <div class="text">Drop an image above<br>to start detection.</div>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Camera
with tab2:
    st.markdown('<div class="section-title">Capture from your device</div>', unsafe_allow_html=True)
    st.caption("Works on phones too — the browser will request camera permission.")
    cam_photo = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam_photo is not None:
        image = np.array(Image.open(cam_photo).convert("RGB"))
        render_result(image, conf)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📸</div>
            <div class="text">Allow camera access,<br>then take a photo to analyze.</div>
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Real-time
with tab3:
    st.markdown('<div class="section-title">Live detection</div>', unsafe_allow_html=True)

    if not WEBRTC_AVAILABLE:
        st.error(
            "Real-time detection requires `streamlit-webrtc` and `av`. "
            "Update `requirements.txt` and redeploy."
        )
    else:
        st.caption("Live camera feed with real-time pothole detection and risk overlay.")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key="pothole-realtime",
            video_processor_factory=PotholeProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {"facingMode": "environment"},
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.conf_threshold = conf

        st.info(
            "If the stream fails to connect on Streamlit Cloud, "
            "add a TURN server to the RTC configuration."
        )

# --- Footer ---
st.markdown("""
<div class="footer">
    RoadVision · Pothole Detection Module<br>
    Built with YOLOv8 + Streamlit
</div>
""", unsafe_allow_html=True)