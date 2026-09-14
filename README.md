# 🛣️ RoadVision

### Real-Time Pothole Detection, Severity Estimation & Road Risk Analysis

RoadVision is a computer vision application that detects potholes in real time using **YOLOv8n**, estimates pothole severity, and calculates a frame-level road risk score. The system is designed for practical road-condition monitoring and supports image upload, snapshot analysis, and live webcam detection through a responsive Streamlit interface.

## 🚀 Live Demo

👉 **[Try RoadVision](https://roadvision.streamlit.app/)**

## ✨ Features

- 🔍 Real-time pothole detection with YOLOv8n
- 📏 Pothole severity estimation: **Small, Medium, Large**

- ⚠️ Frame-level **Road Risk Score** based on detected pothole severity and confidence

- 📷 Image upload and snapshot analysis

- 🎥 Real-time webcam detection

- 📱 Desktop and mobile-friendly Streamlit interface

- ⚡ Interactive CPU inference

- 📊 Detection confidence and severity visualization

- ☁️ Streamlit deployment suitable for headless cloud environments

## 🧠 Model & Dataset

The RoadVision detector was trained on a custom pothole dataset containing:

| Property | Value |
|---|---:|
| Annotated images | **665** |
| Pothole instances | **1,739** |
| Detection model | **YOLOv8n** |
| Annotation format | **YOLO** |

The project includes a custom **VOC → YOLO annotation conversion pipeline**. Severity categories are adapted from the size-based thresholds provided by the dataset author.

> **Note:** Road risk scoring is an application-level heuristic based on pothole severity and model confidence; it should not be interpreted as a certified engineering road-safety measurement.

## 📐 Road Risk Score

For each detected pothole, RoadVision combines severity and detection confidence to estimate relative risk at the frame level.

Conceptually:

```text
Risk ∝ Severity Weight × Detection Confidence
```

Multiple detections can contribute to the overall frame risk. This allows the application to provide a simple prioritization signal for potentially hazardous road conditions.

## 🏗️ System Workflow

```text
                Input
                  │
        ┌─────────┴─────────┐
        │                   │
   Image Upload        Live Webcam
        │                   │
        └─────────┬─────────┘
                  ↓
             YOLOv8n
                  ↓
        Pothole Detection
                  ↓
       ┌──────────┴──────────┐
       ↓                     ↓
   Confidence          Size / Severity
       │                     │
       └──────────┬──────────┘
                  ↓
          Road Risk Score
                  ↓
        Visualization / Output
```

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 / Ultralytics**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Pillow**
- **Streamlit**
- **streamlit-webrtc**
- **PyAV**

## 📂 Project Structure

```text
Road-Vision/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

Model weights and additional assets may be stored separately depending on the deployment configuration.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tauhid-Topu-007/Road-Vision.git
cd Road-Vision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address shown in the terminal.

## 🎥 Application Modes

### 1. Upload

Upload an image and run pothole detection with severity and risk analysis.

### 2. Snapshot

Capture a frame from a supported camera input and analyze it.

### 3. Real-Time Webcam

Use live webcam streaming for continuous pothole detection and visualization.

## 🌐 Deployment

RoadVision is deployed using **Streamlit Community Cloud** and is designed to work in a headless environment.

The deployment required handling practical issues such as:

- OpenCV GUI limitations in cloud environments
- WebRTC peer connections for browser-based video
- Real-time frame processing
- Mobile browser camera permissions
- Responsive UI behavior across desktop and mobile devices

## 📊 Dataset Preparation

The training workflow includes conversion of Pascal VOC-style annotations into YOLO-compatible labels. The converted annotations are then used to train the YOLOv8n pothole detector.

The dataset contains **665 annotated images** and **1,739 pothole instances**.

## 🔮 Future Improvements

Potential extensions include:

- GPS-based pothole localization
- Road-condition mapping and heatmaps
- Temporal tracking of potholes across video frames
- Improved severity estimation using depth or geometric information
- Edge deployment on Raspberry Pi / Jetson devices
- Road-maintenance prioritization dashboards
- Larger and more geographically diverse training datasets
- Model optimization with quantization or TensorRT

## ⚠️ Limitations

- Severity estimation is based on image-derived size thresholds rather than direct physical measurement.
- Risk scores are relative application-level indicators, not official road-safety assessments.
- Detection performance depends on lighting, camera angle, image quality, and dataset coverage.
- CPU real-time performance may vary depending on hardware and input resolution.
- Mobile webcam functionality depends on browser permissions and device support.

## 👨‍💻 Author

**Tauhidul Islam Topu**  
Computer Science & Engineering Student  
Hajee Mohammad Danesh Science & Technology University (HSTU)

- GitHub: [Tauhid-Topu-007](https://github.com/Tauhid-Topu-007)

## 📄 License

This project does not currently specify a separate open-source license. If you plan to distribute or reuse the code publicly, consider adding an appropriate `LICENSE` file.

---

⭐ If you find RoadVision useful, consider starring the repository and sharing feedback or suggestions.
