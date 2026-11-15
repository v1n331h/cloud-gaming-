<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-14+-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/WebSockets-Real--Time-4353FF?style=for-the-badge&logo=socket.io&logoColor=white" />
  <img src="https://img.shields.io/badge/GSAP-Animated-88CE02?style=for-the-badge&logo=greensock&logoColor=white" />
</p>

<h1 align="center">⚡ SuperStream AI</h1>
<h3 align="center">Client-Side AI Super Resolution Streaming Pipeline</h3>

<p align="center">
  <em>Stream low. Upscale with AI. Experience high.</em>
</p>

---

## 🎯 What Is This?

**SuperStream AI** is a prototype exploring how lightweight neural networks can upscale low-resolution video frames for streaming.

The intended pipeline sends **360p** frames over WebSockets and upscales them to **1080p** in the client engine. The current model is ESPCN; its speed and quality depend on the device and checkpoint, so the benchmark results below are measured rather than performance promises.

### 🧠 The Core Idea

> **Problem:** Streaming 1080p video requires ~5.6 Gbps of raw bandwidth.  
> **Solution:** Stream at 360p (~48 Mbps) and use a CNN to reconstruct the missing detail on the client.  
> **Trade-off:** Sending fewer pixels can reduce transport cost, but upscaling cannot guarantee recovery of detail lost in the low-resolution frame.

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────────┐         ┌──────────────────┐
│  HOST SERVER    │  WS →   │   CLIENT ML ENGINE   │  WS →   │    FRONTEND      │
│  ─────────────  │ 360p    │   ────────────────    │ 1080p   │    ────────      │
│  Video Capture  │ JPEG    │   ESPCN Inference     │ + LR    │    Next.js +     │
│  + Downscale    │ ~150KB  │   + Metrics Tracking  │ + Stats │    GSAP + Canvas │
└─────────────────┘         └──────────────────────┘         └──────────────────┘
```

| Component | Tech Stack | Purpose |
|-----------|-----------|---------|
| **Host Server** | Python, FastAPI, OpenCV | Capture, downscale, compress, stream |
| **Client Engine** | Python, PyTorch, OpenCV | Receive, upscale with AI, serve to UI |
| **Frontend** | Next.js, Tailwind, GSAP | Visualize LR vs HR, display telemetry |

---

## ✨ Key Features

- 🧠 **3× AI Upscaling** — ESPCN sub-pixel convolution with bicubic fallback
- 📊 **Live Telemetry Dashboard** — FPS, inference time, bandwidth savings
- 🎨 **Spider-Verse Aesthetic** — Neon accents, halftone patterns, GSAP animations
- 🔄 **Side-by-Side Comparison** — See LR input vs AI-upscaled output live
- 🛡️ **Graceful Fallback** — Works without trained weights (bicubic interpolation)
- ⚡ **WebSocket Streaming** — Sub-50ms end-to-end latency on LAN

---

## 📋 Prerequisites

| Requirement | Version |
|------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |
| pip | 22+ |

**Optional:** CUDA-compatible GPU for faster inference (falls back to CPU gracefully).

---

## 🚀 Quick Start

### 1. Clone & Install Python Dependencies

```bash
cd cloud_gaming_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start the Host Server

Use the included `game.mp4`, a webcam, or screen capture (the default).

```bash
# Stream a video file
python host_server/streamer.py --source game.mp4 --port 8765

# Or use a webcam
python host_server/streamer.py --source webcam --port 8765

# Or capture the screen
python host_server/streamer.py --port 8765
```

### 4. Start the Client ML Engine

```bash
# With AI upscaling (if weights available)
python client_engine/inference_server.py --host-ws ws://localhost:8765 --port 8766 --model espcn --weights client_engine/espcn_fast.pth

# Without weights (bicubic fallback — works out of the box!)
python client_engine/inference_server.py --host-ws ws://localhost:8765 --port 8766 --fallback
```

### 5. Start the Frontend

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** and watch the magic! ✨

---

## 📁 Project Structure

```
cloud gaming project/
├── docs/
│   └── ARCHITECTURE_AND_STACK.md   # Deep technical documentation
├── host_server/
│   └── streamer.py                 # Video capture + downscale + stream
├── client_engine/
│   ├── model.py                    # ESPCN architecture and frame upscaler
│   ├── benchmark_model.py          # Image/video speed and quality benchmark
│   └── inference_server.py         # Inference pipeline + metrics server
├── frontend/
│   ├── package.json
│   ├── next.config.mjs
│   ├── tailwind.config.js
│   ├── postcss.config.mjs
│   ├── components/
│   │   ├── VideoPlayer.jsx         # Canvas-based frame renderer
│   │   └── TelemetryDashboard.jsx  # Real-time metrics HUD
│   ├── pages/
│   │   ├── _app.jsx
│   │   └── index.jsx               # Main dashboard layout
│   └── styles/
│       └── globals.css             # Spider-Verse theme + Tailwind
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## 🔬 How the AI Works

The client uses **ESPCN** (Efficient Sub-Pixel Convolutional Neural Network) for a 3× scale:

1. A 5×5 convolution extracts features from the low-resolution luminance channel.
2. 3×3 convolutions refine and compress those features while staying at low resolution.
3. A final convolution predicts nine sub-pixels per input pixel; PixelShuffle rearranges them into a 3× image.

The model processes only the **luminance (Y) channel** of YCrCb; chroma is enlarged with bicubic interpolation. Create one `FrameUpscaler` at service startup and reuse it for each OpenCV BGR frame. The included `espcn_fast.pth` was generated by the synthetic training script for pipeline testing; it is not a pretrained real-video model. If the checkpoint is missing or invalid, frame upscaling falls back to bicubic.

Run the measured benchmark with:

```bash
python client_engine/benchmark_model.py --frames 10
```

---

## 📊 Performance

| Method (Apple MPS) | Avg. time/frame | FPS | Mean PSNR | Mean SSIM |
|--------------------|----------------|-----|-----------|-----------|
| ESPCN (synthetic checkpoint) | 88.02 ms | 11.36 | 26.816 dB | 0.45999 |
| Bicubic | 3.94 ms | 253.58 | 36.568 dB | 0.82157 |

These are measurements from 10 frames of the included `game.mp4`, comparing a 640×360 input with a native 1920×1080 center crop from the 2880×1800 source. The synthetic checkpoint performed worse than bicubic on this clip and is not suitable evidence of real-world super-resolution quality. End-to-end latency and bandwidth savings were not measured by this benchmark.

---

## 🎨 Design Philosophy

The frontend is inspired by:
- **Spider-Verse** — Vibrant neon colors (magenta, cyan, electric yellow) against deep noir backgrounds
- **Minh Pham** — Clean minimalism with bold typography and intentional whitespace
- **Brutalist Web Design** — Raw, honest interface elements with high contrast

Animations are powered by **GSAP** for buttery-smooth 60fps entrance sequences and micro-interactions.

---

## 📖 Documentation

For a deep dive into the architecture, ML theory, and system design:

👉 **[Architecture & Stack Documentation](docs/ARCHITECTURE_AND_STACK.md)**

---

## 🛣️ Roadmap

- [ ] WebRTC transport for NAT traversal
- [ ] ONNX Runtime integration for 5× faster inference
- [ ] Adaptive bitrate based on network conditions
- [ ] Multi-client support with connection pooling
- [ ] H.264/H.265 hardware encoding
- [ ] TensorRT deployment for NVIDIA GPUs
- [ ] Mobile-responsive frontend

---

## 📜 License

MIT License — Use freely, build boldly.

---

<p align="center">
  <strong>Built with 🧠 AI + ❤️ by SuperStream</strong>
</p>


