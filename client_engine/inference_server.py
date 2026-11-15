"""
============================================================
SuperStream AI — Client ML Inference Server (v2.0)
============================================================
ZERO-LATENCY QUEUE + VISUAL QUALITY FIXES

Major Changes from v1.0:
    ✘ No queue management — frames piled up causing lag.
    ✔ asyncio.Queue(maxsize=1) — always processes the LATEST
      frame, drops stale ones instantly.
    ✘ Dark output — missing tensor normalization.
    ✔ Explicit uint8→float32 (0-255→0.0-1.0) pre-processing
      and float32→uint8 post-processing with np.clip.
    ✘ Bicubic fallback looked blurry.
    ✔ INTER_LANCZOS4 upscale + Unsharp Mask sharpening for
      crisp, high-fidelity fallback output.

Architecture:
    Host Server (ws://host:8765)
        ↓ (LR JPEG frames, binary)
    [This Server] — Frame Queue (maxsize=1, drop-old policy)
        ↓ (inference / sharpen)
    Frontend Dashboard (ws://localhost:8766)

Usage:
    python client_engine/inference_server.py --host-ws ws://localhost:8765
    python client_engine/inference_server.py --host-ws ws://localhost:8765 --fallback
============================================================
"""

import asyncio
import argparse
import base64
import json
import logging
import time
import sys
import signal
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ─── Conditional PyTorch Import ───────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not found. Using enhanced OpenCV fallback.")

import websockets
from websockets.server import serve

# ─── Local imports ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client_engine.model import get_model

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SuperStream.Engine")

# ─── Constants ────────────────────────────────────────────────
DEFAULT_PORT = 8766
DEFAULT_HOST_WS = "ws://localhost:8765"
DEFAULT_UPSCALE = 3
DEFAULT_MODEL = "fsrcnn"
HR_JPEG_QUALITY = 85
METRICS_WINDOW = 60

# ── Unsharp Mask defaults (for OpenCV fallback) ──
USM_RADIUS = 3          # GaussianBlur kernel size (must be odd)
USM_SIGMA = 1.0         # Gaussian sigma
USM_AMOUNT = 1.5        # Sharpening strength (1.0=subtle, 2.0=aggressive)
USM_THRESHOLD = 0       # Not used in addWeighted variant


class MetricsTracker:
    """Tracks rolling-window performance metrics."""

    def __init__(self, window_size: int = METRICS_WINDOW):
        self.inference_times = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        self.frame_sizes_lr = deque(maxlen=window_size)
        self.frame_sizes_hr = deque(maxlen=window_size)
        self.dropped_frames = 0
        self.total_frames = 0
        self.start_time = time.perf_counter()

    def record_inference(self, inference_ms: float):
        self.inference_times.append(inference_ms)

    def record_frame(self, lr_size: int, hr_size: int):
        self.frame_times.append(time.perf_counter())
        self.frame_sizes_lr.append(lr_size)
        self.frame_sizes_hr.append(hr_size)
        self.total_frames += 1

    def record_drop(self):
        self.dropped_frames += 1

    @property
    def avg_inference_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    @property
    def current_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / max(span, 0.001)

    @property
    def bandwidth_saved_pct(self) -> float:
        if not self.frame_sizes_lr or not self.frame_sizes_hr:
            return 0.0
        avg_lr = sum(self.frame_sizes_lr) / len(self.frame_sizes_lr)
        avg_hr = sum(self.frame_sizes_hr) / len(self.frame_sizes_hr)
        if avg_hr <= 0:
            return 0.0
        return (1.0 - avg_lr / avg_hr) * 100.0

    def get_snapshot(self) -> dict:
        elapsed = time.perf_counter() - self.start_time
        return {
            "inference_ms": round(self.avg_inference_ms, 2),
            "fps": round(self.current_fps, 1),
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "bandwidth_saved_pct": round(self.bandwidth_saved_pct, 1),
            "uptime_seconds": round(elapsed, 1),
            "avg_lr_size_kb": round(
                sum(self.frame_sizes_lr) / max(len(self.frame_sizes_lr), 1) / 1024, 1
            ),
            "avg_hr_size_kb": round(
                sum(self.frame_sizes_hr) / max(len(self.frame_sizes_hr), 1) / 1024, 1
            ),
        }


class SuperResolutionEngine:
    """
    Core upscaling engine with two paths:
      1. Neural: FSRCNN/ESPCN with correct tensor normalization.
      2. Fallback: INTER_LANCZOS4 + Unsharp Mask sharpening.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        upscale_factor: int = DEFAULT_UPSCALE,
        weights_path: str | None = None,
        use_fallback: bool = False,
    ):
        self.upscale_factor = upscale_factor
        self.use_fallback = use_fallback
        self.model = None
        self.device = None
        self.model_name = model_name

        if not use_fallback and TORCH_AVAILABLE:
            self._init_model(model_name, upscale_factor, weights_path)
        else:
            if not TORCH_AVAILABLE:
                logger.warning("⚠️ PyTorch unavailable — enhanced OpenCV fallback")
            else:
                logger.info("🔄 Fallback mode enabled (--fallback flag)")
            self.use_fallback = True
            self.model_name = "lanczos4+usm"

    def _init_model(self, name: str, factor: int, weights: str | None):
        """Initialize the neural network model with device selection."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🎮 Using CUDA GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🍎 Using Apple Metal (MPS)")
            else:
                self.device = torch.device("cpu")
                logger.info("💻 Using CPU")

            self.model = get_model(name, factor, weights)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Warmup
            logger.info("🔥 Warming up model...")
            dummy = torch.randn(1, 1, 120, 160).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy)
            logger.info("✅ Model ready")

        except Exception as e:
            logger.error(f"❌ Model init failed: {e}")
            logger.info("🔄 Falling back to Lanczos4+USM")
            self.use_fallback = True
            self.model_name = "lanczos4+usm"

    def upscale_frame(self, lr_frame: np.ndarray) -> tuple[np.ndarray, float]:
        if self.use_fallback:
            return self._upscale_lanczos_usm(lr_frame)
        return self._upscale_neural(lr_frame)

    def _apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Fast Unsharp Mask to boost perceived sharpness.

        USM formula: sharpened = original + amount * (original - blurred)
        Implemented as: cv2.addWeighted(original, 1+amount, blurred, -amount, 0)

        This artificially enhances edge contrast, making a JPEG-compressed
        stream look closer to a raw high-bitrate feed.
        """
        blurred = cv2.GaussianBlur(image, (USM_RADIUS, USM_RADIUS), USM_SIGMA)
        sharpened = cv2.addWeighted(
            image, 1.0 + USM_AMOUNT,
            blurred, -USM_AMOUNT,
            0,
        )
        return sharpened

    def _upscale_lanczos_usm(self, lr_frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Enhanced fallback: Lanczos4 upscale + Unsharp Mask.

        INTER_LANCZOS4 uses an 8×8 Lanczos kernel — the highest quality
        resampling filter in OpenCV. Combined with USM post-sharpening,
        this produces output that is visually MUCH sharper than plain
        INTER_CUBIC, approaching the perceived quality of a real
        super-resolution model.
        """
        start = time.perf_counter()

        h, w = lr_frame.shape[:2]
        target_w = w * self.upscale_factor
        target_h = h * self.upscale_factor

        # Step 1: High-quality Lanczos4 upscale
        hr_frame = cv2.resize(
            lr_frame,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Step 2: Unsharp Mask to recover perceived edge detail
        hr_frame = self._apply_unsharp_mask(hr_frame)

        inference_ms = (time.perf_counter() - start) * 1000
        return hr_frame, inference_ms

    def _upscale_neural(self, lr_frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Neural upscale with CORRECT tensor normalization.

        Pre-processing (THE FIX for dark output):
            uint8 [0, 255] → float32 [0.0, 1.0] via /255.0

        Post-processing:
            float32 [0.0, 1.0] → clip → *255 → uint8
        """
        start = time.perf_counter()

        h, w = lr_frame.shape[:2]
        target_w = w * self.upscale_factor
        target_h = h * self.upscale_factor

        # ── Color space: BGR → YCrCb ──
        ycrcb = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]

        # ── PRE-PROCESSING: Normalize Y to [0.0, 1.0] float32 ──
        # This is CRITICAL. The model expects inputs in [0,1].
        # Feeding raw uint8 [0,255] causes the model to output
        # near-zero values → completely dark frames.
        y_float = y_channel.astype(np.float32) / 255.0
        y_tensor = torch.from_numpy(y_float).unsqueeze(0).unsqueeze(0)
        y_tensor = y_tensor.to(self.device)

        # ── Inference ──
        with torch.no_grad():
            hr_y_tensor = self.model(y_tensor)

        # ── POST-PROCESSING: Denormalize back to uint8 ──
        hr_y = hr_y_tensor.squeeze().cpu().numpy()
        hr_y = np.clip(hr_y * 255.0, 0, 255).astype(np.uint8)

        # ── Ensure dimensions match ──
        if hr_y.shape != (target_h, target_w):
            hr_y = cv2.resize(hr_y, (target_w, target_h),
                              interpolation=cv2.INTER_LANCZOS4)

        # ── Upscale chroma with Lanczos4 (better than cubic) ──
        hr_cr = cv2.resize(cr_channel, (target_w, target_h),
                           interpolation=cv2.INTER_LANCZOS4)
        hr_cb = cv2.resize(cb_channel, (target_w, target_h),
                           interpolation=cv2.INTER_LANCZOS4)

        # ── Merge and convert back to BGR ──
        hr_ycrcb = np.stack([hr_y, hr_cr, hr_cb], axis=2)
        hr_frame = cv2.cvtColor(hr_ycrcb, cv2.COLOR_YCrCb2BGR)

        inference_ms = (time.perf_counter() - start) * 1000
        return hr_frame, inference_ms


class InferenceServer:
    """
    Full pipeline with ZERO-LATENCY frame queue management.

    The key insight: use asyncio.Queue(maxsize=1) as the bridge
    between the WebSocket receiver and the inference processor.
    When a new frame arrives and the queue is full (the previous
    frame hasn't been processed yet), we DROP the old frame and
    replace it with the new one. This guarantees the inference
    loop always processes the LATEST frame, not a stale one
    from 500ms ago.
    """

    def __init__(self, host_ws_url: str, port: int, engine: SuperResolutionEngine):
        self.host_ws_url = host_ws_url
        self.port = port
        self.engine = engine
        self.metrics = MetricsTracker()
        self.frontend_clients: set = set()
        self._running = True
        self._latest_payload: str | None = None
        self._frame_ready = asyncio.Event()

        # ══════════════════════════════════════════════════════
        # THE CRITICAL FIX: maxsize=1 queue with drop-old policy.
        # This is a bounded single-slot buffer. The receiver
        # coroutine writes here; the processor reads from here.
        # If the processor is slow, the receiver REPLACES the
        # queued frame with the newest one — zero build-up.
        # ══════════════════════════════════════════════════════
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def _connect_to_host(self):
        """
        Connect to host and RECEIVE frames into the queue.
        Implements drop-old-frame policy via queue replacement.
        """
        retry_delay = 1.0
        max_retry = 30.0

        while self._running:
            try:
                logger.info(f"🔗 Connecting to Host: {self.host_ws_url}")
                async with websockets.connect(
                    self.host_ws_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.info("✅ Connected to Host Server!")
                    retry_delay = 1.0

                    async for message in ws:
                        if not self._running:
                            break

                        # ── DROP-OLD POLICY ──
                        # If the queue is full (inference is still
                        # processing), drain the stale frame and
                        # replace it with this fresh one.
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                                self.metrics.record_drop()
                            except asyncio.QueueEmpty:
                                pass

                        await self._frame_queue.put(message)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"🔌 Host connection closed: {e}")
            except ConnectionRefusedError:
                logger.warning(f"🔌 Host not available at {self.host_ws_url}")
            except Exception as e:
                logger.error(f"❌ Host connection error: {e}")

            if self._running:
                logger.info(f"🔄 Reconnecting in {retry_delay:.0f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry)

    async def _process_frames(self):
        """
        Continuously pull the latest frame from the queue,
        run inference, and broadcast results to frontend.
        """
        while self._running:
            try:
                # Block until a frame is available (with timeout
                # so we can check _running periodically).
                try:
                    raw_data = await asyncio.wait_for(
                        self._frame_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # ── Decode JPEG ──
                np_arr = np.frombuffer(raw_data, np.uint8)
                lr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if lr_frame is None:
                    logger.warning("⚠️ Failed to decode frame")
                    continue

                lr_h, lr_w = lr_frame.shape[:2]

                # ── Run inference / upscale ──
                hr_frame, inference_ms = self.engine.upscale_frame(lr_frame)
                hr_h, hr_w = hr_frame.shape[:2]

                self.metrics.record_inference(inference_ms)

                # ── Encode both frames as JPEG ──
                _, lr_jpg = cv2.imencode(
                    ".jpg", lr_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                _, hr_jpg = cv2.imencode(
                    ".jpg", hr_frame, [cv2.IMWRITE_JPEG_QUALITY, HR_JPEG_QUALITY]
                )

                lr_bytes = lr_jpg.tobytes()
                hr_bytes = hr_jpg.tobytes()
                self.metrics.record_frame(len(raw_data), len(hr_bytes))

                # ── Build JSON payload ──
                snap = self.metrics.get_snapshot()
                snap.update({
                    "model_name": self.engine.model_name.upper(),
                    "upscale_factor": self.engine.upscale_factor,
                    "input_resolution": f"{lr_w}x{lr_h}",
                    "output_resolution": f"{hr_w}x{hr_h}",
                })

                payload = json.dumps({
                    "type": "frame",
                    "timestamp": time.time(),
                    "lr_frame": base64.b64encode(lr_bytes).decode("ascii"),
                    "hr_frame": base64.b64encode(hr_bytes).decode("ascii"),
                    "metrics": snap,
                })

                self._latest_payload = payload
                self._frame_ready.set()

                # ── Log every 60 frames ──
                if self.metrics.total_frames % 60 == 0:
                    m = snap
                    logger.info(
                        f"📊 Frame #{m['total_frames']:>5d} │ "
                        f"Inf: {m['inference_ms']:>6.1f}ms │ "
                        f"FPS: {m['fps']:>5.1f} │ "
                        f"Dropped: {m['dropped_frames']} │ "
                        f"Model: {m['model_name']} │ "
                        f"{m['input_resolution']} → {m['output_resolution']}"
                    )

            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}", exc_info=True)

    async def _frontend_handler(self, websocket):
        """Send the latest processed frame to a frontend client."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.frontend_clients.add(websocket)
        logger.info(f"🟢 Frontend connected: {client_id} "
                    f"(total: {len(self.frontend_clients)})")

        try:
            while self._running:
                await self._frame_ready.wait()
                self._frame_ready.clear()

                if self._latest_payload:
                    try:
                        await websocket.send(self._latest_payload)
                    except websockets.exceptions.ConnectionClosed:
                        break
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"❌ Frontend error: {e}")
        finally:
            self.frontend_clients.discard(websocket)
            logger.info(f"🔴 Frontend disconnected: {client_id}")

    async def start(self, host: str = "0.0.0.0"):
        """Start the inference server with all concurrent tasks."""
        logger.info("=" * 60)
        logger.info("⚡ SuperStream AI — Client ML Engine v2.0")
        logger.info("=" * 60)
        logger.info(f"🔗 Host Server: {self.host_ws_url}")
        logger.info(f"🌐 Frontend WS: ws://{host}:{self.port}")
        logger.info(f"🧠 Model: {self.engine.model_name.upper()}")
        logger.info(f"🔍 Upscale: {self.engine.upscale_factor}×")
        logger.info(f"🛡️ Fallback: {'ON' if self.engine.use_fallback else 'OFF'}")
        logger.info(f"📦 Queue: maxsize=1 (drop-old policy)")
        logger.info("=" * 60)

        # ── Graceful shutdown ──
        stop = asyncio.get_event_loop().create_future()

        def shutdown_handler():
            logger.info("\n🛑 Shutdown signal received...")
            self._running = False
            self._frame_ready.set()
            if not stop.done():
                stop.set_result(None)

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_handler)
            except NotImplementedError:
                signal.signal(sig, lambda s, f: shutdown_handler())

        # ── Start frontend WS server ──
        async with serve(
            self._frontend_handler, host, self.port,
            max_size=50 * 1024 * 1024,
        ):
            logger.info(f"✅ Frontend server ready on ws://{host}:{self.port}")
            logger.info("⏳ Connecting to Host Server...\n")

            # ── Run host receiver + frame processor concurrently ──
            host_task = asyncio.create_task(self._connect_to_host())
            process_task = asyncio.create_task(self._process_frames())

            await stop

            host_task.cancel()
            process_task.cancel()
            for t in (host_task, process_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        logger.info(f"✅ Stopped. Processed: {self.metrics.total_frames} | "
                    f"Dropped: {self.metrics.dropped_frames}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SuperStream AI — Client ML Inference Server v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client_engine/inference_server.py --host-ws ws://localhost:8765
  python client_engine/inference_server.py --host-ws ws://localhost:8765 --fallback
  python client_engine/inference_server.py --model espcn --weights model.pth
        """,
    )
    parser.add_argument("--host-ws", type=str, default=DEFAULT_HOST_WS,
                        help=f"Host Server WebSocket URL (default: {DEFAULT_HOST_WS})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Frontend WS port (default: {DEFAULT_PORT})")
    parser.add_argument("--bind", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["fsrcnn", "espcn"],
                        help=f"SR model (default: {DEFAULT_MODEL})")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pre-trained weights (.pth)")
    parser.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE,
                        choices=[2, 3, 4],
                        help=f"Upscale factor (default: {DEFAULT_UPSCALE})")
    parser.add_argument("--fallback", action="store_true",
                        help="Force Lanczos4+USM fallback (no neural network)")
    return parser.parse_args()


# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    engine = SuperResolutionEngine(
        model_name=args.model,
        upscale_factor=args.upscale,
        weights_path=args.weights,
        use_fallback=args.fallback,
    )

    server = InferenceServer(
        host_ws_url=args.host_ws,
        port=args.port,
        engine=engine,
    )

    try:
        asyncio.run(server.start(host=args.bind))
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user.")
