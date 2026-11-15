import asyncio
import argparse
import cv2
import numpy as np
import logging
import websockets
from websockets.server import serve
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s")

class RealTimeStreamer:
    def __init__(self, source="screen", port=8765, width=640, height=360):
        self.source = source
        self.port = port
        self.width = width
        self.height = height
        self.sct = None
        self.monitor = None
        self.capture = None

        if source == "screen":
            import mss

            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]
        elif source == "webcam":
            self.capture = cv2.VideoCapture(0)
        elif source.isdigit():
            self.capture = cv2.VideoCapture(int(source))
        else:
            if not Path(source).is_file():
                raise FileNotFoundError(f"Video source does not exist: {source}")
            self.capture = cv2.VideoCapture(source)

        if self.capture is not None and not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open video source: {source}")
        
    async def stream_handler(self, websocket):
        logging.info("🟢 Client connected to TRUE Real-Time Streamer")
        try:
            while True:
                start_time = time.perf_counter()
                
                if self.sct is not None:
                    frame = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
                else:
                    ok, frame = self.capture.read()
                    if not ok:
                        if self.source == "webcam" or self.source.isdigit():
                            await asyncio.sleep(0.1)
                            continue
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                # Center-crop to 16:9 so the stream does not distort the image.
                h, w = frame.shape[:2]
                if w / h > self.width / self.height:
                    crop_w = round(h * self.width / self.height)
                    left = (w - crop_w) // 2
                    frame = frame[:, left : left + crop_w]
                else:
                    crop_h = round(w * self.height / self.width)
                    top = (h - crop_h) // 2
                    frame = frame[top : top + crop_h, :]

                lr_frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )
                
                # 4. Compress and Send
                _, buffer = cv2.imencode(".jpg", lr_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                await websocket.send(buffer.tobytes())
                
                # 5. Lock to ~30 FPS to prevent overwhelming the PyTorch receiver
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0.0, (1.0/30.0) - elapsed)
                await asyncio.sleep(sleep_time)
                
        except websockets.exceptions.ConnectionClosed:
            logging.info("🔴 Client disconnected")
        finally:
            if self.capture is not None:
                self.capture.release()
            if self.sct is not None:
                self.sct.close()

    async def start(self):
        logging.info("⚡ SuperStream Live Host running on ws://localhost:%s", self.port)
        async with serve(self.stream_handler, "0.0.0.0", self.port):
            await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream low-resolution frames over WebSockets")
    parser.add_argument(
        "--source",
        default="screen",
        help="Video file, webcam, numeric camera index, or screen (default: screen)",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        streamer = RealTimeStreamer(source=args.source, port=args.port)
        asyncio.run(streamer.start())
    except KeyboardInterrupt:
        logging.info("Host streamer stopped.")