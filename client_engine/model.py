"""
============================================================
SuperStream AI — Lightweight Super-Resolution CNN Models
============================================================
FSRCNN (Dong et al., ECCV 2016) and ESPCN (Shi et al., CVPR 2016).

Both operate on Y (luminance) channel. Chroma is upscaled
separately via bicubic/lanczos interpolation.
============================================================
"""

import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import math

UPSCALE_FACTOR = 3
DEFAULT_WEIGHTS = Path(__file__).with_name("espcn_fast.pth")


class FSRCNN(nn.Module):
    """Fast Super-Resolution CNN."""

    def __init__(self, upscale_factor=3, num_channels=1, d=56, s=12, m=4):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2), nn.PReLU(d))
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s))

        mapping = []
        for _ in range(m):
            mapping += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)]
        self.mapping = nn.Sequential(*mapping)

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d))
        self.deconvolution = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9, stride=upscale_factor,
            padding=4, output_padding=upscale_factor - 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return torch.clamp(out, 0.0, 1.0)


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN).

    4-layer CNN ending in nn.PixelShuffle. All convolutions run
    at LR resolution; only the final PixelShuffle produces HR output.

    Architecture:
        Conv1: 5x5, 1->64, ReLU   (broad receptive field)
        Conv2: 3x3, 64->64, ReLU  (deep feature extraction)
        Conv3: 3x3, 64->32, ReLU  (feature compression)
        Conv4: 3x3, 32->r^2        (sub-pixel reconstruction)
        PixelShuffle(r)            (spatial rearrangement)

    About 59.7K parameters for 3x upscale.
    """

    def __init__(self, upscale_factor=3, num_channels=1):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor ** 2),
                               kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == self.upscale_factor ** 2:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = self.conv4(out)
        out = self.pixel_shuffle(out)
        return torch.clamp(out, 0.0, 1.0)


def _available_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrameUpscaler:
    """Load ESPCN once and upscale OpenCV BGR frames to three times size.

    Construct one instance when the inference service starts and reuse it for
    each frame. If the checkpoint is absent or invalid, `upscale` uses bicubic
    interpolation and `using_fallback` reports that state.
    """

    def __init__(
        self,
        weights_path=DEFAULT_WEIGHTS,
        device=None,
        upscale_factor=UPSCALE_FACTOR,
    ):
        if upscale_factor != UPSCALE_FACTOR:
            raise ValueError(f"This frame interface is configured for 3x, got {upscale_factor}x")

        self.device = torch.device(device) if device is not None else _available_device()
        self.upscale_factor = upscale_factor
        self.model = None

        model = ESPCN(upscale_factor=upscale_factor, num_channels=1).to(self.device)
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("Checkpoint does not contain a state dictionary")
            model.load_state_dict(state)
        except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            warnings.warn(
                f"Could not load ESPCN weights from {weights_path}: {exc}. "
                "Using bicubic resize instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            self.model = model.eval()

    @property
    def using_fallback(self):
        return self.model is None

    def upscale(self, frame_bgr):
        """Return a uint8 BGR frame at 3x width and height."""
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8
        ):
            raise ValueError("frame_bgr must be an HxWx3 uint8 NumPy BGR image")

        height, width = frame_bgr.shape[:2]
        output_size = (width * self.upscale_factor, height * self.upscale_factor)
        if self.model is None:
            return cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_CUBIC)

        # The training interface is single-channel, so ESPCN restores luminance.
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        luma = ycrcb[:, :, 0].astype(np.float32) / 255.0
        luma_tensor = torch.from_numpy(luma).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            upscaled_luma = self.model(luma_tensor).clamp(0.0, 1.0)[0, 0]
        luma_output = upscaled_luma.cpu().numpy()
        luma_output = np.rint(luma_output * 255.0).clip(0, 255).astype(np.uint8)

        # Preserve color with inexpensive bicubic resizing of the two chroma planes.
        chroma_output = cv2.resize(
            ycrcb[:, :, 1:],
            output_size,
            interpolation=cv2.INTER_CUBIC,
        )
        ycrcb_output = np.dstack((luma_output, chroma_output))
        return cv2.cvtColor(ycrcb_output, cv2.COLOR_YCrCb2BGR)


def get_model(model_name="fsrcnn", upscale_factor=3, weights_path=None):
    """Factory: create and optionally load a SR model."""
    models = {"fsrcnn": FSRCNN, "espcn": ESPCN}
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")

    model = models[model_name.lower()](upscale_factor=upscale_factor)

    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights from: {weights_path}")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}")
            print("   Continuing with randomly initialized weights.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {model_name.upper()} | {upscale_factor}× | {total_params:,} params | {total_params * 4 / 1024:.1f} KB")
    return model


if __name__ == "__main__":
    import time
    print("=" * 50)
    print("🧪 Model Architecture Test")
    print("=" * 50)
    for name in ["fsrcnn", "espcn"]:
        print(f"\n{'─' * 50}")
        model = get_model(name, upscale_factor=3)
        dummy = torch.randn(1, 1, 360, 640)
        with torch.no_grad():
            _ = model(dummy)
        times = []
        with torch.no_grad():
            for _ in range(20):
                s = time.perf_counter()
                out = model(dummy)
                times.append((time.perf_counter() - s) * 1000)
        avg = sum(times[5:]) / len(times[5:])
        print(f"   In: {list(dummy.shape)} → Out: {list(out.shape)}")
        print(f"   Avg: {avg:.2f}ms | ~{1000/avg:.1f} FPS")
