"""Measure ESPCN and bicubic quality/speed on an image and a short video clip."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client_engine.model import FrameUpscaler


WIDTH = 640
HEIGHT = 360
VIDEO_FRAME_COUNT = 10


def resize_reference(frame: np.ndarray) -> np.ndarray:
    """Make a 16:9 1080p reference, preferring native source pixels."""
    height, width = frame.shape[:2]
    target_width = WIDTH * 3
    target_height = HEIGHT * 3
    if width >= target_width and height >= target_height:
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        return frame[top : top + target_height, left : left + target_width].copy()

    target_ratio = WIDTH / HEIGHT
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = round(height * target_ratio)
        left = (width - crop_width) // 2
        frame = frame[:, left : left + crop_width]
    elif source_ratio < target_ratio:
        crop_height = round(width / target_ratio)
        top = (height - crop_height) // 2
        frame = frame[top : top + crop_height, :]

    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > target_width else cv2.INTER_CUBIC
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def make_low_resolution(reference: np.ndarray) -> np.ndarray:
    return cv2.resize(reference, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def bicubic_upscale(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3), interpolation=cv2.INTER_CUBIC)


def psnr(reference: np.ndarray, result: np.ndarray) -> float:
    error = np.mean((reference.astype(np.float64) - result.astype(np.float64)) ** 2)
    if error == 0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / error)


def ssim(reference: np.ndarray, result: np.ndarray) -> float:
    """Compute mean channel-wise SSIM with the standard 11x11 Gaussian window."""
    reference_f = reference.astype(np.float64)
    result_f = result.astype(np.float64)
    mu_reference = cv2.GaussianBlur(reference_f, (11, 11), 1.5)
    mu_result = cv2.GaussianBlur(result_f, (11, 11), 1.5)
    mu_reference_sq = mu_reference * mu_reference
    mu_result_sq = mu_result * mu_result
    mu_product = mu_reference * mu_result
    sigma_reference = cv2.GaussianBlur(reference_f * reference_f, (11, 11), 1.5) - mu_reference_sq
    sigma_result = cv2.GaussianBlur(result_f * result_f, (11, 11), 1.5) - mu_result_sq
    sigma_product = cv2.GaussianBlur(reference_f * result_f, (11, 11), 1.5) - mu_product

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    score = ((2 * mu_product + c1) * (2 * sigma_product + c2)) / (
        (mu_reference_sq + mu_result_sq + c1) * (sigma_reference + sigma_result + c2)
    )
    return float(np.mean(score))


def quality(reference: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    return psnr(reference, result), ssim(reference, result)


def print_quality(label: str, reference: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    values = quality(reference, result)
    print(f"  {label}: PSNR={values[0]:.3f} dB, SSIM={values[1]:.5f}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "game.mp4",
        help="Source video; its first frame is also used as the sample image",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=VIDEO_FRAME_COUNT,
        help="Number of video frames to time (default: 10)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional checkpoint path (defaults to client_engine/espcn_fast.pth)",
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(args.frames):
        ok, source_frame = capture.read()
        if not ok:
            break
        reference = resize_reference(source_frame)
        frames.append((make_low_resolution(reference), reference))
    capture.release()
    if not frames:
        raise RuntimeError(f"No video frames could be read from {args.video}")

    upscaler = FrameUpscaler() if args.weights is None else FrameUpscaler(args.weights)
    mode = "bicubic fallback" if upscaler.using_fallback else "ESPCN"
    print(f"Video: {args.video}")
    print(f"Source resolution: {source_width}x{source_height}")
    print(f"Evaluation: {WIDTH}x{HEIGHT} -> {WIDTH * 3}x{HEIGHT * 3}")
    print(f"Device: {upscaler.device}; active upscaler: {mode}")
    print(f"Frames timed: {len(frames)}")

    sample_lr, sample_reference = frames[0]
    sample_result = upscaler.upscale(sample_lr)
    sample_bicubic = bicubic_upscale(sample_lr)
    print("Sample image (first video frame) quality:")
    print_quality(mode, sample_reference, sample_result)
    print_quality("Plain bicubic", sample_reference, sample_bicubic)

    # Warm up once, outside the timed clip, so startup costs do not skew FPS.
    upscaler.upscale(frames[0][0])
    model_times: list[float] = []
    bicubic_times: list[float] = []
    model_scores: list[tuple[float, float]] = []
    bicubic_scores: list[tuple[float, float]] = []
    for low_resolution, reference in frames:
        start = time.perf_counter()
        result = upscaler.upscale(low_resolution)
        model_times.append(time.perf_counter() - start)
        model_scores.append(quality(reference, result))

        start = time.perf_counter()
        bicubic_result = bicubic_upscale(low_resolution)
        bicubic_times.append(time.perf_counter() - start)
        bicubic_scores.append(quality(reference, bicubic_result))

    model_average = float(np.mean(model_times))
    bicubic_average = float(np.mean(bicubic_times))
    print("Short video clip:")
    print(f"  {mode}: {model_average * 1000:.2f} ms/frame, {1.0 / model_average:.2f} FPS")
    print(
        f"  Plain bicubic: {bicubic_average * 1000:.2f} ms/frame, "
        f"{1.0 / bicubic_average:.2f} FPS"
    )
    print(
        f"  Mean {mode} quality: PSNR={np.mean([score[0] for score in model_scores]):.3f} dB, "
        f"SSIM={np.mean([score[1] for score in model_scores]):.5f}"
    )
    print(
        "  Mean plain bicubic quality: "
        f"PSNR={np.mean([score[0] for score in bicubic_scores]):.3f} dB, "
        f"SSIM={np.mean([score[1] for score in bicubic_scores]):.5f}"
    )
    if upscaler.using_fallback:
        print("NOTE: ESPCN weights were unavailable; the active output is bicubic fallback, not neural inference.")
    if source_width * HEIGHT != source_height * WIDTH:
        print("NOTE: Source frames were center-cropped to 16:9 before evaluation.")
    if source_width >= WIDTH * 3 and source_height >= HEIGHT * 3:
        if (source_width, source_height) != (WIDTH * 3, HEIGHT * 3):
            print("NOTE: Quality uses a center-cropped, native-resolution 1920x1080 region.")
    elif (source_width, source_height) != (WIDTH * 3, HEIGHT * 3):
        print("NOTE: The 1920x1080 reference was resized from the source video frame.")


if __name__ == "__main__":
    main()
