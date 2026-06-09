"""Test LPR pipeline on Intel NPU.

Run inside the Frigate container with models pre-downloaded:
    docker exec <container> python3 /test_images/test_lpr_npu.py /test_images

Or mount this script and images into the container:
    docker run --rm --network host \
      --device /dev/accel0:/dev/accel/accel0 \
      --device /dev/dri:/dev/dri \
      -v /path/to/config:/config \
      -v /path/to/tests/lpr_npu:/test_images \
      frigate:LPR_OV202602_v2 \
      python3 /test_images/test_lpr_npu.py /test_images
"""

import os
import sys
import time

import cv2
import numpy as np
import openvino as ov


def load_characters(keys_path: str) -> list[str] | None:
    """Load PaddleOCR character dictionary."""
    if not os.path.exists(keys_path):
        print(f"WARNING: {keys_path} not found, will show indices only")
        return None
    with open(keys_path, "r", encoding="utf-8") as f:
        chars = f.read().splitlines()
    # Index 0 = CTC blank, append space at end
    chars = ["blank"] + chars + [" "]
    return chars


def preprocess_detection(image: np.ndarray, max_size: int = 960):
    """Preprocess image for PaddleOCR detection model."""
    h, w = image.shape[:2]
    ratio = min(max_size / max(h, w), 1.0)
    resize_h = max(int(round(int(h * ratio) / 32) * 32), 32)
    resize_w = max(int(round(int(w * ratio) / 32) * 32), 32)
    resized = cv2.resize(image, (resize_w, resize_h))

    mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1).astype("float64")
    std = 1 / np.array([58.395, 57.12, 57.375]).reshape(1, -1).astype("float64")
    img = resized.astype("float32")
    cv2.subtract(img, mean, img)
    cv2.multiply(img, std, img)
    return img.transpose((2, 0, 1))[np.newaxis, ...], ratio, resize_h, resize_w


def preprocess_recognition(
    image: np.ndarray, target_w: int = 320, target_h: int = 48
) -> np.ndarray:
    """Preprocess cropped plate image for PaddleOCR recognition model."""
    h, w = image.shape[:2]
    ratio = w / h
    resized_w = min(target_w, int(target_h * ratio))
    resized = cv2.resize(image, (resized_w, target_h))
    resized = resized.transpose((2, 0, 1)).astype("float32")
    resized = (resized / 255.0 - 0.5) / 0.5
    padded = np.zeros((3, target_h, target_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded[np.newaxis, ...]


def decode_recognition(output: np.ndarray, chars: list[str] | None) -> str:
    """CTC greedy decode recognition output."""
    preds = output[0]
    if len(preds.shape) == 3:
        preds = preds[0]
    indices = np.argmax(preds, axis=1)

    result = []
    prev = 0
    for idx in indices:
        if idx != 0 and idx != prev:
            if chars and idx < len(chars):
                result.append(chars[idx])
            else:
                result.append(f"[{idx}]")
        prev = idx
    return "".join(result)


def find_text_regions(
    det_output: np.ndarray, ratio: float, orig_w: int, orig_h: int, threshold: float = 0.3
) -> list[tuple[int, int, int, int]]:
    """Extract bounding boxes from detection model output."""
    det_map = det_output[0, 0]
    mask = (det_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1 = int(x / ratio) if ratio < 1 else x
        y1 = int(y / ratio) if ratio < 1 else y
        x2 = int((x + bw) / ratio) if ratio < 1 else x + bw
        y2 = int((y + bh) / ratio) if ratio < 1 else y + bh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        if x2 - x1 > 10 and y2 - y1 > 5:
            boxes.append((x1, y1, x2, y2))
    return boxes


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_dir> [model_cache_dir]")
        print(f"  image_dir: directory containing .jpg test images")
        print(f"  model_cache_dir: path to model_cache (default: /config/model_cache)")
        sys.exit(1)

    image_dir = sys.argv[1]
    model_cache = sys.argv[2] if len(sys.argv) > 2 else "/config/model_cache"

    print("=" * 60)
    print("LPR NPU Test")
    print("=" * 60)
    print(f"OpenVINO version: {ov.__version__}")
    print()

    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")

    has_npu = "NPU" in devices
    if not has_npu:
        print("WARNING: NPU not available, using CPU for all models")
    print()

    # Load models
    det_path = os.path.join(model_cache, "paddleocr-onnx/detection_v5-small.onnx")
    cls_path = os.path.join(model_cache, "paddleocr-onnx/classification.onnx")
    rec_path = os.path.join(model_cache, "paddleocr-onnx/recognition_v4.onnx")
    keys_path = os.path.join(model_cache, "paddleocr-onnx/ppocr_keys_v1.txt")

    for p in [det_path, cls_path, rec_path]:
        if not os.path.exists(p):
            print(f"ERROR: Model not found: {p}")
            sys.exit(1)

    print("Loading detection model on CPU...")
    det_model = core.compile_model(det_path, "CPU")
    print("  OK")

    target_device = "NPU" if has_npu else "CPU"

    print(f"Loading classification model on {target_device}...")
    cls_ov = core.read_model(cls_path)
    cls_ov.reshape({"x": [1, 3, 48, 192]})
    cls_model = core.compile_model(cls_ov, target_device)
    print("  OK")

    print(f"Loading recognition model on {target_device}...")
    rec_ov = core.read_model(rec_path)
    rec_ov.reshape({"x": [1, 3, 48, 320]})
    rec_model = core.compile_model(rec_ov, target_device)
    print("  OK")

    chars = load_characters(keys_path)
    if chars:
        print(f"Loaded {len(chars)} characters from dictionary")
    print()

    # Process images
    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} image(s)...")
    print()

    total_plates = 0
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"{img_name}: Failed to read")
            continue

        h, w = image.shape[:2]
        print(f"--- {img_name} ({w}x{h}) ---")

        # Detection
        det_input, ratio, rh, rw = preprocess_detection(image)
        t0 = time.perf_counter()
        det_out = det_model.infer_new_request({"x": det_input})
        det_time = (time.perf_counter() - t0) * 1000
        det_output = list(det_out.values())[0]
        print(f"  Detection ({rw}x{rh}, CPU): {det_time:.1f}ms")

        boxes = find_text_regions(det_output, ratio, w, h)
        print(f"  Found {len(boxes)} text region(s)")

        if not boxes:
            print("  No detection, trying full image as plate region...")
            boxes = [(0, 0, w, h)]

        # Recognition
        for i, (x1, y1, x2, y2) in enumerate(boxes[:5]):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rec_input = preprocess_recognition(crop)
            t0 = time.perf_counter()
            rec_out = rec_model.infer_new_request({"x": rec_input})
            rec_time = (time.perf_counter() - t0) * 1000
            rec_output = list(rec_out.values())[0]

            text = decode_recognition(rec_output, chars) if chars else "(no dict)"
            print(
                f'  Plate[{i}] ({x1},{y1})-({x2},{y2}): "{text}" '
                f"({target_device}, {rec_time:.1f}ms)"
            )
            total_plates += 1

        print()

    print("=" * 60)
    print(f"Summary: {total_plates} plate(s) recognized from {len(images)} image(s)")
    print(f"Device: Detection=CPU, Classification={target_device}, Recognition={target_device}")
    print("=" * 60)


if __name__ == "__main__":
    main()
