"""Test LPR through Frigate's internal pipeline (mixin.py).

This script exercises the same code path that Frigate's real-time and
post-processing LPR uses, but feeds images directly instead of requiring
a camera stream and car detection.

Usage (inside the container):
    python3 /test_images/test_lpr_frigate_service.py /test_images /config/model_cache

Requirements:
    - Models pre-downloaded in /config/model_cache/paddleocr-onnx/
    - Container started with NPU device access
"""

import os
import sys
import time

sys.path.insert(0, "/opt/frigate")
os.environ.setdefault("CONFIG_DIR", "/config")

import cv2
import numpy as np


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_dir> [model_cache_dir]")
        sys.exit(1)

    image_dir = sys.argv[1]
    model_cache = sys.argv[2] if len(sys.argv) > 2 else "/config/model_cache"

    # Set environment so frigate modules find the model cache
    os.environ["MODEL_CACHE_DIR"] = model_cache

    print("=" * 60)
    print("LPR Frigate Service Pipeline Test")
    print("=" * 60)
    print()

    # Import frigate modules - avoid circular imports by importing directly
    from unittest.mock import MagicMock

    import openvino as ov

    core = ov.Core()
    has_npu = "NPU" in core.available_devices
    device = "NPU" if has_npu else "CPU"
    print(f"OpenVINO: {ov.__version__}")
    print(f"Device: {device}")
    print(f"Available: {core.available_devices}")
    print()

    # Import after setting up env to avoid circular import issues
    from frigate.embeddings.onnx.lpr_embedding import (
        PaddleOCRClassification,
        PaddleOCRDetection,
        PaddleOCRRecognition,
    )

    # Create model runner manually (avoiding LicensePlateModelRunner circular import)
    print("Initializing PaddleOCR models...")
    requestor = MagicMock()

    print("  Loading detection model...")
    detection_model = PaddleOCRDetection(
        model_size="small", requestor=requestor, device=device
    )
    detection_model._load_model_and_utils()
    print(f"    OK (runner type: {type(detection_model.runner).__name__})")

    print("  Loading classification model...")
    classification_model = PaddleOCRClassification(
        model_size="small", requestor=requestor, device=device
    )
    classification_model._load_model_and_utils()
    print(f"    OK (runner type: {type(classification_model.runner).__name__})")

    print("  Loading recognition model...")
    recognition_model = PaddleOCRRecognition(
        model_size="small", requestor=requestor, device=device
    )
    recognition_model._load_model_and_utils()
    print(f"    OK (runner type: {type(recognition_model.runner).__name__})")

    # Check recognition model input width (should be 320 on NPU)
    rec_width = recognition_model.runner.get_input_width()
    print(f"  Recognition input width: {rec_width}")
    print()

    # Build a mock model_runner object
    class MockModelRunner:
        pass

    model_runner = MockModelRunner()
    model_runner.detection_model = detection_model
    model_runner.classification_model = classification_model
    model_runner.recognition_model = recognition_model

    # Import mixin
    from frigate.data_processing.common.license_plate.mixin import (
        LicensePlateProcessingMixin,
    )

    # Create a minimal LPR config
    from frigate.config.classification import LicensePlateRecognitionConfig

    lpr_config = LicensePlateRecognitionConfig(
        enabled=True,
        recognition_threshold=0.5,  # Lower threshold for testing
        min_plate_length=2,
        detection_threshold=0.5,
    )

    # CTC decoder is needed for recognition output - it's defined in mixin.py
    from frigate.data_processing.common.license_plate.mixin import CTCDecoder

    ctc_decoder = CTCDecoder(
        character_dict_path=os.path.join(model_cache, "paddleocr-onnx", "ppocr_keys_v1.txt")
    )

    class TestProcessor(LicensePlateProcessingMixin):
        def __init__(self, model_runner_instance, config):
            self.model_runner = model_runner_instance
            self.lpr_config = config
            # Mock the full config
            self.config = MagicMock()
            self.config.lpr.debug_save_plates = False
            # Camera-level config for enhancement
            cam_lpr = MagicMock()
            cam_lpr.enhancement = 0
            self.config.cameras.__getitem__ = lambda s, k: MagicMock(lpr=cam_lpr)
            # Initialize mixin parameters
            self.batch_size = 6
            self.ctc_decoder = ctc_decoder
            self.min_size = 8
            self.max_size = 960
            self.box_thresh = 0.6
            self.mask_thresh = 0.6
            self.similarity_threshold = 0.8
            self.cluster_threshold = 0.85

    processor = TestProcessor(model_runner, lpr_config)

    # Process test images
    images = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} image(s) through Frigate LPR pipeline...")
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

        t0 = time.perf_counter()
        plates, scores, areas = processor._process_license_plate(
            camera="test_cam",
            id=f"test_{img_name}",
            image=image,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        if plates:
            for i, (plate, score, area) in enumerate(zip(plates, scores, areas)):
                avg_score = sum(score) / len(score) if score else 0
                print(
                    f'  Plate[{i}]: "{plate}" '
                    f"(confidence={avg_score:.3f}, area={area}, time={elapsed:.1f}ms)"
                )
                total_plates += 1
        else:
            print(f"  No plates detected ({elapsed:.1f}ms)")

        print()

    print("=" * 60)
    print(f"Summary: {total_plates} plate(s) from {len(images)} image(s)")
    print(f"Device: {device}")
    print("=" * 60)


if __name__ == "__main__":
    main()
