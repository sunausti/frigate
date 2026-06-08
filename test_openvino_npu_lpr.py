"""Test PaddleOCR LPR models on Intel OpenVINO devices (CPU/GPU/NPU).

Usage:
    # Download models first (if not already cached by Frigate):
    mkdir -p models && cd models
    wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v5/detection_v5-small.onnx
    wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx
    wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/recognition_v4.onnx

    # Run test:
    python test_openvino_npu_lpr.py --model-dir ./models
    python test_openvino_npu_lpr.py --model-dir ./models --device NPU
    python test_openvino_npu_lpr.py --model-dir ./models --device GPU
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import openvino as ov
except ImportError:
    print("ERROR: openvino package not installed. Run: pip install openvino")
    sys.exit(1)


def print_separator():
    print("=" * 70)


def print_device_info(core: ov.Core):
    """Print available OpenVINO devices and their properties."""
    print_separator()
    print(f"OpenVINO version: {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    print_separator()

    for device in core.available_devices:
        try:
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {device}: {full_name}")
        except Exception:
            print(f"  {device}: (unable to get device name)")
    print_separator()


def test_model_compile(
    core: ov.Core, model_path: str, device: str, model_name: str
) -> tuple[bool, str]:
    """Try to compile a model on the specified device.

    Returns:
        (success, error_message)
    """
    print(f"\n[{model_name}] Compiling on {device}...")

    if not os.path.isfile(model_path):
        return False, f"Model file not found: {model_path}"

    try:
        start = time.perf_counter()
        compiled_model = core.compile_model(model_path, device)
        compile_time = (time.perf_counter() - start) * 1000
        print(f"  Compile SUCCESS ({compile_time:.1f} ms)")

        # Print model input/output info
        print(f"  Inputs:")
        for inp in compiled_model.inputs:
            print(f"    - {inp.get_any_name()}: {inp.get_partial_shape()} ({inp.get_element_type()})")
        print(f"  Outputs:")
        for out in compiled_model.outputs:
            print(f"    - {out.get_any_name()}: {out.get_partial_shape()} ({out.get_element_type()})")

        return True, ""
    except Exception as e:
        return False, str(e)


def test_model_inference(
    core: ov.Core,
    model_path: str,
    device: str,
    model_name: str,
    input_data: dict[str, np.ndarray],
    num_runs: int = 10,
) -> tuple[bool, str, float]:
    """Try to run inference on the specified device.

    Returns:
        (success, error_message, avg_latency_ms)
    """
    print(f"\n[{model_name}] Running inference on {device}...")

    try:
        compiled_model = core.compile_model(model_path, device)
        infer_request = compiled_model.create_infer_request()

        # Warmup
        infer_request.infer(input_data)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            infer_request.infer(input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        # Get output info
        outputs = []
        for i, out in enumerate(compiled_model.outputs):
            output_data = infer_request.get_output_tensor(i).data
            outputs.append(output_data)
            print(f"  Output[{i}] shape: {output_data.shape}, dtype: {output_data.dtype}")

        print(f"  Latency: avg={avg_latency:.2f}ms, min={min_latency:.2f}ms, max={max_latency:.2f}ms")

        return True, "", avg_latency

    except Exception as e:
        return False, str(e), 0.0


def test_reset_state(
    core: ov.Core, model_path: str, device: str, model_name: str
) -> tuple[bool, str]:
    """Test if reset_state() works on the model (required for RNN models)."""
    print(f"\n[{model_name}] Testing reset_state() on {device}...")

    try:
        compiled_model = core.compile_model(model_path, device)
        infer_request = compiled_model.create_infer_request()
        infer_request.reset_state()
        print(f"  reset_state() SUCCESS")
        return True, ""
    except Exception as e:
        print(f"  reset_state() FAILED: {e}")
        return False, str(e)


def test_dynamic_shape(
    core: ov.Core, model_path: str, device: str, model_name: str,
    shapes: list[tuple[int, ...]], input_name: str = "x"
) -> tuple[bool, str]:
    """Test if model works with different input shapes (dynamic shape support)."""
    print(f"\n[{model_name}] Testing dynamic shapes on {device}...")

    try:
        compiled_model = core.compile_model(model_path, device)
        infer_request = compiled_model.create_infer_request()

        for shape in shapes:
            dummy = np.random.randn(*shape).astype(np.float32)
            try:
                infer_request.infer({input_name: dummy})
                print(f"  Shape {shape}: OK")
            except Exception as e:
                print(f"  Shape {shape}: FAILED - {e}")
                return False, str(e)

        print(f"  Dynamic shapes: ALL PASSED")
        return True, ""
    except Exception as e:
        return False, str(e)


def test_static_reshape(
    core: ov.Core, model_path: str, device: str, model_name: str,
    static_shape: dict[str, list[int]]
) -> tuple[bool, str]:
    """Test if model can be reshaped to static dimensions for NPU compatibility."""
    print(f"\n[{model_name}] Testing static reshape {static_shape} on {device}...")

    try:
        model = core.read_model(model_path)

        # Reshape to static
        model.reshape(static_shape)

        compiled_model = core.compile_model(model, device)
        infer_request = compiled_model.create_infer_request()

        # Create matching input
        input_data = {}
        for name, shape in static_shape.items():
            input_data[name] = np.random.randn(*shape).astype(np.float32)

        infer_request.infer(input_data)
        print(f"  Static reshape + inference: SUCCESS")
        return True, ""
    except Exception as e:
        print(f"  Static reshape FAILED: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Test PaddleOCR LPR models on OpenVINO devices"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directory containing ONNX model files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (CPU, GPU, NPU). If not set, tests all available devices.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of inference runs for benchmarking",
    )
    args = parser.parse_args()

    core = ov.Core()
    print_device_info(core)

    # Model paths
    models = {
        "detection": {
            "path": os.path.join(args.model_dir, "detection_v5-small.onnx"),
            "input_name": "x",
            "input_shape": (1, 3, 960, 960),
            "description": "PaddleOCR Text Detection (CNN, no RNN)",
            "has_rnn": False,
        },
        "classification": {
            "path": os.path.join(args.model_dir, "classification.onnx"),
            "input_name": "x",
            "input_shape": (1, 3, 48, 192),
            "description": "PaddleOCR Orientation Classification (CNN)",
            "has_rnn": False,
        },
        "recognition": {
            "path": os.path.join(args.model_dir, "recognition_v4.onnx"),
            "input_name": "x",
            "input_shape": (1, 3, 48, 320),
            "description": "PaddleOCR Text Recognition (CNN + LSTM/RNN)",
            "has_rnn": True,
        },
    }

    # Determine devices to test
    if args.device:
        devices = [args.device]
    else:
        devices = [d for d in core.available_devices if d != "GNA"]

    # Check model files exist
    missing = []
    for name, info in models.items():
        if not os.path.isfile(info["path"]):
            missing.append(f"  {name}: {info['path']}")

    if missing:
        print("ERROR: Missing model files:")
        print("\n".join(missing))
        print("\nDownload them with:")
        print(f"  mkdir -p {args.model_dir} && cd {args.model_dir}")
        print("  wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v5/detection_v5-small.onnx")
        print("  wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx")
        print("  wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/recognition_v4.onnx")
        sys.exit(1)

    # Results summary
    results = {}

    for device in devices:
        print_separator()
        print(f"TESTING DEVICE: {device}")
        print_separator()

        results[device] = {}

        for model_name, info in models.items():
            print(f"\n{'─' * 50}")
            print(f"Model: {model_name} - {info['description']}")
            print(f"File:  {info['path']}")
            print(f"{'─' * 50}")

            result = {"compile": False, "inference": False, "reset_state": None, "latency_ms": 0}

            # Test 1: Compile
            success, err = test_model_compile(core, info["path"], device, model_name)
            result["compile"] = success
            if not success:
                print(f"  COMPILE FAILED: {err}")
                # If NPU fails, try static reshape
                if device == "NPU":
                    print(f"\n  Trying static reshape workaround for NPU...")
                    static_ok, static_err = test_static_reshape(
                        core, info["path"], device, model_name,
                        {info["input_name"]: list(info["input_shape"])}
                    )
                    result["static_reshape"] = static_ok
                results[device][model_name] = result
                continue

            # Test 2: Inference with fixed shape
            input_data = {
                info["input_name"]: np.random.randn(*info["input_shape"]).astype(np.float32)
            }
            success, err, latency = test_model_inference(
                core, info["path"], device, model_name, input_data, args.num_runs
            )
            result["inference"] = success
            result["latency_ms"] = latency
            if not success:
                print(f"  INFERENCE FAILED: {err}")

            # Test 3: reset_state (only for RNN models)
            if info["has_rnn"]:
                success, err = test_reset_state(core, info["path"], device, model_name)
                result["reset_state"] = success

            # Test 4: Dynamic shapes (recognition model has dynamic width)
            if model_name == "recognition" and result["inference"]:
                dynamic_shapes = [
                    (1, 3, 48, 160),  # short plate
                    (1, 3, 48, 320),  # medium plate
                    (1, 3, 48, 480),  # long plate
                ]
                success, err = test_dynamic_shape(
                    core, info["path"], device, model_name,
                    dynamic_shapes, info["input_name"]
                )
                result["dynamic_shapes"] = success

            results[device][model_name] = result

    # Print summary
    print("\n")
    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Device':<8} {'Model':<16} {'Compile':<10} {'Inference':<12} {'Latency':<12} {'reset_state':<14} {'Dynamic':<10}")
    print("-" * 82)

    for device, device_results in results.items():
        for model_name, result in device_results.items():
            compile_str = "OK" if result["compile"] else "FAIL"
            infer_str = "OK" if result["inference"] else "FAIL"
            latency_str = f"{result['latency_ms']:.2f}ms" if result["latency_ms"] > 0 else "-"
            reset_str = (
                "OK" if result.get("reset_state") is True
                else "FAIL" if result.get("reset_state") is False
                else "N/A"
            )
            dynamic_str = (
                "OK" if result.get("dynamic_shapes") is True
                else "FAIL" if result.get("dynamic_shapes") is False
                else "N/A"
            )
            print(f"{device:<8} {model_name:<16} {compile_str:<10} {infer_str:<12} {latency_str:<12} {reset_str:<14} {dynamic_str:<10}")

    print_separator()

    # Recommendation
    print("\nRECOMMENDATION FOR FRIGATE:")
    npu_results = results.get("NPU", {})
    if not npu_results:
        print("  NPU not detected or not tested. Use --device NPU to force test.")
    else:
        all_ok = all(r["inference"] for r in npu_results.values())
        if all_ok:
            rec_reset = npu_results.get("recognition", {}).get("reset_state")
            rec_dynamic = npu_results.get("recognition", {}).get("dynamic_shapes")
            print("  All models compile and run on NPU!")
            if rec_reset is False:
                print("  WARNING: reset_state() fails on NPU for recognition model.")
                print("  The recognition model may produce incorrect results for")
                print("  consecutive inferences without state reset between them.")
                print("  Consider creating a new infer_request per inference instead.")
            if rec_dynamic is False:
                print("  WARNING: Dynamic shapes not supported on NPU.")
                print("  Need to use static reshape (fixed width) for recognition model.")
            print("\n  To enable in Frigate, remove 'paddleocr' from")
            print("  OpenVINOModelRunner.is_model_npu_supported() exclusion list in")
            print("  frigate/detectors/detection_runners.py:281")
        else:
            failed = [name for name, r in npu_results.items() if not r["inference"]]
            passed = [name for name, r in npu_results.items() if r["inference"]]
            if passed:
                print(f"  Partial NPU support: {', '.join(passed)} work on NPU.")
                print(f"  Failed on NPU: {', '.join(failed)}")
                print("  Consider hybrid approach: working models on NPU, others on GPU.")
            else:
                print("  NPU does not support these PaddleOCR models.")
                print("  Keep using GPU for LPR (current Frigate default behavior).")


if __name__ == "__main__":
    main()
