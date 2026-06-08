"""Fix PaddleOCR recognition_v4 model for Intel NPU compatibility.

This script:
1. Analyzes the ONNX model to find problematic AvgPool nodes
2. Fixes them by adjusting kernel_shape where kernel > input dimension
3. Saves the fixed model
4. Tests compilation and inference on NPU
5. Compares CPU vs NPU output to verify accuracy is preserved

Usage:
    python fix_recognition_for_npu.py --model /path/to/recognition_v4.onnx

Requirements:
    pip install onnx openvino numpy
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import onnx
    from onnx import helper, shape_inference, TensorProto
    from onnx.tools import update_model_dims
except ImportError:
    print("ERROR: onnx package not installed. Run: pip install onnx")
    sys.exit(1)

try:
    import openvino as ov
except ImportError:
    print("ERROR: openvino package not installed. Run: pip install openvino")
    sys.exit(1)


def print_separator(char="="):
    print(char * 70)


# ─────────────────────────────────────────────────────────────────────
# Step 1: Analyze the model
# ─────────────────────────────────────────────────────────────────────


def analyze_model(model_path: str, input_shape: list[int]) -> list[dict]:
    """Analyze AvgPool nodes and their input shapes with static input."""
    print_separator()
    print("STEP 1: Analyzing model")
    print_separator()

    model = onnx.load(model_path)

    # Set static input shape for shape inference
    # Modify the input to have static dimensions
    for inp in model.graph.input:
        if inp.name == "x":
            dim_proto = inp.type.tensor_type.shape.dim
            for i, d in enumerate(input_shape):
                dim_proto[i].dim_value = d
                dim_proto[i].ClearField("dim_param")

    # Run shape inference to propagate shapes
    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference: OK")
    except Exception as e:
        print(f"Shape inference warning: {e}")

    # Build a map of tensor name -> shape from value_info
    shape_map = {}
    for vi in model.graph.value_info:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        shape_map[vi.name] = dims

    # Also add graph inputs
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        shape_map[inp.name] = dims

    # Find all AvgPool nodes
    avgpool_nodes = []
    for node in model.graph.node:
        if node.op_type in ["AveragePool", "GlobalAveragePool"]:
            info = {
                "name": node.name,
                "op_type": node.op_type,
                "input_name": node.input[0],
                "output_name": node.output[0],
                "kernel_shape": None,
                "strides": None,
                "pads": None,
                "ceil_mode": 0,
                "input_shape": shape_map.get(node.input[0], None),
                "output_shape": shape_map.get(node.output[0], None),
                "problematic": False,
            }

            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    info["kernel_shape"] = list(attr.ints)
                elif attr.name == "strides":
                    info["strides"] = list(attr.ints)
                elif attr.name == "pads":
                    info["pads"] = list(attr.ints)
                elif attr.name == "ceil_mode":
                    info["ceil_mode"] = attr.i

            # Check if problematic: kernel > input spatial dim
            if info["kernel_shape"] and info["input_shape"]:
                input_shape_val = info["input_shape"]
                if len(input_shape_val) >= 4:
                    h = input_shape_val[2]
                    w = input_shape_val[3]
                    kh = info["kernel_shape"][0]
                    kw = info["kernel_shape"][1] if len(info["kernel_shape"]) > 1 else 1

                    if isinstance(h, int) and h < kh:
                        info["problematic"] = True
                        info["problem_detail"] = f"H={h} < kernel_h={kh}"
                    if isinstance(w, int) and w < kw:
                        info["problematic"] = True
                        info["problem_detail"] = f"W={w} < kernel_w={kw}"

            avgpool_nodes.append(info)

    # Print findings
    print(f"\nFound {len(avgpool_nodes)} AvgPool node(s):\n")
    for i, info in enumerate(avgpool_nodes):
        status = "PROBLEMATIC" if info["problematic"] else "OK"
        print(f"  [{i+1}] {info['name']} ({info['op_type']}) [{status}]")
        print(f"      input:  {info['input_name']} -> shape: {info['input_shape']}")
        print(f"      output: {info['output_name']} -> shape: {info['output_shape']}")
        print(f"      kernel_shape: {info['kernel_shape']}")
        print(f"      strides: {info['strides']}")
        print(f"      pads: {info['pads']}")
        if info["problematic"]:
            print(f"      PROBLEM: {info['problem_detail']}")
        print()

    problematic = [n for n in avgpool_nodes if n["problematic"]]
    print(f"Problematic nodes: {len(problematic)} / {len(avgpool_nodes)}")
    return problematic


# ─────────────────────────────────────────────────────────────────────
# Step 2: Fix the model
# ─────────────────────────────────────────────────────────────────────


def fix_model(model_path: str, output_path: str, input_shape: list[int]) -> str:
    """Fix AvgPool nodes by clamping kernel to input dimensions."""
    print_separator()
    print("STEP 2: Fixing model")
    print_separator()

    model = onnx.load(model_path)

    # First, set static input shape and infer shapes
    for inp in model.graph.input:
        if inp.name == "x":
            dim_proto = inp.type.tensor_type.shape.dim
            for i, d in enumerate(input_shape):
                dim_proto[i].dim_value = d
                dim_proto[i].ClearField("dim_param")

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    # Build shape map
    shape_map = {}
    for vi in model.graph.value_info:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else 0)
        shape_map[vi.name] = dims

    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else 0)
        shape_map[inp.name] = dims

    # Fix problematic AvgPool nodes
    fixed_count = 0
    for node in model.graph.node:
        if node.op_type != "AveragePool":
            continue

        input_shape_val = shape_map.get(node.input[0])
        if input_shape_val is None or len(input_shape_val) < 4:
            continue

        h = input_shape_val[2]
        w = input_shape_val[3]

        kernel_attr = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_attr = attr
                break

        if kernel_attr is None:
            continue

        kernel = list(kernel_attr.ints)
        new_kernel = kernel.copy()
        changed = False

        # Clamp kernel to not exceed input spatial dimensions
        if h > 0 and kernel[0] > h:
            new_kernel[0] = h
            changed = True
        if len(kernel) > 1 and w > 0 and kernel[1] > w:
            new_kernel[1] = w
            changed = True

        if changed:
            print(f"  Fixing node: {node.name}")
            print(f"    input shape: {input_shape_val}")
            print(f"    kernel: {kernel} -> {new_kernel}")

            # Update kernel_shape
            kernel_attr.ClearField("ints")
            kernel_attr.ints.extend(new_kernel)

            # Also fix strides if they exceed the new kernel
            for attr in node.attribute:
                if attr.name == "strides":
                    strides = list(attr.ints)
                    new_strides = strides.copy()
                    for j in range(len(new_strides)):
                        if j < len(new_kernel) and new_strides[j] > new_kernel[j]:
                            new_strides[j] = new_kernel[j]
                    if new_strides != strides:
                        print(f"    strides: {strides} -> {new_strides}")
                        attr.ClearField("ints")
                        attr.ints.extend(new_strides)

            fixed_count += 1

    if fixed_count == 0:
        print("  No nodes needed fixing.")
        print("  Trying alternative: setting static shape only...")
        # Reload original and just set static shape
        model = onnx.load(model_path)
        for inp in model.graph.input:
            if inp.name == "x":
                dim_proto = inp.type.tensor_type.shape.dim
                for i, d in enumerate(input_shape):
                    dim_proto[i].dim_value = d
                    dim_proto[i].ClearField("dim_param")

    # Save fixed model
    onnx.save(model, output_path)
    print(f"\n  Saved fixed model: {output_path}")
    print(f"  Fixed {fixed_count} node(s)")
    return output_path


# ─────────────────────────────────────────────────────────────────────
# Step 3: Test on NPU
# ─────────────────────────────────────────────────────────────────────


def test_npu_compile(model_path: str, device: str = "NPU") -> bool:
    """Test if the fixed model compiles on NPU."""
    print_separator()
    print(f"STEP 3: Testing compilation on {device}")
    print_separator()

    core = ov.Core()

    if device not in core.available_devices:
        print(f"  WARNING: {device} not available. Available: {core.available_devices}")
        print(f"  Skipping NPU test, using CPU for validation instead.")
        device = "CPU"

    try:
        start = time.perf_counter()
        compiled = core.compile_model(model_path, device)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Compile on {device}: SUCCESS ({elapsed:.1f} ms)")

        # Print model info
        print(f"  Inputs:")
        for inp in compiled.inputs:
            print(f"    {inp.get_any_name()}: {inp.get_partial_shape()}")
        print(f"  Outputs:")
        for out in compiled.outputs:
            print(f"    {out.get_any_name()}: {out.get_partial_shape()}")

        return True
    except Exception as e:
        print(f"  Compile on {device}: FAILED")
        print(f"  Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Step 4: Compare CPU vs NPU outputs
# ─────────────────────────────────────────────────────────────────────


def compare_outputs(
    original_model_path: str,
    fixed_model_path: str,
    input_shape: list[int],
    npu_device: str = "NPU",
    num_tests: int = 5,
):
    """Compare inference results between CPU (original) and NPU (fixed)."""
    print_separator()
    print("STEP 4: Comparing CPU vs NPU outputs")
    print_separator()

    core = ov.Core()

    # Use original model on CPU (ground truth)
    try:
        cpu_compiled = core.compile_model(original_model_path, "CPU")
        cpu_request = cpu_compiled.create_infer_request()
    except Exception as e:
        print(f"  Failed to compile original model on CPU: {e}")
        return

    # Use fixed model on NPU (or CPU if NPU unavailable)
    target_device = npu_device
    if target_device not in core.available_devices:
        print(f"  {target_device} not available, comparing fixed model on CPU instead")
        target_device = "CPU"

    try:
        npu_compiled = core.compile_model(fixed_model_path, target_device)
        npu_request = npu_compiled.create_infer_request()
    except Exception as e:
        print(f"  Failed to compile fixed model on {target_device}: {e}")
        return

    print(f"  Reference: original model on CPU")
    print(f"  Test:      fixed model on {target_device}")
    print()

    max_abs_diff_all = 0
    max_rel_diff_all = 0
    latencies_cpu = []
    latencies_npu = []

    for i in range(num_tests):
        # Generate random input simulating a preprocessed plate image
        # Values normalized to [0, 1] range as PaddleOCR does
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

        # CPU inference (original model)
        start = time.perf_counter()
        cpu_request.infer({"x": dummy_input})
        latencies_cpu.append((time.perf_counter() - start) * 1000)
        cpu_output = cpu_request.get_output_tensor(0).data.copy()

        # NPU inference (fixed model) - need reset_state for RNN
        try:
            npu_request.reset_state()
        except Exception:
            pass

        start = time.perf_counter()
        npu_request.infer({"x": dummy_input})
        latencies_npu.append((time.perf_counter() - start) * 1000)
        npu_output = npu_request.get_output_tensor(0).data.copy()

        # Compare
        abs_diff = np.abs(cpu_output - npu_output)
        max_abs = abs_diff.max()
        mean_abs = abs_diff.mean()

        # Relative difference (avoid divide by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(
                np.abs(cpu_output) > 1e-7,
                abs_diff / np.abs(cpu_output),
                0,
            )
        max_rel = rel_diff.max()

        max_abs_diff_all = max(max_abs_diff_all, max_abs)
        max_rel_diff_all = max(max_rel_diff_all, max_rel)

        print(
            f"  Test {i+1}: max_abs_diff={max_abs:.6e}, "
            f"mean_abs_diff={mean_abs:.6e}, max_rel_diff={max_rel:.4f}"
        )

    print()
    print(f"  Overall max absolute difference: {max_abs_diff_all:.6e}")
    print(f"  Overall max relative difference: {max_rel_diff_all:.4f}")
    print()
    print(f"  CPU avg latency: {sum(latencies_cpu)/len(latencies_cpu):.2f} ms")
    print(f"  {target_device} avg latency: {sum(latencies_npu)/len(latencies_npu):.2f} ms")
    print()

    # Verdict
    if max_abs_diff_all < 1e-3:
        print("  VERDICT: Outputs are nearly identical. No accuracy impact.")
    elif max_abs_diff_all < 1e-2:
        print("  VERDICT: Minor numerical differences (likely FP16 vs FP32).")
        print("           Should not affect recognition accuracy.")
    elif max_abs_diff_all < 0.1:
        print("  VERDICT: Noticeable differences. Test with real plate images")
        print("           to verify recognition accuracy is acceptable.")
    else:
        print("  VERDICT: Large differences detected! The fix may affect accuracy.")
        print("           Do NOT use in production without further validation.")


# ─────────────────────────────────────────────────────────────────────
# Step 5: Benchmark
# ─────────────────────────────────────────────────────────────────────


def benchmark(model_path: str, input_shape: list[int], device: str, num_runs: int = 50):
    """Benchmark inference speed."""
    print_separator()
    print(f"STEP 5: Benchmarking on {device} ({num_runs} runs)")
    print_separator()

    core = ov.Core()

    if device not in core.available_devices:
        print(f"  {device} not available, skipping benchmark")
        return

    try:
        compiled = core.compile_model(model_path, device)
        request = compiled.create_infer_request()
    except Exception as e:
        print(f"  Failed to compile: {e}")
        return

    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(5):
        try:
            request.reset_state()
        except Exception:
            pass
        request.infer({"x": dummy_input})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        try:
            request.reset_state()
        except Exception:
            pass
        start = time.perf_counter()
        request.infer({"x": dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    avg = sum(latencies) / len(latencies)
    min_l = min(latencies)
    max_l = max(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"  avg: {avg:.2f} ms")
    print(f"  min: {min_l:.2f} ms")
    print(f"  max: {max_l:.2f} ms")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")
    print(f"  throughput: {1000/avg:.1f} inferences/sec")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fix PaddleOCR recognition_v4 model for Intel NPU"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to recognition_v4.onnx",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for fixed model (default: <model>_npu_fixed.onnx)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Fixed input width for NPU (default: 320)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="NPU",
        help="Target device (default: NPU)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of benchmark runs (default: 50)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the benchmark step",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    output_path = args.output or args.model.replace(".onnx", "_npu_fixed.onnx")
    input_shape = [1, 3, 48, args.width]

    print(f"Input model:  {args.model}")
    print(f"Output model: {output_path}")
    print(f"Input shape:  {input_shape}")
    print(f"Target device: {args.device}")
    print()

    # Step 1: Analyze
    problematic_nodes = analyze_model(args.model, input_shape)

    # Step 2: Fix
    fixed_path = fix_model(args.model, output_path, input_shape)

    # Step 3: Test NPU compilation
    compile_ok = test_npu_compile(fixed_path, args.device)

    if not compile_ok:
        print("\nFix did not resolve the compilation issue.")
        print("Possible next steps:")
        print("  1. Try a different --width value (e.g., 160, 256, 480)")
        print("  2. Try OpenVINO IR conversion: ovc <model> --input 'x[1,3,48,320]'")
        print("  3. The model architecture may be fundamentally incompatible with NPU")
        sys.exit(1)

    # Step 4: Compare outputs
    compare_outputs(args.model, fixed_path, input_shape, args.device)

    # Step 5: Benchmark
    if not args.skip_benchmark:
        benchmark(fixed_path, input_shape, args.device, args.num_runs)
        # Also benchmark CPU for comparison
        benchmark(fixed_path, input_shape, "CPU", args.num_runs)

    # Final summary
    print()
    print_separator()
    print("DONE")
    print_separator()
    print(f"Fixed model saved to: {fixed_path}")
    print()
    print("To use in Frigate, you would need to:")
    print("  1. Place the fixed model in the model cache directory")
    print("  2. Remove 'paddleocr' from NPU exclusion list in")
    print("     frigate/detectors/detection_runners.py:281")
    print("  3. Handle the static shape requirement (fixed width)")
    print("     in frigate/data_processing/common/license_plate/mixin.py")


if __name__ == "__main__":
    main()
