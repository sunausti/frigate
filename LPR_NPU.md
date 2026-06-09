# Frigate LPR on Intel NPU

This document describes the changes made to enable Frigate's License Plate Recognition (LPR) pipeline to run on Intel NPU via OpenVINO.

## Overview

The LPR pipeline uses 4 models:

| Model | Purpose | Device | Latency (warm) |
|-------|---------|--------|----------------|
| SSD MobileNet V2 | Object detection (main detector) | NPU | ~5.4ms |
| PaddleOCR Detection | Text region detection | CPU (fallback) | ~29ms |
| PaddleOCR Classification | Text orientation | NPU | ~4.4ms |
| PaddleOCR Recognition | Character recognition | NPU | ~5.4ms |

## Modified Files

### 1. `frigate/detectors/detection_runners.py`

Core NPU support logic in `OpenVINOModelRunner`:

- **Removed `paddleocr` from NPU exclusion list** (`is_model_npu_supported`)
- **Added `device` param to `is_complex_model()`** — PaddleOCR on NPU marked non-complex (no `reset_state` needed)
- **Added `_needs_npu_static_reshape()`** — Identifies PaddleOCR models that need static input shapes for NPU
- **Added `_get_npu_static_shape()`** — Determines correct static shape per model:
  - Recognition `[?,3,48,?]` → `[1,3,48,320]`
  - Classification `[?,3,?,?]` (rank-2 output) → `[1,3,48,192]`
  - Detection `[?,3,?,?]` (spatial output) → `None` (skip reshape)
- **Added `_run_batched_single_input()`** — Handles batch>1 inference on NPU (compiled with batch=1)
- **Added graceful CPU fallback** — If NPU compile fails, automatically falls back to CPU with a warning
- **Class constant `PADDLEOCR_NPU_RECOGNITION_WIDTH = 320`** — Fixed width for recognition model

### 2. `docker/main/Dockerfile`

Added `npu-libs` build stage that copies Intel NPU runtime libraries from `openvino/ubuntu24_dev:2026.2.0`:

- `libopenvino_intel_npu_compiler.so` — NPU model compiler
- `libopenvino_intel_npu_compiler_loader.so` — Compiler loader (dlopen'd by plugin)
- `libnpu_driver_compiler.so` — Low-level NPU driver compiler
- `libze_intel_npu.so` — Level Zero NPU driver
- `libze_loader.so` — Level Zero loader

These are NOT included in the pip `openvino` package and must be added separately.

### 3. `docker/main/requirements-wheels.txt`

```
openvino == 2026.2.*   # (was 2024.6.*)
```

### 4. `docker/main/requirements-ov.txt`

```
openvino-dev==2024.6.0   # Build-time only, provides deprecated `mo` tool
```

Kept at 2024.6.0 because `docker/main/build_ov_model.py` uses `openvino.tools.mo` which was removed in newer versions. This is only used at build time for SSD model conversion.

## Build Instructions

### Full Build

```bash
cd /path/to/frigate
DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_MULTI_PLATFORM=1 \
  --build-arg http_proxy=http://child-prc.intel.com:913 \
  --build-arg https_proxy=http://child-prc.intel.com:913 \
  -t frigate:LPR_OV202602 \
  -f docker/main/Dockerfile .
```

### Incremental Patch Build (faster, reuses existing base image)

If you already have a built `frigate:LPR_OV202602` image and only changed Python code:

```dockerfile
# Dockerfile.patch
FROM openvino/ubuntu24_dev:2026.2.0 AS npu-libs
USER root
RUN mkdir -p /npu-rootfs/usr/lib/x86_64-linux-gnu && \
    cp /usr/lib/x86_64-linux-gnu/libnpu_driver_compiler.so /npu-rootfs/usr/lib/x86_64-linux-gnu/ && \
    cp /usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1.32.1 /npu-rootfs/usr/lib/x86_64-linux-gnu/ && \
    ln -s libze_intel_npu.so.1.32.1 /npu-rootfs/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1 && \
    ln -s libze_intel_npu.so.1 /npu-rootfs/usr/lib/x86_64-linux-gnu/libze_intel_npu.so && \
    cp /usr/lib/x86_64-linux-gnu/libze_loader.so.1.27.0 /npu-rootfs/usr/lib/x86_64-linux-gnu/ && \
    ln -s libze_loader.so.1.27.0 /npu-rootfs/usr/lib/x86_64-linux-gnu/libze_loader.so.1 && \
    ln -s libze_loader.so.1 /npu-rootfs/usr/lib/x86_64-linux-gnu/libze_loader.so && \
    mkdir -p /npu-rootfs/usr/local/lib/python3.11/dist-packages/openvino/libs && \
    cp /opt/intel/openvino_2026.2.0.0/runtime/lib/intel64/libopenvino_intel_npu_compiler.so \
       /npu-rootfs/usr/local/lib/python3.11/dist-packages/openvino/libs/ && \
    cp /opt/intel/openvino_2026.2.0.0/runtime/lib/intel64/libopenvino_intel_npu_compiler_loader.so \
       /npu-rootfs/usr/local/lib/python3.11/dist-packages/openvino/libs/

FROM frigate:LPR_OV202602
COPY --from=npu-libs /npu-rootfs/ /
COPY frigate/detectors/detection_runners.py /opt/frigate/frigate/detectors/detection_runners.py
RUN ldconfig
```

```bash
docker build -t frigate:LPR_OV202602_v2 -f Dockerfile.patch /path/to/frigate
```

## Deploy to Device

```bash
# Save and push
docker save frigate:LPR_OV202602_v2 | gzip > /tmp/frigate_lpr.tar.gz
adb push /tmp/frigate_lpr.tar.gz /data/vendor/docker/sunausti/frigate_lpr.tar.gz
adb shell "docker load < /data/vendor/docker/sunausti/frigate_lpr.tar.gz"
```

## Run Container

```bash
docker run --rm -d --name frigate \
  --network host \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /path/to/config:/config \
  -v /path/to/storage:/media/frigate \
  frigate:LPR_OV202602_v2
```

### Minimal config.yml for LPR + NPU

```yaml
mqtt:
  enabled: false

detectors:
  openvino:
    type: openvino
    device: NPU

lpr:
  enabled: true
  device: NPU

cameras:
  my_camera:
    ffmpeg:
      inputs:
        - path: rtsp://your_camera_ip:554/stream
          roles:
            - detect
    detect:
      width: 1280
      height: 720
```

### Pre-download Models (avoid proxy/network issues)

Models are downloaded from GitHub on first run. To avoid network issues, pre-place them:

```bash
mkdir -p /path/to/config/model_cache/paddleocr-onnx
mkdir -p /path/to/config/model_cache/yolov9_license_plate

# Download and place:
# - paddleocr-onnx/detection_v5-small.onnx
# - paddleocr-onnx/classification.onnx
# - paddleocr-onnx/recognition_v4.onnx
# - yolov9_license_plate/yolov9-256-license-plates.onnx
```

## Testing

### Test Files

Test images and script are located in `tests/lpr_npu/`:

```
tests/lpr_npu/
├── test_lpr_npu.py      # Standalone LPR test script
├── 103.jpg              # CCPD2019 test image (皖A·SD888)
└── 88.jpg               # CCPD2019 test image (皖A·QK091)
```

### Running the Test

**Step 1**: Start the container with test images mounted:

```bash
# On device (via adb shell)
docker run --rm -d --name frigate_lpr_test \
  --network host \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /path/to/config:/config \
  -v /path/to/tests/lpr_npu:/test_images \
  frigate:LPR_OV202602_v2
```

**Step 2**: Wait for Frigate to finish starting (~20s), then run the test script:

```bash
docker exec frigate_lpr_test python3 /test_images/test_lpr_npu.py /test_images /config/model_cache
```

**Step 3**: Expected output:

```
============================================================
LPR NPU Test
============================================================
OpenVINO version: 2026.2.0-21903-52ddc073857-releases/2026/2

Available devices: ['CPU', 'GPU', 'NPU']

Loading detection model on CPU...
  OK
Loading classification model on NPU...
  OK
Loading recognition model on NPU...
  OK
Loaded 6625 characters from dictionary

Processing 2 image(s)...

--- 103.jpg (720x1160) ---
  Detection (608x960, CPU): 76.7ms
  Found 1 text region(s)
  Plate[0] (47,350)-(549,509): "皖A·SD888" (NPU, 12.0ms)

--- 88.jpg (720x1160) ---
  Detection (608x960, CPU): 47.7ms
  Found 1 text region(s)
  Plate[0] (154,443)-(619,621): "皖A·QK091" (NPU, 5.8ms)

============================================================
Summary: 2 plate(s) recognized from 2 image(s)
Device: Detection=CPU, Classification=NPU, Recognition=NPU
============================================================
```

**Step 4**: Stop the container:

```bash
docker stop frigate_lpr_test
```

### Test via adb (end-to-end from host)

```bash
# Push test images to device
adb shell "mkdir -p /data/vendor/docker/sunausti/test_lpr"
adb push tests/lpr_npu/ /data/vendor/docker/sunausti/test_lpr/

# Start container
adb shell "docker run --rm -d --name frigate_lpr_test \
  --network host \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /data/vendor/docker/sunausti/frigate_config:/config \
  -v /data/vendor/docker/sunausti/test_lpr:/test_images \
  frigate:LPR_OV202602_v2"

# Wait for startup then run test
sleep 20
adb shell "docker exec frigate_lpr_test python3 /test_images/test_lpr_npu.py /test_images /config/model_cache"

# Cleanup
adb shell "docker stop frigate_lpr_test"
```

### Test Results (Intel AI Boost NPU)

| Image | Plate | Device | Recognition Time |
|-------|-------|--------|-----------------|
| 103.jpg | 皖A·SD888 | NPU | 12.0ms (cold) / ~5.4ms (warm) |
| 88.jpg | 皖A·QK091 | NPU | 5.8ms |

## Limitations

### PaddleOCR Detection Model Cannot Run on NPU

The detection model has fully dynamic spatial dimensions `[?,3,?,?]` and its output shape depends on input size. The mixin.py preprocessing resizes input images to variable dimensions (multiples of 32, up to 960x960) based on aspect ratio. Since NPU requires all dimensions to be static at compile time, this model cannot run on NPU without changing the preprocessing to always pad/resize to a fixed size (e.g., 960x960).

**Impact**: Detection falls back to CPU (~29ms per inference). This is acceptable since detection runs once per frame and is not the bottleneck.

### Recognition Width Fixed at 320px

The recognition model is compiled with static width=320. Images are padded to exactly 320px by mixin.py (which calls `get_input_width()` returning 320). This may slightly affect recognition accuracy for very wide license plates, though testing shows negligible precision difference (max_abs=0.027).

### NPU Libraries Version-Locked

The NPU driver libraries are copied from `openvino/ubuntu24_dev:2026.2.0`. If the device firmware is updated, these may need to be updated to match. Version mismatch between `libze_intel_npu.so` and device firmware can cause compilation failures.

### Single-Sample Batch Inference

NPU is compiled with batch_size=1. The LPR mixin uses batch_size=6 for classification/recognition, so the `_run_batched_single_input()` method iterates one sample at a time. This is slightly less efficient than true batched inference but avoids the complexity of compiling multiple batch sizes.

### Build-Time OpenVINO Version Split

- **Runtime**: `openvino == 2026.2.*` (pip package)
- **Build-time**: `openvino-dev == 2024.6.0` (for `mo` tool used in SSD model conversion)

This split exists because the `openvino.tools.mo` model optimizer was deprecated/removed in newer OpenVINO versions, but `build_ov_model.py` still requires it.

## Technical Details

### Why NPU Needs Static Shapes

Intel NPU compiler requires all tensor dimensions to be known at compile time. The NPU hardware has fixed-size processing elements that must be configured during model compilation. Dynamic shapes (`?` dimensions) cause compilation failure:

```
Compilation failed. vclAllocatedExecutableCreate3 result: 0x78000004
```

### Why `model.reshape()` Fails on Detection Model

OpenVINO's `model.reshape({'x': [1,3,48,320]})` propagates shapes through the graph. The detection model has an internal `Add` node that broadcasts two tensors:
- `Add.186[0]:f32[1,96,3,20]` + `Resize.0[0]:f32[1,96,4,20]` → shapes incompatible

This happens because 48x320 is wrong for the detection model (it expects much larger spatial dimensions like 640x640 or 960x960). The fix correctly identifies detection models by their output pattern and skips reshape.

### Model Identification Logic

Models are identified by their input/output shape patterns without relying on filenames:

```
Recognition:    input=[?,3,48,?]     output=[?,1..,6625]   → H=48 static
Classification: input=[?,3,?,?]      output=[?,2]          → rank-2 output (2 classes)
Detection:      input=[?,3,?,?]      output=[?,1,32..,32..]→ spatial rank-4 output
```
