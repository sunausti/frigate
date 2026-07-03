# Frigate LPR on Intel NPU

This document describes the changes made to enable Frigate's License Plate Recognition (LPR) pipeline to run on Intel NPU via OpenVINO.

## Overview

The LPR pipeline uses 5 models:

| Model | Purpose | Device | Latency (warm) |
|-------|---------|--------|----------------|
| SSD MobileNet V2 | Object detection (main detector) | NPU | ~5.4ms |
| YOLOv9 License Plate | Plate region detection | NPU (NMS on CPU) | ~1ms + NMS |
| PaddleOCR Detection | Text region detection | CPU (fallback) | ~29ms |
| PaddleOCR Classification | Text orientation | NPU | ~4.4ms |
| PaddleOCR Recognition | Character recognition | NPU | ~5.4ms |

## Modified Files

### 1. `frigate/detectors/detection_runners.py`

Core NPU support logic in `OpenVINOModelRunner`:

- **Removed `paddleocr` from NPU exclusion list** (`is_model_npu_supported`)
- **Added `device` param to `is_complex_model()`** — PaddleOCR on NPU marked non-complex (no `reset_state` needed)
- **Added `_needs_npu_static_reshape()`** — Identifies models that need special handling for NPU:
  - PaddleOCR: dynamic input shapes need static reshape
  - YOLOv9: contains NMS ops that NPU cannot execute
- **Added `_get_npu_static_shape()`** — Determines correct static shape per model:
  - Recognition `[?,3,48,?]` → `[1,3,48,320]`
  - Classification `[?,3,?,?]` (rank-2 output) → `[1,3,48,192]`
  - Detection `[?,3,?,?]` (spatial output) → `None` (skip reshape)
  - YOLOv9 → `None` (static input, NMS handled separately)
- **Added `_strip_nms_for_npu()`** — Removes NonMaxSuppression/TopK ops from YOLOv9 graph for NPU compilation; stores NMS parameters for CPU post-processing
- **Added `_cpu_nms_postprocess()`** — Runs NMS on CPU after NPU inference, outputs in original YOLOv9 format `[100, 7]`
- **Added `_run_batched_single_input()`** — Handles batch>1 inference on NPU (compiled with batch=1)
- **Added graceful CPU/GPU fallback** — If NPU compile fails, automatically falls back to CPU with a warning
- **NMS post-processing in `run()`** — When `_npu_nms_params` is set, applies CPU NMS to raw NPU outputs
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

### YOLOv9 License Plate Detector — NMS Stripped for NPU

The YOLOv9 model contains `NonMaxSuppression` and `TopK` ops that Intel NPU cannot execute (causes `ZE_RESULT_ERROR_DEVICE_LOST` — device hang). The model compiles successfully on NPU but crashes during inference.

**Solution**: `_strip_nms_for_npu()` automatically detects and removes the NMS subgraph before NPU compilation:
- Pre-NMS tensors (boxes `[1,1344,4]` + scores `[1,1,1344]`) are output from NPU (~1ms)
- `_cpu_nms_postprocess()` runs NMS on CPU (microseconds for 1344 candidates)
- Output format is preserved: `[100, 7]` matching the original `[batch_idx, x1, y1, x2, y2, class_id, score]`

**NMS Parameters** (extracted from model constants at graph-strip time):
- `iou_threshold`: 0.45
- `score_threshold`: 0.001
- `max_output_boxes`: 100

**Performance**: NPU inference ~1ms vs GPU ~3ms (3x speedup), with negligible CPU NMS overhead.

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

## Deployment on Android Devices (Verified Working)

This section documents the verified deployment process on Android devices with Intel NPU.

### Critical NPU Device Path Fix

**Problem**: On Android systems, the NPU device is located at `/dev/accel0`, but OpenVINO's Intel NPU driver expects the standard Linux path `/dev/accel/accel0`. Without the correct path, OpenVINO will fail to detect the NPU with the error:

```
Unrecognized device ID! 0x0x0
```

**Solution**: Mount the device to BOTH paths when starting the container:

```bash
docker run -d \
  --name frigate \
  --privileged \
  --shm-size=256m \
  --device /dev/accel0:/dev/accel0 \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /data/frigate/config:/config \
  -v /data/frigate/media:/media/frigate \
  -p 5000:5000 \
  -p 8554:8554 \
  --restart=unless-stopped \
  frigate:LPR_OV202602_v2
```

**Verification**: After starting the container, verify NPU is detected:

```bash
docker exec frigate python3 -c 'import openvino as ov; print(ov.Core().available_devices)'
# Expected output: ['CPU', 'GPU', 'NPU']
```

### Complete Deployment Steps (via ADB)

#### Step 1: Build and Save Image

On development machine:

```bash
# Build the image (see "Build Instructions" section above)
docker build -t frigate:LPR_OV202602_v2 -f docker/main/Dockerfile .

# Save image to tar.gz (will be ~2.1GB compressed)
docker save frigate:LPR_OV202602_v2 | gzip > /tmp/frigate_lpr_ov202602_v2.tar.gz
```

#### Step 2: Transfer Image to Device

```bash
# Connect device via ADB
adb devices

# Push image (takes ~2-3 minutes depending on USB speed)
adb push /tmp/frigate_lpr_ov202602_v2.tar.gz /data/frigate_lpr.tar.gz

# Load image on device
adb shell "docker load -i /data/frigate_lpr.tar.gz"

# Verify image loaded
adb shell "docker images | grep frigate"
```

#### Step 3: Pre-download LPR Models (Recommended)

If the device cannot access GitHub (DNS issues, proxy, firewall), pre-download models on development machine and push to device:

```bash
# On development machine, download models
mkdir -p /tmp/lpr_models/paddleocr-onnx /tmp/lpr_models/yolov9

cd /tmp/lpr_models/paddleocr-onnx
wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v5/detection_v5-small.onnx
wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx
wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/recognition_v4.onnx
wget https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/ppocr_keys_v1.txt

cd /tmp/lpr_models/yolov9
wget https://github.com/hawkeye217/yolov9-license-plates/raw/refs/heads/master/models/yolov9-256-license-plates.onnx

# Push models to device
adb shell "mkdir -p /data/frigate/config/model_cache/paddleocr-onnx"
adb shell "mkdir -p /data/frigate/config/model_cache/yolov9_license_plate"
adb push /tmp/lpr_models/paddleocr-onnx/* /data/frigate/config/model_cache/paddleocr-onnx/
adb push /tmp/lpr_models/yolov9/* /data/frigate/config/model_cache/yolov9_license_plate/
```

#### Step 4: Create Configuration

Create minimal config on device:

```bash
adb shell "cat > /data/frigate/config/config.yml << 'EOF'
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
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://your_camera_ip:554/stream
          roles:
            - detect
    detect:
      enabled: true
      width: 1920
      height: 1080
      fps: 5

logger:
  default: info
  logs:
    frigate.detectors: info
    frigate.embeddings: debug
EOF
"
```

#### Step 5: Start Container with Correct Device Paths

```bash
adb shell "docker run -d \
  --name frigate \
  --privileged \
  --shm-size=256m \
  --device /dev/accel0:/dev/accel0 \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /data/frigate/config:/config \
  -v /data/frigate/media:/media/frigate \
  -v /data/frigate/clips:/media/frigate/clips \
  -v /data/frigate/recordings:/media/frigate/recordings \
  -p 5000:5000 \
  -p 8554:8554 \
  -p 8555:8555/tcp \
  -p 8555:8555/udp \
  --restart=unless-stopped \
  frigate:LPR_OV202602_v2"
```

#### Step 6: Verify NPU Detection and Model Loading

```bash
# Wait for startup (20-30 seconds)
sleep 30

# Check NPU detection
adb shell "docker exec frigate python3 -c 'import openvino as ov; print(\"Devices:\", ov.Core().available_devices)'"
# Expected: Devices: ['CPU', 'GPU', 'NPU']

# Check logs for model loading
adb shell "docker logs frigate 2>&1" | grep -E "NPU|Reshaping|Loading"

# Check detector status via API
adb shell "docker exec frigate curl -s http://localhost:5000/api/stats" | grep -A 5 detectors

# Expected to see:
# "detectors": {
#     "openvino": {
#         "inference_speed": 10.0,
#         "detection_start": 0.0,
#         ...
```

#### Step 7: Monitor LPR Performance

```bash
# View embeddings stats (includes plate_recognition metrics)
adb shell "docker exec frigate curl -s http://localhost:5000/api/stats" | grep -A 3 embeddings

# Check LPR logs
adb shell "docker logs frigate 2>&1" | grep -i "plate\|lpr"
```

### Troubleshooting

#### NPU Not Detected

**Symptom**: `ov.Core().available_devices` returns `['CPU', 'GPU']` without `NPU`.

**Root Cause**: Device path mismatch between Android (`/dev/accel0`) and OpenVINO expectation (`/dev/accel/accel0`).

**Fix**: Ensure BOTH device paths are mounted:
```bash
--device /dev/accel0:/dev/accel0 \
--device /dev/accel0:/dev/accel/accel0
```

**Verification**:
```bash
# Check device exists in container at both paths
adb shell "docker exec frigate ls -la /dev/accel0 /dev/accel/accel0"
# Both should show: crw-rw-rw-. 1 root 1003 261, 0 ...

# Check device vendor/device ID
adb shell "docker exec frigate cat /sys/class/accel/accel0/device/vendor"  # Should show: 0x8086
adb shell "docker exec frigate cat /sys/class/accel/accel0/device/device"  # Should show: 0xb03e
```

#### Model Download Failures

**Symptom**: `FileNotFoundError: OpenVINO model file /config/model_cache/paddleocr-onnx/detection_v5-small.onnx not found`

**Root Cause**: Device cannot reach GitHub (DNS resolution failure, proxy, firewall).

**Fix**: Pre-download and push models (see Step 3 above).

#### NPU Compilation Errors

**Symptom**: Logs show `Compilation failed. vclAllocatedExecutableCreate3 result: 0x78000004`

**Possible Causes**:
1. **Level Zero library version mismatch**: Container's `libze_intel_npu.so` version doesn't match device firmware
2. **Model shape incompatibility**: Some models cannot be reshaped for NPU

**Fix**: The code includes automatic CPU fallback. Models that fail NPU compilation will run on CPU with a warning.

### Verified Hardware

- **Device**: Intel Core Ultra X7 358H (with Intel AI Boost NPU)
- **NPU Device ID**: `0x8086:0xb03e`
- **OpenVINO**: 2026.2.0
- **Platform**: Android-based system running Docker

### Performance Metrics (Verified)

| Component | Device | Inference Time |
|-----------|--------|----------------|
| Main Detector (SSD MobileNet V2) | NPU | ~10ms |
| YOLOv9 License Plate Detection | NPU (NMS on CPU) | ~1ms + ~0.1ms NMS |
| PaddleOCR Detection | CPU (fallback) | ~29ms |
| PaddleOCR Classification | NPU | ~4.4ms |
| PaddleOCR Recognition | NPU | ~5.4ms |

**Total LPR Pipeline**: ~8ms plate recognition speed (verified with `plate_recognition_speed: 8.12`), ~4.2 plates/sec throughput.

**End-to-End Verified Result**: `沪A·HG0162` recognized with confidence 1.000 on full NPU configuration.

## Video Stream Setup with go2rtc (Verified Working)

For testing LPR functionality without a live camera, you can use go2rtc to loop a video file and provide an RTSP stream.

### Network Architecture: Host Network Mode (Recommended)

**Problem**: When containers run on separate Docker bridge networks, Frigate cannot reliably connect to go2rtc's RTSP stream due to network isolation issues.

**Solution**: Use **host network mode** for both containers. This eliminates network isolation and allows direct localhost communication.

### Step 1: Prepare Video File

```bash
# Push test video to device
adb push /path/to/test_video.mp4 /data/videos/test.mp4

# Verify file exists
adb shell "ls -lh /data/videos/test.mp4"
```

### Step 2: Configure and Start go2rtc Container

Create go2rtc configuration:

```bash
# Create config directory
adb shell "mkdir -p /data/go2rtc"

# Create configuration file
cat > /tmp/go2rtc.yaml << 'EOF'
streams:
  lpr_test:
    # Loop video file with hardware acceleration
    - ffmpeg:/videos/test.mp4#video=copy#audio=copy#rawArgs=-stream_loop -1 -re

api:
  listen: ":1984"

rtsp:
  listen: ":8556"

webrtc:
  listen: ":8557"

log:
  level: info
EOF

# Push config to device
adb push /tmp/go2rtc.yaml /data/go2rtc/go2rtc.yaml
```

Start go2rtc container with host network:

```bash
adb shell "docker run -d \
  --name go2rtc \
  --restart=unless-stopped \
  --network host \
  --device /dev/dri:/dev/dri \
  -v /data/go2rtc:/config \
  -v /data/videos:/videos \
  alexxit/go2rtc:latest"
```

**Key Configuration Notes**:
- `--network host`: Uses host network namespace for direct localhost access
- `--device /dev/dri:/dev/dri`: Provides GPU access (not used with `copy` mode, but needed if transcoding)
- `#video=copy#audio=copy`: **Passthrough mode** — H.264 bitstream is forwarded directly without any decode/encode. go2rtc's ffmpeg neither uses CPU nor GPU for processing. This is the most efficient approach when the source video is already H.264.
- `#rawArgs=-stream_loop -1 -re`: Loops video infinitely at original frame rate
- Video decoding happens on the **Frigate side**, where ffmpeg uses VAAPI GPU hardware acceleration (`-hwaccel vaapi`) for decoding the H.264 stream into raw frames for detection.

### Step 3: Verify go2rtc Stream

```bash
# Check stream status via API
adb shell "curl -s http://localhost:1984/api/streams"

# Expected output (formatted):
# {
#   "lpr_test": {
#     "producers": [{
#       "format_name": "rtsp",
#       "source": "exec:ffmpeg ... /videos/test.mp4 ...",
#       "medias": ["video, recvonly, H264", "audio, recvonly, PCML/48000"]
#     }],
#     "consumers": [...]
#   }
# }
```

### Step 4: Configure Frigate to Use go2rtc Stream

Create Frigate configuration:

```bash
cat > /tmp/frigate_config.yml << 'EOF'
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
  lpr_camera:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8556/lpr_test
          roles:
            - detect
    detect:
      enabled: true
      width: 1920
      height: 1080
      fps: 5
    motion:
      enabled: false  # Disable motion detection to force detection on every frame
    objects:
      track:
        - car
        - person
        - license_plate

logger:
  default: info
  logs:
    frigate.detectors: info
    frigate.embeddings: debug
EOF

# Push config to device
adb push /tmp/frigate_config.yml /data/frigate/config/config.yml
```

**Configuration Notes**:
- `path: rtsp://127.0.0.1:8556/lpr_test`: Uses localhost because of host network mode
- `motion.enabled: false`: Forces detection on every frame (useful for testing)
- `objects.track`: Must include `car` or vehicles for LPR to trigger on detected vehicles

### Step 5: Start Frigate Container with Host Network

```bash
adb shell "docker run -d \
  --name frigate \
  --privileged \
  --shm-size=256m \
  --network host \
  --device /dev/accel0:/dev/accel0 \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /data/frigate/config:/config \
  -v /data/frigate/media:/media/frigate \
  -v /data/frigate/clips:/media/frigate/clips \
  -v /data/frigate/recordings:/media/frigate/recordings \
  -v /data/videos:/data/videos \
  --restart=unless-stopped \
  frigate:LPR_OV202602_v2"
```

**Key Configuration Notes**:
- `--network host`: Shares host network with go2rtc for localhost RTSP access
- Both NPU device paths (`/dev/accel0` and `/dev/accel/accel0`) required
- No port mappings needed with host network mode

### Step 6: Verify Video Stream Connection

Wait 30-40 seconds for container startup, then check connection:

```bash
# Check camera stats
adb shell "docker exec frigate curl -s http://localhost:5000/api/stats" | grep -A 15 lpr_camera

# Expected output:
# "lpr_camera": {
#     "camera_fps": 5.1,              # ✓ Receiving video frames
#     "process_fps": 5.1,             # ✓ Processing frames
#     "detection_fps": 5.0,           # ✓ Running detection
#     "connection_quality": "excellent", # ✓ Stable connection
#     "reconnects_last_hour": 0,      # ✓ No reconnects
#     ...
# }
```

**Success Indicators**:
- ✅ `camera_fps > 0`: Video stream connected
- ✅ `connection_quality: "excellent"`: Stable RTSP connection
- ✅ `reconnects_last_hour: 0`: No connection issues
- ✅ `detection_fps > 0`: NPU detection running

### Step 7: Monitor LPR Detection

```bash
# View embeddings/LPR stats
adb shell "docker exec frigate curl -s http://localhost:5000/api/stats" | grep -A 5 embeddings

# Expected output:
# "embeddings": {
#     "plate_recognition_speed": 15.2,  # NPU inference time
#     "plate_recognition": 0.5          # Plates recognized per second
# }

# Check detected events
adb shell "docker exec frigate curl -s http://localhost:5000/api/events"
```

### Troubleshooting Video Streams

#### Stream 404 Not Found

**Symptom**: `[in#0] method DESCRIBE failed: 404 (Not Found)`

**Causes & Fixes**:

1. **go2rtc not started**: 
   ```bash
   adb shell "docker ps | grep go2rtc"  # Should show "Up X minutes"
   ```

2. **Stream configuration error**:
   ```bash
   adb shell "curl -s http://localhost:1984/api/streams"
   # Should show "lpr_test" with producers
   ```

3. **Incorrect stream name**: Verify Frigate config matches go2rtc stream name (`lpr_test`)

#### Connection Refused / Timeout

**Symptom**: `Connection to tcp://127.0.0.1:8556 failed: Connection refused`

**Cause**: Not using host network mode, or go2rtc not listening.

**Fix**: Ensure both containers use `--network host`

**Verify**:
```bash
# Check go2rtc is listening on 8556
adb shell "netstat -tlnp | grep 8556"
# Should show: tcp6 ... :::8556 ... LISTEN 9026/go2rtc
```

#### Zero FPS Despite Connection

**Symptom**: `camera_fps: 0` or `detection_fps: 0`

**Causes**:

1. **No objects tracked**: Add objects to config:
   ```yaml
   objects:
     track:
       - car
       - license_plate
   ```

2. **Motion detection blocking**: Disable motion for testing:
   ```yaml
   motion:
     enabled: false
   ```

3. **Video file not accessible**:
   ```bash
   adb shell "docker exec go2rtc ls -lh /videos/test.mp4"
   ```

### Alternative: Using Frigate's Built-in go2rtc

Instead of a separate go2rtc container, Frigate includes an embedded go2rtc instance:

```yaml
# In Frigate config.yml
go2rtc:
  streams:
    lpr_test:
      - ffmpeg:/data/videos/test.mp4#video=copy#audio=copy#rawArgs=-stream_loop -1 -re

cameras:
  lpr_camera:
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8554/lpr_test  # Port 8554 for internal go2rtc
          roles:
            - detect
```

**Note**: Must mount video file into Frigate container:
```bash
-v /data/videos:/data/videos
```

### Performance Considerations

**Stream Encoding (go2rtc side)**:
- `#video=copy` = **passthrough** (no CPU/GPU usage on go2rtc side, zero processing overhead)
- `#hardware=vaapi` = GPU transcoding (only needed if source is not H.264)
- `#video=h264` = CPU transcoding (fallback, high CPU usage)
- **Recommended**: Always use `copy` when source is already H.264

**Video Decoding (Frigate side)**:
- Frigate's ffmpeg automatically uses VAAPI GPU hardware acceleration for decoding
- Verified by: `-hwaccel vaapi -hwaccel_device /dev/dri/renderD128` in ffmpeg cmdline
- GPU decode uses ~4% of iGPU (shown in `gpu_usages.dec` stats)
- Detection frames are decoded on GPU then downloaded to CPU memory for inference

**Network Bandwidth**:
- Host network mode = zero network overhead
- Bridge network mode = copy overhead through Docker bridge

**Video File Format**:
- **Recommended**: H.264/AAC MP4 (native RTSP passthrough, no transcoding)
- **Avoid**: VP9/AV1 (requires CPU/GPU transcoding on go2rtc side)

### Verified Stream Configuration

**Tested Setup**:
- Video file: 1920x1080 H.264 @ 30 FPS, 280MB
- go2rtc: Loop playback with passthrough (`-c:v copy`), zero CPU/GPU usage
- Frigate: VAAPI GPU decode → 5 FPS detection on NPU
- Connection: Host network, localhost RTSP
- Result: Stable 5.1 FPS, excellent quality, 0 reconnects

## End-to-End LPR → OpenClaw Notification Pipeline

### Pipeline Architecture

```
┌─────────────┐    RTSP     ┌─────────────┐   events API   ┌──────────────┐
│   go2rtc    │───────────▶│   Frigate    │◀──── poll ─────│   OpenClaw   │
│ (video loop)│  H.264 copy │  (LPR GPU)  │                │  (listener)  │
└─────────────┘             └──────┬──────┘                └──────┬───────┘
                                   │                              │
                                   │ MQTT (state)                 │ openclaw agent
                                   ▼                              ▼
                            ┌─────────────┐              ┌───────────────┐
                            │  Mosquitto  │              │ vision-care   │
                            │   broker    │              │    agent      │
                            └─────────────┘              └───────────────┘
```

**Data flow (per frame):**

```
Video frame (1920x1080)
  │
  ▼
Motion Detection → triggers LPR pipeline
  │
  ▼
YOLOv9 License Plate Detection (GPU, ~9ms)
  │  detects plate bounding box
  ▼
PaddleOCR Detection → Classification → Recognition (GPU/CPU)
  │  extracts plate characters
  ▼
Cluster & Match → "沪A·HG0162" (conf: 0.952)
  │
  ▼
Event stored in Frigate DB (/api/events)
  │
  ▼ (polled every 5s by OpenClaw listener)
frigate-notify.sh → downloads snapshot → openclaw agent --message "车牌识别摄像头检测到车牌 沪A·HG0162"
  │
  ▼
OpenClaw vision-care agent processes notification
```

### Docker Deployment (Complete)

Five containers run on the Android device with `--network host`:

| Container | Image | Purpose | Port |
|-----------|-------|---------|------|
| `go2rtc` | `alexxit/go2rtc:latest` | Video loop → RTSP stream | 8556 (RTSP), 1984 (API) |
| `frigate` | `frigate:LPR_OV202602_v2` | Object detection + LPR | 5000 (API) |
| `mosquitto` | `eclipse-mosquitto:2` | MQTT broker | 1883 |
| `openclaw-basic-chat` | `openclaw-basic-chat:*` | AI agent platform | 18789 (gateway) |
| `ovms-qwen36` | `openvino/model_server:2026.2-gpu` | LLM inference | — |

#### Deploy Mosquitto Broker

```bash
# Create config
adb shell "mkdir -p /data/mosquitto && echo -e 'listener 1883\nallow_anonymous true' > /data/mosquitto/mosquitto.conf"

# Start container
adb shell "docker run -d \
  --name mosquitto \
  --network host \
  --restart=unless-stopped \
  -v /data/mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf \
  eclipse-mosquitto:2"

# Verify
adb shell "docker logs mosquitto 2>&1 | grep 'running'"
# Expected: mosquitto version 2.1.2 running
```

#### Deploy Frigate with MQTT + GPU

```bash
adb shell "docker run -d \
  --name frigate \
  --privileged \
  --shm-size=256m \
  --network host \
  --device /dev/accel0:/dev/accel0 \
  --device /dev/accel0:/dev/accel/accel0 \
  --device /dev/dri:/dev/dri \
  -v /data/frigate/config:/config \
  -v /data/frigate/media:/media/frigate \
  -v /data/videos:/data/videos \
  --restart=unless-stopped \
  frigate:LPR_OV202602_v2"
```

#### Frigate config.yml (GPU mode, with MQTT)

```yaml
mqtt:
  enabled: true
  host: 127.0.0.1
  port: 1883
  topic_prefix: frigate

detectors:
  openvino:
    type: openvino
    device: GPU

model:
  width: 300
  height: 300
  input_tensor: nhwc

lpr:
  enabled: true
  device: GPU
  detection_threshold: 0.3
  recognition_threshold: 0.7
  min_plate_length: 5
  min_area: 300
  model_size: small
  debug_save_plates: true
  known_plates:
    Test_Car:
      - 沪AHG0162

cameras:
  lpr_camera:
    enabled: true
    type: lpr
    lpr:
      enabled: true
      enhancement: 3
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8556/lpr_test
          roles:
            - detect
    detect:
      enabled: false
      width: 1920
      height: 1080
      fps: 5
    objects:
      track: []
    motion:
      enabled: true
      threshold: 10
      contour_area: 5
      improve_contrast: true

logger:
  default: info
  logs:
    frigate.detectors: info
    frigate.embeddings: debug
    frigate.data_processing.common.license_plate: debug
version: 0.18-0
```

### Known Issues & Fixes

#### NPU `ZE_RESULT_ERROR_DEVICE_LOST` / `ZE_RESULT_ERROR_UNKNOWN`

**Symptom**: Every frame reports:
```
Error running YOLOv9 license plate detection model: ...
L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe
```

YOLOv9 compiles to NPU successfully but inference hangs on first frame. All subsequent frames also fail. PaddleOCR classification/recognition models may still work on NPU.

**Root Cause**: NPU hardware enters an unrecoverable state after extended use (observed after ~7 days uptime). Docker restart does NOT reset the NPU — the device firmware needs a full reset.

**Workaround**: Switch `lpr.device` and `detectors.device` from `NPU` to `GPU`:
```yaml
detectors:
  openvino:
    type: openvino
    device: GPU    # was: NPU

lpr:
  enabled: true
  device: GPU      # was: NPU
```

**Performance impact**: YOLOv9 inference goes from ~1ms (NPU) to ~9ms (GPU). Total LPR pipeline still fast enough for real-time processing.

**Permanent fix**: Reboot the Android device (`adb reboot`) to reset NPU hardware state, then switch back to `device: NPU`.

#### `motion.enabled: false` + `detect.enabled: true` — Config Validation Error

**Symptom**: Frigate enters safe mode with:
```
Camera lpr_camera has motion detection disabled and object detection enabled
but object detection requires motion detection.
```

**Fix**: Use `type: lpr` dedicated camera mode (does not require `detect.enabled: true`) OR enable both motion and detect:
```yaml
cameras:
  lpr_camera:
    type: lpr           # dedicated LPR mode
    detect:
      enabled: false    # not needed for type: lpr
    motion:
      enabled: true     # must be true
```

#### Dedicated LPR Camera Does Not Publish to `frigate/events` MQTT Topic

**Symptom**: Plates are recognized and stored in the database (`/api/events`), but no messages appear on `frigate/events` MQTT topic.

**Root Cause**: The `create_lpr_event()` path in `object_processing.py` sends events to the internal event maintainer (database) but does NOT call `dispatcher.publish("events", ...)` which is the MQTT publisher. Only `TrackedObject` updates (from `detect.enabled: true` cameras) publish to MQTT.

**Workaround**: Use API polling instead of MQTT subscription for dedicated LPR events. The OpenClaw listener polls `GET /api/events?camera=lpr_camera&limit=5&has_plate=1` every 5 seconds.

### OpenClaw Notification Scripts

#### Listener: `frigate-mqtt-listener.py`

Located at `/opt/openclaw-env/scripts/frigate-mqtt-listener.py` inside the `openclaw-basic-chat` container.

Polls Frigate's events API for new plate recognitions and triggers notifications:

```python
#!/usr/bin/env python3
"""Frigate LPR event listener - polls events API and triggers OpenClaw agent."""
import json
import os
import subprocess
import time
import urllib.request

FRIGATE_URL = os.environ.get("FRIGATE_URL", "http://127.0.0.1:5000")
CAMERA_FILTER = os.environ.get("FRIGATE_CAMERA_FILTER", "lpr_camera")
EVENT_COOLDOWN = int(os.environ.get("FRIGATE_EVENT_COOLDOWN", "30"))
POLL_INTERVAL = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTIFY_SCRIPT = os.path.join(SCRIPT_DIR, "frigate-notify.sh")

seen_events = set()
last_notify_at = 0


def poll_events():
    global last_notify_at

    url = f"{FRIGATE_URL}/api/events?camera={CAMERA_FILTER}&limit=5&has_plate=1"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            events = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[frigate-lpr] poll error: {e}", flush=True)
        return

    for event in events:
        event_id = event.get("id", "")
        if event_id in seen_events:
            continue

        plate = event.get("data", {}).get("recognized_license_plate", "")
        camera = event.get("camera", "")

        if not plate or not camera:
            seen_events.add(event_id)
            continue

        now = int(time.time())
        if (now - last_notify_at) < EVENT_COOLDOWN:
            seen_events.add(event_id)
            continue

        seen_events.add(event_id)
        last_notify_at = now

        print(f"[frigate-lpr] plate={plate} event={event_id} camera={camera}", flush=True)
        subprocess.Popen([NOTIFY_SCRIPT, camera, event_id, plate])


def main():
    print(f"[frigate-lpr] polling {FRIGATE_URL}/api/events camera={CAMERA_FILTER} "
          f"cooldown={EVENT_COOLDOWN}s interval={POLL_INTERVAL}s", flush=True)
    time.sleep(10)

    while True:
        poll_events()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
```

**Environment variables** (from OpenClaw container):
- `FRIGATE_URL=http://127.0.0.1:5000`
- `FRIGATE_CAMERA_FILTER=lpr_camera`
- `FRIGATE_EVENT_COOLDOWN=30`

#### Notify: `frigate-notify.sh`

Located at `/opt/openclaw-env/scripts/frigate-notify.sh`. Downloads a snapshot from Frigate and calls `openclaw agent` to notify the vision-care agent:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <camera> <event-id> [plate]" >&2
  exit 64
fi

CAMERA="$1"
EVENT_ID="$2"
PLATE="${3:-}"
FRIGATE_URL="${FRIGATE_URL:-http://127.0.0.1:5000}"
OPENCLAW_AGENT="${VISION_CARE_AGENT_ID:-vision-care}"
OPENCLAW_TMP="/tmp/openclaw-1000"
GALLERY_DIR="/home/node/.openclaw/workspace-vision-care/Gallery/frigate"
STAMP="$(date +%Y%m%d-%H%M%S)"
SNAPSHOT="$GALLERY_DIR/${CAMERA}-${EVENT_ID}-${STAMP}.jpg"
ALERT_STAMP="$(date +%m%d_%H%M)"
case "$CAMERA" in
  cam1)
    ALERT_SNAPSHOT="/tmp/openclaw-1000/frigate_cam1_${ALERT_STAMP}_alert.jpg"
    ;;
  *)
    ALERT_SNAPSHOT="$OPENCLAW_TMP/frigate_${CAMERA}_alert.jpg"
    ;;
esac

mkdir -p "$GALLERY_DIR" "$OPENCLAW_TMP"

# Try event snapshot first, fall back to latest frame
if ! curl -fsSL "$FRIGATE_URL/api/events/$EVENT_ID/snapshot.jpg" -o "$SNAPSHOT"; then
  if ! curl -fsSL "$FRIGATE_URL/api/$CAMERA/latest.jpg" -o "$SNAPSHOT"; then
    echo "[frigate-notify] Could not get snapshot, proceeding without image"
    SNAPSHOT=""
  fi
fi

if [ -n "$SNAPSHOT" ]; then
  if [ "$CAMERA" = "cam1" ]; then
    rm -f "$OPENCLAW_TMP"/frigate_cam1*.jpg
  fi
  cp "$SNAPSHOT" "$ALERT_SNAPSHOT"
fi

case "$CAMERA" in
  cam1)
    LOCATION="行车记录仪"
    ;;
  cam2)
    LOCATION="客厅"
    ;;
  aqara)
    LOCATION="餐厅"
    ;;
  lpr_camera)
    LOCATION="车牌识别摄像头"
    ;;
  *)
    LOCATION="$CAMERA"
    ;;
esac

if [ -n "$PLATE" ] && [ -n "$SNAPSHOT" ]; then
  MSG="${LOCATION}检测到车牌 ${PLATE}。截图路径：${ALERT_SNAPSHOT}。必须原样使用消息里的截图路径调用 image 工具；不要改写路径；不要使用 find 搜索 workspace；不要重新猜测或寻找其他截图。"
elif [ -n "$PLATE" ]; then
  MSG="${LOCATION}检测到车牌 ${PLATE}。"
else
  MSG="Frigate detected motion from ${LOCATION}."
fi

openclaw agent \
  --agent "$OPENCLAW_AGENT" \
  --session-key "agent:${OPENCLAW_AGENT}:traffic-light-alerts" \
  --message "$MSG"
```

#### Startup Hook: `50-frigate-mqtt-listener.sh`

Located at `/opt/openclaw-env/hooks/50-frigate-mqtt-listener.sh`. Auto-starts the listener when OpenClaw container boots:

```bash
#!/bin/sh

LOG_FILE="/tmp/frigate-mqtt-listener.log"

if pgrep -f "frigate-mqtt-listener.py" >/dev/null 2>&1; then
    echo "[frigate-mqtt] listener already running"
    return 0 2>/dev/null || exit 0
fi

export FRIGATE_CAMERA_FILTER=lpr_camera

nohup python3 -u /opt/openclaw-env/scripts/frigate-mqtt-listener.py >"$LOG_FILE" 2>&1 &
echo "[frigate-mqtt] listener started, log=$LOG_FILE"
```

### Verification

#### Check Pipeline Health

```bash
# 1. Verify all containers are running
adb shell "docker ps --format '{{.Names}}\t{{.Status}}'"

# 2. Verify Mosquitto has Frigate connected
adb shell "docker logs mosquitto 2>&1 | grep 'frigate'"
# Expected: New client connected ... as frigate

# 3. Verify Frigate is detecting plates
adb shell "docker logs frigate --since 30s 2>&1 | grep 'Found license plate'"

# 4. Verify events are in database
adb shell "curl -s 'http://127.0.0.1:5000/api/events?limit=3'" | python3 -m json.tool

# 5. Verify OpenClaw listener is polling
adb shell "docker exec openclaw-basic-chat cat /tmp/frigate-mqtt-listener.log"
# Expected: [frigate-lpr] plate=沪A·HG0162 event=... camera=lpr_camera

# 6. Verify snapshot exists
adb shell "docker exec openclaw-basic-chat ls -la /tmp/openclaw-1000/frigate_lpr_camera_alert.jpg"
```

#### Manual Test (Simulate Event)

```bash
# Publish a fake event to verify MQTT → OpenClaw flow
adb shell "docker exec mosquitto mosquitto_pub -t 'frigate/events' \
  -m '{\"type\":\"update\",\"after\":{\"id\":\"manual-test\",\"camera\":\"lpr_camera\",\"recognized_license_plate\":\"测试ABC123\"}}'"

# Check listener received it (only works if listener uses MQTT mode)
adb shell "docker exec openclaw-basic-chat cat /tmp/frigate-mqtt-listener.log"
```

### Performance (GPU Mode, Verified)

| Component | Device | Inference Time |
|-----------|--------|----------------|
| Main Detector (SSD MobileNet V2) | GPU | ~15ms |
| YOLOv9 License Plate Detection | GPU | ~9ms |
| PaddleOCR Detection | CPU (fallback) | ~29ms |
| PaddleOCR Classification | GPU | ~8ms |
| PaddleOCR Recognition | GPU | ~10ms |
| **Notification latency** | API poll | **≤5s** (poll interval) |

**Verified plates recognized**: `沪A·HG0162`, `沪N·Z1079`, `沪A·A50521`, `豫A625L0`, `皖E·4697`, `沪ABV5385`

## Repository Change Audit (2026-07-03)

Audit of the local repo to confirm every source change required for NPU LPR is
committed, and record of the `LPR_OV20260703_v1` rebuild/redeploy.

### Root Cause of the "NPU doesn't work" Regression

The device was running image `frigate:LPR_OV202602_v2`, which shipped an **older**
`frigate/detectors/detection_runners.py` whose `_needs_npu_static_reshape()`
returned `True` for `paddleocr` **only** — not `yolov9_license_plate`. As a
result, YOLOv9 fell through to the `else` branch and the **full NMS+TopK ONNX was
compiled directly onto the NPU**. That graph compiles successfully but hangs at
inference on every frame:

```
frigate.data_processing.common.license_plate.mixin WARNING : Error running YOLOv9
  license plate detection model: Exception from src/inference/src/cpp/infer_request.cpp:224:
L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe
```

This is **not** a firmware/reboot-recoverable NPU fault. It was reproduced from a
fresh reboot (device uptime 8 min, NPU present as `0x8086:0xb03e`) and after
container restarts. Isolation tests inside the same container on the same NPU:

| Graph run on NPU (30 iters, real data, `NPU_TURBO=YES`) | Result |
|---------------------------------------------------------|--------|
| Full ONNX with `NonMaxSuppression`+`TopK` baked in       | compiles OK, **inference fails every frame** (`ZE_RESULT_ERROR`) |
| NMS-stripped graph (`boxes[1,1344,4]` + `scores[1,1,1344]`) | **0 failures, ~0.65ms warm** |
| paddleocr cls + rec + yolov9(no-nms) all on NPU, one `Core` | 0 failures |

Conclusion: the NPU hardware and the stripped graph are fine; the only problem was
the stale `detection_runners.py`. Re-exporting the model with `nms=False` was
**not** required — `_strip_nms_for_npu()` already produces the equivalent no-NMS
graph at runtime.

### Source Change Verification (all committed, clean working tree)

| File | Required change | Status |
|------|-----------------|--------|
| `frigate/detectors/detection_runners.py` | `_strip_nms_for_npu` (l.531), `_cpu_nms_postprocess` (l.604), `_get_npu_static_shape` (l.462), `_run_batched_single_input` (l.775), `_needs_npu_static_reshape` incl. `yolov9_license_plate` (l.442), `PADDLEOCR_NPU_RECOGNITION_WIDTH` (l.268) | ✅ present |
| `docker/main/Dockerfile` | `npu-libs` build stage (l.64) + `COPY --from=npu-libs /npu-rootfs/ /` | ✅ present (compiler libs added 2026-07-03) |
| `docker/main/requirements-wheels.txt` | `openvino == 2026.2.*` (l.45) | ✅ present |
| `docker/main/requirements-ov.txt` | `openvino-dev==2024.6.0` | ✅ present |

### ✅ Resolved: Dockerfile `npu-libs` Stage Now Includes Compiler Libraries

Previously the main `docker/main/Dockerfile` `npu-libs` stage copied only the
driver/loader libraries (`libnpu_driver_compiler.so`, `libze_intel_npu.so.1.32.1`,
`libze_loader.so.1.27.0`) and omitted the two NPU **compiler** libraries that a
working NPU runtime requires:

- `libopenvino_intel_npu_compiler.so` (~117 MB)
- `libopenvino_intel_npu_compiler_loader.so`

These are **not** shipped in the pip `openvino` wheel (only the `_plugin.so` is),
so a fresh **full** build straight from `docker/main/Dockerfile` used to be missing
them and NPU would fail to compile any model. As of 2026-07-03 the `npu-libs` stage
copies them into `/usr/local/lib/python3.11/dist-packages/openvino/libs/`, so full
builds are now self-sufficient (no `Dockerfile.patch` step required for NPU libs):

```dockerfile
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
```

### Rebuild & Redeploy Record — `frigate:LPR_OV20260703_v1`

Because only `detection_runners.py` changed, the new image was built on top of the
existing NPU-lib-complete base (`LPR_OV202602_v2`) rather than a slow full build:

```dockerfile
# Dockerfile.lpr_npu
FROM frigate:LPR_OV202602_v2
COPY frigate/detectors/detection_runners.py /opt/frigate/frigate/detectors/detection_runners.py
```

```bash
# build
DOCKER_BUILDKIT=1 docker build -t frigate:LPR_OV20260703_v1 -f Dockerfile.lpr_npu .
# save + push
docker save frigate:LPR_OV20260703_v1 | gzip > frigate_LPR_OV20260703_v1.tar.gz   # ~2.1 GB
adb push frigate_LPR_OV20260703_v1.tar.gz /data/vendor/docker/sunausti/frigate_image/
adb shell "docker load -i /data/vendor/docker/sunausti/frigate_image/frigate_LPR_OV20260703_v1.tar.gz"
# recreate container (same params as before: host net, privileged, both accel paths)
adb shell "docker rm -f frigate"
adb shell "docker run -d --name frigate --privileged --shm-size=256m --network host \
  --device /dev/accel0:/dev/accel0 --device /dev/accel0:/dev/accel/accel0 --device /dev/dri:/dev/dri \
  -v /data/frigate/config:/config -v /data/frigate/media:/media/frigate -v /data/videos:/data/videos \
  --restart=unless-stopped frigate:LPR_OV20260703_v1"
```

### Post-Deploy Verification (NPU, config `device: NPU`)

```
frigate.detectors.detection_runners INFO : Stripped NMS from yolov9_license_plate for NPU (iou=0.450, score=0.0010, max=100)
frigate.detectors.detection_runners INFO : Compiled yolov9_license_plate on NPU with NMS stripped (NMS will run on CPU post-inference)
```

- `ZE_RESULT_ERROR` count after deploy: **0**
- Main detector (SSD): **~10ms** (NPU)
- YOLOv9 plate detection: **~2.9ms** (NPU, NMS on CPU)
- PaddleOCR recognition: **~13ms** (NPU), ~1.5 plates/sec
- End-to-end recognized: **`沪A·HG0162` (conf 1.000)**

The fix is now baked into the image layer, so it survives container
recreation (not just `docker restart`).
