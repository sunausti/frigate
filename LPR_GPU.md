# Frigate LPR on Intel GPU (Verified Working)

This document describes the verified deployment of Frigate's License Plate Recognition (LPR) pipeline on Intel GPU via OpenVINO, using go2rtc to loop a video file as the RTSP source.

## Overview

| Component | Device | Notes |
|-----------|--------|-------|
| Main Detector (SSD MobileNet V2) | GPU | Object detection |
| YOLOv9 License Plate Detection | GPU | ~3ms per frame |
| PaddleOCR Detection | CPU (fallback) | ~30ms, dynamic shapes |
| PaddleOCR Classification | GPU | ~4ms |
| PaddleOCR Recognition | GPU | ~5ms |

**Verified Result**: Successfully recognized `沪A·HG0162` with 100% confidence.

## Architecture

```
┌──────────────────┐       RTSP (port 8556)       ┌──────────────────────┐
│  go2rtc container │ ──────────────────────────── │  Frigate container    │
│  (host network)   │  rtsp://127.0.0.1:8556/     │  (host network)       │
│                    │  lpr_test                    │                       │
│  Loop video file   │                             │  GPU: Detection + LPR │
│  /videos/test.mp4  │                             │  NPU: PaddleOCR (opt) │
└──────────────────┘                               └──────────────────────┘
```

Both containers use **host network mode** for direct localhost RTSP communication.

## Quick Start

### 1. Push Video File to Device

```bash
adb shell "mkdir -p /data/videos"
adb push /path/to/video.mp4 /data/videos/test.mp4
```

### 2. Create go2rtc Configuration

```bash
adb shell "mkdir -p /data/go2rtc"
adb shell 'cat > /data/go2rtc/go2rtc.yaml << "EOF"
streams:
  lpr_test:
    - ffmpeg:/videos/test.mp4#video=copy#audio=copy#rawArgs=-stream_loop -1 -re

api:
  listen: ":1984"

rtsp:
  listen: ":8556"

webrtc:
  listen: ":8557"

log:
  level: info
EOF'
```
visit http://<host>:1984/stream.html?src=lpr_test for web open video

### 3. Start go2rtc Container

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

### 4. Pre-download LPR Models

Device may not have internet access. Download on host and push:

```bash
# On development machine
mkdir -p /tmp/lpr_models/paddleocr-onnx /tmp/lpr_models/yolov9

wget -P /tmp/lpr_models/paddleocr-onnx \
  https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v5/detection_v5-small.onnx \
  https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx \
  https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/recognition_v4.onnx \
  https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/ppocr_keys_v1.txt

wget -P /tmp/lpr_models/yolov9 \
  https://github.com/hawkeye217/yolov9-license-plates/raw/refs/heads/master/models/yolov9-256-license-plates.onnx

# Push to device
adb shell "mkdir -p /data/frigate/config/model_cache/paddleocr-onnx /data/frigate/config/model_cache/yolov9_license_plate"
adb push /tmp/lpr_models/paddleocr-onnx/* /data/frigate/config/model_cache/paddleocr-onnx/
adb push /tmp/lpr_models/yolov9/* /data/frigate/config/model_cache/yolov9_license_plate/
```

### 5. Create Frigate Configuration

```bash
adb shell "mkdir -p /data/frigate/config"
adb shell 'cat > /data/frigate/config/config.yml << "EOF"
mqtt:
  enabled: false

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
      - "沪AHG0162"

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
EOF'
```

### 6. Start Frigate Container

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

### 7. Verify

```bash
# Wait 60 seconds for startup
sleep 60

# Check stream connection
adb shell 'docker exec frigate curl -s http://localhost:5000/api/stats' | grep camera_fps
# Expected: "camera_fps": 5.0

# Check LPR running
adb shell 'docker exec frigate curl -s http://localhost:5000/api/stats' | grep plate_recognition
# Expected: "plate_recognition": 4.1 (or similar > 0)

# Check recognized plates in logs
adb shell 'docker logs frigate 2>&1' | grep "Final clustered plate"
# Expected: Final clustered plate: '沪A·HG0162' (conf: 1.000)
```

## Configuration Reference

### go2rtc Configuration (`/data/go2rtc/go2rtc.yaml`)

```yaml
streams:
  lpr_test:
    # Loop video infinitely at original frame rate
    - ffmpeg:/videos/test.mp4#video=copy#audio=copy#rawArgs=-stream_loop -1 -re

api:
  listen: ":1984"   # Web UI and API

rtsp:
  listen: ":8556"   # RTSP output (avoid 8554 conflict with Frigate's internal go2rtc)

webrtc:
  listen: ":8557"   # WebRTC

log:
  level: info
```

**Key Parameters**:
- `#video=copy#audio=copy`: Pass-through without re-encoding (efficient)
- `#rawArgs=-stream_loop -1 -re`: Loop infinitely, play at real-time rate
- Port 8556: Must differ from Frigate's internal go2rtc (8554)

### Frigate Configuration (`/data/frigate/config/config.yml`)

```yaml
mqtt:
  enabled: false

detectors:
  openvino:
    type: openvino
    device: GPU          # GPU for main object detection

model:
  width: 300             # MUST match SSD MobileNet V2 input (300x300)
  height: 300
  input_tensor: nhwc

lpr:
  enabled: true
  device: GPU            # GPU for LPR models (YOLOv9 + PaddleOCR)
  detection_threshold: 0.3   # Lower = detect more plates (default 0.7)
  recognition_threshold: 0.7 # Lower = accept lower confidence OCR
  min_plate_length: 5        # Minimum characters to accept
  min_area: 300              # Minimum plate pixel area
  model_size: small          # "small" supports Chinese + Latin characters
  debug_save_plates: true    # Save plate images for debugging
  known_plates:
    My_Car:
      - "沪AHG0162"          # Known plate to match (assigned sub_label "My_Car")

cameras:
  lpr_camera:
    enabled: true
    type: lpr                # REQUIRED: Dedicated LPR camera mode
    lpr:
      enabled: true
      enhancement: 3         # Image enhancement (0-10) before OCR
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8556/lpr_test
          roles:
            - detect
    detect:
      enabled: false         # Disable standard object detection
      width: 1920            # Frame size for LPR processing
      height: 1080
      fps: 5                 # Frames per second to process
    objects:
      track: []              # REQUIRED: Empty list for dedicated LPR mode
    motion:
      enabled: true
      threshold: 10          # Low threshold to trigger on small motion
      contour_area: 5        # Small contour area for sensitivity
      improve_contrast: true
```

## Important Notes

### Dedicated LPR Camera Mode (`type: lpr`)

This mode bypasses standard object detection and runs YOLOv9 license plate detection directly on frames with motion:
- **Must set** `type: lpr` on the camera
- **Must set** `detect.enabled: false`
- **Must set** `objects.track: []` (empty list)
- Motion detection triggers LPR processing
- No need to detect `car` first

### Model Size Configuration

The `model` section MUST specify `width: 300` and `height: 300` to match the SSD MobileNet V2 input tensor. Without this, the shared memory buffer allocation fails:

```
TypeError: buffer is too small for requested array
```

### Host Network Mode

Both containers MUST use `--network host`:
- Eliminates Docker bridge network isolation
- Allows direct `127.0.0.1` RTSP communication
- No port mapping needed (ports are directly on host)

### NPU Device Paths (for future NPU use)

On Android devices, mount NPU to both paths:
```bash
--device /dev/accel0:/dev/accel0 \
--device /dev/accel0:/dev/accel/accel0
```

### Chinese Character Support

- Use `model_size: small` — supports both Chinese and Latin characters
- `model_size: large` only supports Latin characters
- OCR output includes `·` separator (e.g., `沪A·HG0162`)
- Configure `known_plates` without separator: `沪AHG0162`

## Container Startup Scripts

### Full Startup Script

```bash
#!/bin/bash
# start_lpr.sh - Start go2rtc and Frigate for LPR

# Stop existing containers
docker stop go2rtc frigate 2>/dev/null
docker rm go2rtc frigate 2>/dev/null

# Start go2rtc
docker run -d \
  --name go2rtc \
  --restart=unless-stopped \
  --network host \
  --device /dev/dri:/dev/dri \
  -v /data/go2rtc:/config \
  -v /data/videos:/videos \
  alexxit/go2rtc:latest

# Wait for go2rtc to start
sleep 5

# Start Frigate
docker run -d \
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
  frigate:LPR_OV202602_v2

echo "Waiting for Frigate to start..."
sleep 60

# Verify
echo "=== go2rtc streams ==="
curl -s http://localhost:1984/api/streams | python3 -m json.tool 2>/dev/null || echo "go2rtc API not ready"

echo "=== Frigate stats ==="
docker exec frigate curl -s http://localhost:5000/api/stats | python3 -m json.tool 2>/dev/null | grep -E "camera_fps|plate_recognition"
```

### Stop Script

```bash
#!/bin/bash
# stop_lpr.sh
docker stop frigate go2rtc
docker rm frigate go2rtc
```

## Verification Commands

```bash
# Check both containers are running
docker ps | grep -E "frigate|go2rtc"

# Check go2rtc stream health
curl -s http://localhost:1984/api/streams

# Check Frigate camera connection
docker exec frigate curl -s http://localhost:5000/api/stats | grep -E "camera_fps|connection_quality"

# Check LPR performance
docker exec frigate curl -s http://localhost:5000/api/stats | grep -E "plate_recognition"

# Watch LPR logs live
docker logs -f frigate 2>&1 | grep -i "plate"

# Check recognized plates
docker logs frigate 2>&1 | grep "Final clustered plate"

# Check known plate matches
docker logs frigate 2>&1 | grep "Matched plate"

# View saved debug plate images
ls -R /data/frigate/clips/lpr/
```

## FAQ

### Q: YOLOv9 runs but reports "Detected no license plates in full frame"

**Causes**:
1. Video does not contain visible plates at the current frame
2. `detection_threshold` too high — lower to 0.3
3. `min_area` too high — lower to 300
4. Plate is too small in the frame — increase camera resolution

**Fix**: Lower thresholds and wait for video to loop to frames with visible plates.

### Q: "buffer is too small for requested array" crash

**Cause**: Missing or incorrect `model` section in config.

**Fix**: Add explicit model dimensions:
```yaml
model:
  width: 300
  height: 300
  input_tensor: nhwc
```

### Q: "motion detection disabled and object detection enabled" error

**Cause**: Cannot disable motion when object detection is enabled.

**Fix**: For dedicated LPR mode, set `detect.enabled: false` and keep `motion.enabled: true`.

### Q: Stream 404 Not Found

**Causes**:
1. go2rtc not started or crashed
2. Wrong RTSP port (Frigate internal go2rtc uses 8554)
3. Stream name mismatch

**Fix**: Verify go2rtc is running and stream name matches:
```bash
curl -s http://localhost:1984/api/streams
```

### Q: camera_fps > 0 but plate_recognition = 0

**Causes**:
1. No motion detected — lower `motion.threshold` to 10
2. Not using `type: lpr` — dedicated mode required for video without cars
3. Model download failed — pre-push models

**Fix**: Ensure config has `type: lpr`, `detect.enabled: false`, and `objects.track: []`.

### Q: OCR output has `·` but known_plates doesn't match

The replace_rules strip `·` before matching. Or define known_plates without it:
```yaml
known_plates:
  My_Car:
    - "沪AHG0162"    # Without separator
```

The matching system handles the `·` automatically.

### Q: How to switch from GPU to NPU?

Change `device` in both `detectors` and `lpr`:
```yaml
detectors:
  openvino:
    type: openvino
    device: NPU

lpr:
  device: NPU
```

Note: PaddleOCR detection model falls back to CPU on NPU due to dynamic shapes.

### Q: High CPU usage from go2rtc ffmpeg

Use `#video=copy` to avoid re-encoding. If transcoding is needed, use hardware acceleration:
```yaml
streams:
  lpr_test:
    - ffmpeg:/videos/test.mp4#video=h264#hardware=vaapi
```

### Q: How to add multiple known plates?

```yaml
lpr:
  known_plates:
    My_Car:
      - "沪AHG0162"
    Wife_Car:
      - "京B12345"
    Company:
      - "粤A.*888"    # Regex pattern
```

### Q: Where are debug plate images saved?

In `/media/frigate/clips/lpr/<camera>/<event_id>/`. Disable `debug_save_plates` in production to save storage.

## Performance Metrics (Verified)

| Metric | Value |
|--------|-------|
| Camera FPS | 5.0 |
| Connection Quality | excellent |
| YOLOv9 LPD Inference | ~3ms |
| Plate Recognition Speed | ~30ms |
| Plate Recognition Rate | ~4 plates/sec |
| Recognition Confidence | 0.95 - 1.00 |

## Verified Hardware

- **Platform**: Android-based system with Docker
- **CPU/GPU**: Intel Core Ultra X7 358H (iGPU)
- **NPU**: Intel AI Boost (0x8086:0xb03e)
- **OpenVINO**: 2026.2.0
- **Frigate**: 0.18.0
- **go2rtc**: 1.9.14
- **Docker Image**: `frigate:LPR_OV202602_v2`
