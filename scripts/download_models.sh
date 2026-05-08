#!/usr/bin/env bash
# Pointers to the public ONNX checkpoints this repo expects.
#
#   yolov8s.onnx       Ultralytics YOLOv8s, person class only
#   osnet_x0_25.onnx   Kaiyang Zhou's OSNet (light variant)
#
# We do not redistribute weights; produce them locally:

cat <<EOF
Required ONNX files (placed in models/onnx/):

  1. YOLOv8s
     pip install ultralytics
     yolo export model=yolov8s.pt format=onnx imgsz=640
     mv yolov8s.onnx models/onnx/

  2. OSNet x0_25 (256-d)
     git clone https://github.com/KaiyangZhou/deep-person-reid.git
     cd deep-person-reid
     # Follow the export instructions in deep-person-reid/tools/export.py
     # for osnet_x0_25, target shape (1, 3, 256, 128)

After both files are present, run scripts/build_engines.sh.
EOF
