#!/usr/bin/env bash
# Pointers to the public ONNX checkpoints this repo expects.
#
#   yolo11l.onnx       Ultralytics YOLO11l, person class only
#   osnet_x0_25.onnx   Kaiyang Zhou's OSNet (light variant)
#
# The detector decodes the Ultralytics detection head (1 x 84 x N),
# shared by YOLOv8 / YOLOv9 / YOLO11, so any of those exports works -
# pick the size/resolution that fits your recall vs latency budget.
# (YOLOv10 is NMS-free with a different head and is not supported.)
#
# We do not redistribute weights; produce them locally:

cat <<EOF
Required ONNX files (placed in models/onnx/):

  1. YOLO11l (person detector; larger input -> better small-person recall)
     pip install ultralytics
     yolo export model=yolo11l.pt format=onnx imgsz=960
     mv yolo11l.onnx models/onnx/

  2. OSNet x0_25 (256-d)
     git clone https://github.com/KaiyangZhou/deep-person-reid.git
     cd deep-person-reid
     # Follow the export instructions in deep-person-reid/tools/export.py
     # for osnet_x0_25, target shape (1, 3, 256, 128)

After both files are present, run scripts/build_engines.sh.
EOF
