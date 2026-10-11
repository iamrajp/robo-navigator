"""Camera helper - run with .camvenv/bin/python (system python + picamera2 0.3.17).
The newer system picamera2 (0.3.25) gives a blurry, mis-coloured picture on this camera; 0.3.17 matches car-server.
Setup:  uv venv --system-site-packages .camvenv -p /usr/bin/python3 && uv pip install -p .camvenv/bin/python --no-deps picamera2==0.3.17
Writes each JPEG frame to stdout as a 4-byte big-endian length + the bytes.
nav_server.py starts this as a subprocess (the uv venv's newer numpy can't load picamera2)."""
import io
import struct
import sys

import numpy as np
from PIL import Image
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
out = sys.stdout.buffer
try:
    while True:
        buf = io.BytesIO()
        frame = np.ascontiguousarray(cam.capture_array()[::-1, ::-1, ::-1])   # upside down + BGR->RGB (the ISP hflip/vflip breaks colours)
        Image.fromarray(frame).save(buf, format="JPEG", quality=85)
        b = buf.getvalue()
        out.write(struct.pack(">I", len(b)) + b)
        out.flush()
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    cam.stop()
