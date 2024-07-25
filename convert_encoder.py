import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL="sam_vit_b_01ec64.pth.encoder.patched.onnx"
RKNN_MODEL="sam_vit_b_01ec64.pth.encoder.onnx.rknn"
rknn = RKNN(verbose=True)

# pre-process config
print('--> config model')
rknn.config(target_platform='rk3588', single_core_mode=True)
print('done')

# Load model
print("--> Loading model")
ret = rknn.load_onnx(
    model=ONNX_MODEL, inputs=["input_image"], input_size_list=[[1024, 1024, 3]]
)
if ret != 0:
    print("Load model failed!")
    exit(ret)
print("done")

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export rknn model
print('--> Export rknn model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')
