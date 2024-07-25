---
license: unknown
---

# 适用于RKNN2的Segment Anything模型
# Segment Anything Model for RKNN2 (English readme see below)


## 模型使用

- 安装RKNPU2 2.0.0b23版本运行库
- 在开发板上安装python-opencv, rknn-toolkit-lite2, onnxruntime等
- 从 https://huggingface.co/happyme531/segment-anything-rknn2 下载模型文件(`sam_vit_b_01ec64.pth.encoder.patched.onnx.rknn`,`sam_vit_b_01ec64.pth.decoder.onnx`)
- 执行run_sam_rknn.py即可

## 效果展示

输入:

![](input.jpg)

提示：

`{"type": "point", "data": [540, 512], "label": 1}`

输出:

![](output.jpg)

性能: RK3588，单NPU核心，耗时约22000ms  
..性能瓶颈: Softmax太大，NPU无法执行

## 模型转换

- (使用RKNN-Toolkit2 2.0.0b23版本测试)
- 使用`https://github.com/vietanhdev/samexporter` 导出ONNX模型
- 编辑`convert_encoder.py`, 修改模型路径:
  ```
  ONNX_MODEL="sam_vit_b_01ec64.pth.encoder.onnx"
  ```
- 执行`convert_encoder.py`
- 现在会输出一个rknn文件, 但它的执行速度非常慢(~120s), 因为模型结构需要调整
- 执行`patch_graph.py`, 会生成调整后的onnx文件
- 再次编辑`convert_encoder.py`, 修改模型路径, 执行转换即可
- decoder模型运行很快，因此无需转换，直接用onnxruntime cpu运行即可

# English readme

## Model Usage

- Install RKNPU2 2.0.0b23 version runtime library
- Install python-opencv, rknn-toolkit-lite2, onnxruntime, etc. on the development board
- Download model files from https://huggingface.co/happyme531/segment-anything-rknn2 (`sam_vit_b_01ec64.pth.encoder.patched.onnx.rknn`, `sam_vit_b_01ec64.pth.decoder.onnx`)
- Execute run_sam_rknn.py

## Demo

Input:

![](input.jpg)

Prompt:

`{"type": "point", "data": [540, 512], "label": 1}`

Output:

![](output.jpg)

Performance: RK3588, single NPU core, takes about 22000ms  
..Performance bottleneck: Softmax is too large, NPU cannot execute

## Model Conversion

- (Tested with RKNN-Toolkit2 2.0.0b23 version)
- Use `https://github.com/vietanhdev/samexporter` to export ONNX model
- Edit `convert_encoder.py`, modify the model path:
  ```
  ONNX_MODEL="sam_vit_b_01ec64.pth.encoder.onnx"
  ```
- Execute `convert_encoder.py`
- Now it will output an rknn file, but its execution speed is very slow (~120s) because the model structure needs adjustment
- Execute `patch_graph.py`, which will generate an adjusted onnx file
- Edit `convert_encoder.py` again, modify the model path, and execute the conversion
- The decoder model runs quickly, so there's no need for conversion. It can be run directly using onnxruntime CPU.