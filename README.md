# Sapiens-Pytorch-Inference
 Minimal code and examples for inferencing Sapiens foundation human models in Pytorch

![!ONNX Sapiens_normal_segmentation](https://github.com/user-attachments/assets/a8f433f0-5f43-4797-89c6-5b33c58cbd01)

# Why
- Make it easy to run the models by creating a `SapiensPredictor` class that allows to run multiple tasks simultaneously
- Add several examples to run the models on images, videos, and with a webcam in real-time.
- Download models automatically from HuggigFace if not available locally.
- Add a script for ONNX export. However, ONNX inference is not recommended due to the slow speed.
- Added Object Detection to allow the model to be run for each detected person. However, this mode is disabled as it produces the worst results.

> [!CAUTION]
> - Use 1B models, since the accuracy of lower models is not good (especially for segmentation)
> - Exported ONNX models are too slow.
> - Input sizes other than 768x1024 don't produce good results.
> - Running Sapiens models on a cropped person produces worse results, even if you crop a wider rectangle around the person.

## Installation
```bash
git clone https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git
cd Sapiens-Pytorch-Inference
pip install -r requirements.txt
```
