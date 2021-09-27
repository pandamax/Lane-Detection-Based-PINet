#### inference

##### verify the conformance of converted models 
(in my experiment:if the first six decimal places are the same,think it reasonable.)
- diff_fuse_model_inference.py: verify the conformance of the model before and after fuse_bn operation.
- diff_pytorch_onnx_inference.py: verify the conformance of the pytorch model and converted onnx model.
- diff_pytorch_caffe_inference.py: verify the conformance of the pytorch model and converted caffe model.


##### inference test of converted models
- merged_caffemodel_inference.py: inference test of converted caffe model.
- onnx_inference.py: inference test of converted onnx model.
- pytorch_caffemodel_inference.py: inference test of the base pytorch model and corresponding converted caffe model.
