##### convert model
- convert pytorch model to onnx, and simplify it.you can set params in `onnx_converter.sh`.
  ```shell
  cd ${path_to_converter_folder}
  ./onnx_converter.sh
  ```
- convert pytorch model to caffe model.you can set params in `caffe_converter.sh`.
  ```shell
  cd ${path_to_converter_folder}
  ./caffe_converter.sh
  ```
  simplify the caffe prototxt manually.
  - change the input layer name.
  - remove the layers that are not related to the final output.
  - example in `./converter/caffe_models, pinet_256x256_16_0.7522_base.prototxt -> pinet_256x256_16_0.7522.prototxt`.
