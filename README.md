
## Use PINet As an Lane Detetctor. 

- the project use PINet as lane detector, supporting training on [VIL-100](https://github.com/yujun0-0/MMA-Net/tree/main/dataset). 
  At the same time, the project supports model conversion, including onnx and caffe formats, as well as model forward acceleration 
  processing before model deployment.Model forward acceleration mainly includes model cutting, simplification and merge batchnorm layer. 
  Meanwhile, it includes the verification and comparison of the conformance after the model conversion. 

## Dependency
- python (tested on python3)
- pytorch (tested on pytorch 1.6.0 with GPU(RTX2080ti))
- opencv
- numpy
- visdom (for visualization)
- sklearn (for evaluation)
- ujson (for evaluation)
- csaps (for spline fitting)
- onnx  (tested version 1.8.1)                               
- onnx-simplifier(tested version 0.2.10) 
- caffe(cpu version,for converting and inferencing caffe model)

## Dataset (VIL-100)
### VIL-100 download
- [VIL-100 Dataset: A Large Annotated Dataset of Video Instance Lane Detection](https://github.com/yujun0-0/MMA-Net/tree/main/dataset)
### datasets structure:
```
VIL-100
    |----Annotations
    |----data
    |----JPEGImages
    |----Json
    |----train.json
```

### Parse VIL-100
for parsing and converting dataset vil100.

- vis_vil.py: visualize datasets on original image,incude points and curves form.
- vil2mask.py：generate lane instance mask.
- vil2tusimples.py: convert datasets to tusimple-like format.
- vis_converted.py: visualize converted tusimple-like format.

the [scripts](https://github.com/pandamax/parse_vil100) above have tested on wsl/linux.

### Get Converted Labels
- Finally, you should get the converted label files in data folder,`train_converted.json and test_converted.json`
```
dataset/VIL-100
            |----Annotations
            |----data
                |----train_converted.json
                |----test_converted.json
                |----...
                                        
            |----JPEGImages
            |----Json
            |----train.json
          
```
Next, you need to change "train_root_url" and "test_root_url" to your "train_set" and "test_set" directory path in "parameters.py". For example,

```python
# In "parameters.py"
train_root_url= "path to VIL-100"
test_root_url= "path to VIL-100"
```

## Test
We provide trained model, and it is saved in "savefile" directory. You can run "test.py" for testing, and it has some mode like following functions 
- mode 0 : Visualize results on test set
- mode 1 : Run the model on the given video. If you want to use this mode, enter your video path at line 63 in "test.py"
- mode 2 : Run the model on the given image. If you want to use this mode, enter your image path at line 82 in "test.py"
- mode 3 : Test the model on whole test set, and save result as json file.

You can change mode at line 24 in "parameters.py".

If you want to use other trained model, just change following 2 lines.
```python
# In "parameters.py"
line 13 : model_path = "<your model path>/"
# In "test.py"
line 42 : lane_agent.load_weights(<>, "tensor(<>)")
```

## Train
If you want to train from scratch, make line 13 blank in "parameters.py", and run "train.py"
```python
# In "parameters.py"
line 14 : model_path = ""
```
"train.py" will save sample result images(in "test_result/"), trained model(in "savefile/").

If you want to train from a trained model, just change following 2 lines.
```python
# In "parameters.py"
line 15 : model_path = "<your model path>/"
# In "train.py"
line 58 : lane_agent.load_weights(<>, "tensor(<>)")
```

## Network Clipping 
PINet is made of several hourglass modules; these hourglass modules are train by the same loss function.

We can make ligher model without addtional training by clipping some hourglass modules.

```python
# In "hourglass_network.py"
self.layer1 = hourglass_block(128, 128)
self.layer2 = hourglass_block(128, 128)
#self.layer3 = hourglass_block(128, 128)
#self.layer4 = hourglass_block(128, 128) some layers can be commentted 
```
## Net MACs
- get the network macs.
```python
# in flopscount.py
python flopscount.py
```

## Net Converter
- convert pytorch model to onnx, and simplify it.you can set params in `onnx_converter.sh`.For the convenience of model conversion,
  Conv and BN in the model are combined in a Sequential module.
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

## Verify the Converted Models And Inference

### verify the conformance of converted models 
(in my experiment:if the first six decimal places are the same,think it reasonable.)
- diff_fuse_model_inference.py: verify the conformance of the model before and after fuse_bn operation.
- diff_pytorch_onnx_inference.py: verify the conformance of the pytorch model and converted onnx model.
- diff_pytorch_caffe_inference.py: verify the conformance of the pytorch model and converted caffe model.


### inference test of converted models
- merged_caffemodel_inference.py: inference test of converted caffe model.
- onnx_inference.py: inference test of converted onnx model.
- pytorch_caffemodel_inference.py: inference test of the base pytorch model and corresponding converted caffe model.


## Demo
<table>
    <tr>
        <td><img src=assets/demo.gif /></td>
    </tr>
</table>

## Reference
- [key points estimation and point instance segmentation approach for lane detection](https://github.com/koyeongmin/PINet_new)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- [pytorch_to_caffe](https://github.com/WolffyChen/PytorchToCaffe/blob/master/pytorch_to_caffe.py)
- [pytorch_bn_fusion](https://github.com/MIPT-Oulu/pytorch_bn_fusion)
