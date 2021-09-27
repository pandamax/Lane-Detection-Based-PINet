#!/bin/bash
## pytorch->merge bn->caffe model

model_dir='savefile'
caffe_models_dir='./caffe_models'
input_size=(256 256) #hw
index=16
loss=0.7522

pytorch2caffe_shell='pinet_pytorch_to_caffe.py'

echo "converting ..."

echo "=== start ==="
if [ -x "${pytorch2caffe_shell}" ]; then
    python ${pytorch2caffe_shell} \
    --caffe_models_dir ${caffe_models_dir} --model_dir ${model_dir} \
    --model_index ${index} --model_loss ${loss} --y_size ${input_size[0]} --x_size ${input_size[1]} 
else
    echo "error occured in ${pytorch2caffe_shell}!, failed!"
fi

echo "finished!"
