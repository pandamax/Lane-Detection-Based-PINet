#!/bin/bash
# pytorch->onnx sim(merge bn)->onnx_edit->onnx_remove_initalizer_from_input
model_dir='savefile'
onnx_models_dir='./onnx_models'
input_size=(256 256) #hw
index=16
loss=0.7522

pytorch2onnx_shell='pytorch2onnx.py'
onnx_edit_shell='onnx_edit.py'
rifi_shell='remove_initializer_from_input.py'

echo "converting ..."

echo "=== step 1 ==="
if [ -x "${pytorch2onnx_shell}" ]; then
    python ${pytorch2onnx_shell} \
    --onnx_models_dir ${onnx_models_dir} --model_dir ${model_dir} \
    --model_index ${index} --model_loss ${loss} --y_size ${input_size[0]} --x_size ${input_size[1]} 
else
    echo "error occured in ${pytorch2onnx_shell}!,step 1 failed!"
fi

echo "=== step 2 ==="
# step 2
if [ -x "${onnx_edit_shell}" ]; then
    python ${onnx_edit_shell} ${onnx_models_dir}/pinet_${input_size[0]}x${input_size[1]}_${index}_${loss}_sim.onnx \
    ${onnx_models_dir}/pinet_${input_size[0]}x${input_size[1]}_${index}_${loss}_sim_edit.onnx \
    --inputs input.1[1,3,256,256] \
    --outputs 1544[1,1,32,32],1551[1,2,32,32],1558[1,4,32,32]
else
    echo "error occured in ${onnx_edit_shell}!,step 2 failed!"
fi


echo "=== step 3 ==="
# step 3
if [ -x "${rifi_shell}" ]; then
    python ${rifi_shell} \
    --input ${onnx_models_dir}/pinet_${input_size[0]}x${input_size[1]}_${index}_${loss}_sim_edit.onnx \
    --out ${onnx_models_dir}/pinet_${input_size[0]}x${input_size[1]}_${index}_${loss}_sim_edit_rifi.onnx
else
    echo "error occured in ${rifi_shell}!,step 3 failed!"
fi

echo "finished!"
