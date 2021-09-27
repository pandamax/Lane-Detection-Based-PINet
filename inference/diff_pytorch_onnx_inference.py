
'''
compare base_onnx model with sim_onnx_model.
if the first six decimal places are the same,think it reasonable.

'''
import sys
sys.path.insert(0,'..')
import torch
import numpy 
import onnxruntime as rt 
import os
from hourglass_network import lane_detection_network

# pth model.

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference(model_pth, model_base,  model_sim, save_dir):
    
    img = torch.ones(1, 3, 256, 256, device='cpu') #定义输入的数据类型(B,C,H,W)为(10,3,224,224)

    # log img.
    # print('img_tensor_size:',img.size())
    # print('img_tensor:', img[0])
    # test_image = img.numpy().astype(np.float32)
    test_image = to_numpy(img)

    # print('img_numpy_shape:', test_image.shape)
    # print('img_numpy:', test_image[0])

    ################################
    #   PTH MODEL
    ################################
    state_dict = torch.load(model_pth, map_location='cpu')
    # state_dict = torch.load(model_pth)
    net = lane_detection_network()
    net.load_state_dict(state_dict)
    net.eval()
    with torch.no_grad():
        result, _ = net(img)
        confidences_pth = result[-1][0] #result[-1][1] result[-1][2] offsets feature.

    confidences_pth = confidences_pth.numpy()
    print('confidences_pth_shape:',confidences_pth.shape)# 1 5 192 256
    # print('confidence_pth:',confidences_pth)

    ################################
    #   ONNX MODEL
    ################################
    # base model onnx.
    sess_base = rt.InferenceSession(model_base)
    input_name = sess_base.get_inputs()[0].name

    confidences_base, offsets_base, instances_base = sess_base.get_outputs()[0].name, sess_base.get_outputs()[1].name, sess_base.get_outputs()[2].name
    output_name_base =  [confidences_base, offsets_base, instances_base]
    print('output_name_base:',output_name_base)

    # sim model onnx.
    sess_sim = rt.InferenceSession(model_sim)
    confidences_sim, offsets_sim, instances_sim = sess_sim.get_outputs()[0].name, sess_sim.get_outputs()[1].name, sess_sim.get_outputs()[2].name
    output_name_sim =  [confidences_sim, offsets_sim, instances_sim]
    # print('output_name_sim:',output_name_sim)

    base_onnx = sess_base.run(output_name_base, {input_name:test_image})[0]
    print('base_onnx.shape:', base_onnx.shape)

    sim_onnx = sess_sim.run(output_name_sim, {input_name:test_image})[0]
    print('sim_onnx.shape:', sim_onnx.shape)


    # check confidence
    b, c, h, w = base_onnx.shape
    total_diff = 0
    total_AxB = 0
    total_AxA= 1e-16
    total_BxB= 1e-16
    for i in range(b):
        for j in range(c):
            for k in range(h):
                for m in range(w):
                    pth_val = confidences_pth[i,j,k,m]
                    base_val= base_onnx[i,j,k,m]
                    rifi_val= sim_onnx[i,j,k,m]
                    # Cosine Similarity

                    total_AxB += pth_val * rifi_val
                    total_AxA += pth_val * pth_val
                    total_BxB += rifi_val *rifi_val

                    # write val into txt.
                    diff_file = open('{}/pytorch2onnx_simlarity.txt'.format(save_dir),'a+')
                    diff_pth_onnx = abs(rifi_val - pth_val)
                    diff_onnx = abs(rifi_val - base_val)
                    total_diff += diff_onnx
                    diff_file.write('mode_pth:{:20}\tbase_onnx:{:20}\tsim_onnx:{:20}\tdiff_pth_onnx:{:20}\tdiff_onnx:{:20}\n'.format(pth_val, \
                                    base_val, rifi_val, diff_pth_onnx, diff_onnx))
                    diff_file.close()
    print("confidence_total_diff_onnx:",total_diff)
    # Cosine Similarity
    cosine_sim = total_AxB /(pow(total_AxA, 1/2)*pow(total_BxB, 1/2))
    print("cosine similarity:", cosine_sim)

if __name__ == '__main__':
    index = 16
    loss = 0.7522
    pth_model = '../savefile/{}_tensor({})_lane_detection_network.pkl'.format(index, loss)
    # if base_onnx_model = '../converter/onnx_models/pinet_256x256_{}_{}_sim.onnx',    
    # confidences_base, offsets_base, instances_base = sess_base.get_outputs()[3].name, sess_base.get_outputs()[4].name, sess_base.get_outputs()[5].name
    base_onnx_model = '../converter/onnx_models/pinet_256x256_{}_{}_sim_edit.onnx'.format(index, loss) 
    sim_edit_rifi_onnx_model ='../converter/onnx_models/pinet_256x256_{}_{}_sim_edit_rifi.onnx'.format(index, loss)
    save_dir = './pytorch2onnx_conformance'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    inference(pth_model, base_onnx_model, sim_edit_rifi_onnx_model, save_dir)

