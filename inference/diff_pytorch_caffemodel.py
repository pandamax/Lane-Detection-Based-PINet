
'''
compare pytorch model with converted caffe model.
if the first six decimal places are the same,think it reasonable.

'''
import sys
sys.path.insert(0,'..')
import torch
import numpy 
import onnxruntime as rt 
import os
from hourglass_network import lane_detection_network
import caffe
import pdb

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference(model_pth, model_caffe, save_dir):
    
    img = torch.ones(1, 3, 256, 256, device='cpu') #(B,C,H,W)

    # to_numpy.
    test_image = to_numpy(img)

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
    print('confidences_pth_shape:',confidences_pth.shape)#
    # print('confidence_pth:',confidences_pth)

    ################################
    #   CAFFE MODEL
    ################################
    caffe_out_layer = 'conv113'
    caffe.set_mode_cpu()
    protofile = model_caffe + '.prototxt'
    weightfile = model_caffe + '.caffemodel'

    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['blob1'].data[...] = test_image
    output = net.forward(end = caffe_out_layer)
    confidences_cfe = output[caffe_out_layer]

    # check confidence
    b, c, h, w = confidences_cfe.shape
    total_diff = 0
    total_AxB = 0
    total_AxA= 1e-16
    total_BxB= 1e-16
    for i in range(b):
        for j in range(c):
            for k in range(h):
                for m in range(w):
                    pth_val = confidences_pth[i,j,k,m]
                    cfe_val= confidences_cfe[i,j,k,m]

                    total_AxB += pth_val * cfe_val
                    total_AxA += pth_val * pth_val
                    total_BxB += cfe_val * cfe_val

                    # write val into txt.
                    diff_file = open('{}/pytorch2onnx_simlarity.txt'.format(save_dir),'a+')
                    diff = abs(cfe_val - pth_val)
                    total_diff += diff
                    diff_file.write('mode_pth:{:20}\tmodel_caffe:{:20}\tdiff:{:20}\n'.format(pth_val, cfe_val, diff))
                    diff_file.close()
    print("confidence_total_diff:",total_diff)
    # Cosine Similarity
    cosine_sim = total_AxB /(pow(total_AxA, 1/2)*pow(total_BxB, 1/2))
    print("cosine similarity:", cosine_sim)

if __name__ == '__main__':
    index = 16
    loss = 0.7522
    pth_model = '../savefile/{}_tensor({})_lane_detection_network.pkl'.format(index, loss)
    # if base_onnx_model = '../converter/onnx_models/pinet_256x256_{}_{}_sim.onnx',    
    # confidences_base, offsets_base, instances_base = sess_base.get_outputs()[3].name, sess_base.get_outputs()[4].name, sess_base.get_outputs()[5].name
    caffe_model = '../converter/caffe_models/pinet_256x256_{}_{}'.format(index, loss) 
    save_dir = './pytorch2caffe_conformance'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    inference(pth_model, caffe_model, save_dir)

