'''
Verify the conformance of the model before and after fuse_bn operation.
'''

import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
import numpy as np
from converter.merge_bn_seq import fuse_bn_recursively
from hourglass_network import lane_detection_network
import pdb

if __name__ == '__main__':


    save_dir = './fuse_bn_test'
    model_pth_name = '../savefile/16_tensor(0.7522)_lane_detection_network'
    img = torch.ones(1, 3, 256, 256, device='cpu') #(B,C,H,W)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net = lane_detection_network()

    pth_path = os.path.join('{}.pkl'.format(model_pth_name))    
    state_dict = torch.load(pth_path, map_location='cpu')
  
    # print(state_dict.keys())
    net.load_state_dict(state_dict)
    # Benchmarking
    # First, we run the network the way it is
    net.eval()
    with torch.no_grad():
        result, _ = net(img)
        confidences, offsets, instances = result[-1]

    net = fuse_bn_recursively(net)
    net.eval()
    with torch.no_grad():
        result, _ = net(img)
        confidences_fuse, offsets_fuse, instances_fuse = result[-1]

    # check confidence
    confidences_np = confidences.numpy()
    confidences_fuse_np = confidences_fuse.numpy()
    b, c, h, w = confidences_np.shape
    total_diff = 0
    total_AxB = 0
    total_AxA= 1e-16
    total_BxB= 1e-16
    for i in range(b):
        for j in range(c):
            for k in range(h):
                for m in range(w):
                    base_val = confidences_np[i,j,k,m]
                    fuse_val= confidences_fuse_np[i,j,k,m]

                    total_AxB += base_val * fuse_val
                    total_AxA += base_val * base_val
                    total_BxB += fuse_val * fuse_val

                    # write val into txt.
                    diff_file = open('{}/conf.txt'.format(save_dir),'a+')
                    diff = abs(fuse_val - base_val)
                    total_diff += diff
                    diff_file.write('not merge:{}\tmerged:{}\tdiff:{}\n'.format(base_val, fuse_val, diff))
                    diff_file.close()
    print("confidence_total_diff:",total_diff)
    # Cosine Similarity
    cosine_sim = total_AxB /(pow(total_AxA, 1/2)*pow(total_BxB, 1/2))
    print("cosine similarity:",cosine_sim)
    # for d in range(len(confidences.shape)):
    #     pytorch_similarity = torch.cosine_similarity(confidences, confidences_fuse, dim=d)
    #     print("dim {} cosine similarity:{}".format(d, pytorch_similarity))
   
    