import sys
sys.path.insert(0,'..')
import torch
from torch.autograd import Variable
from hourglass_network import lane_detection_network

import pytorch_to_caffe_git as pytorch_to_caffe
import os
import merge_bn_seq
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--caffe_models_dir', default='caffe_models', type=str, help='the dir path to save converted onnx model')
    parser.add_argument('--model_dir', default='savefile', type=str, help='the dir path to save trained pytorch model')
    parser.add_argument('--model_index', default= 16, type=int, help='the trained pytorch model index')
    parser.add_argument('--model_loss', default= 0.7522, type=float, help='the trained pytorch model loss value')
    parser.add_argument('--y_size', default= 256, type=int, help='net input size y')
    parser.add_argument('--x_size', default= 256, type=int, help='net input size x')

    return parser.parse_args()

if __name__=='__main__':

    args = parse_args()
    save_dir = './{}'.format(args.caffe_models_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    network = 'pinet_{}x{}_{}_{}'.format(args.y_size, args.x_size, args.model_index, args.model_loss)

    net = lane_detection_network()
    model_pth = '../{}/{}_tensor({})_lane_detection_network.pkl'.format(args.model_dir, args.model_index, args.model_loss)

    state_dict = torch.load(model_pth, map_location='cpu')
  
    print(state_dict.keys())
    net.load_state_dict(state_dict)

    net.eval()

    net = merge_bn_seq.fuse_bn_recursively(net)

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    #for k,v in state_dict.items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    # net.load_state_dict(new_state_dict,False)
    # net = merge_bn_seq.fuse_module(net)
    #net.load_state_dict(torch.load(pth_path, map_location='cpu'))
    # net.eval()

    input = Variable(torch.ones([1, 3, args.y_size, args.x_size]))
    
    pytorch_to_caffe.trans_net(net, input, network)
    pytorch_to_caffe.save_prototxt('{}/{}.prototxt'.format(save_dir, network))
    pytorch_to_caffe.save_caffemodel('{}/{}.caffemodel'.format(save_dir, network))
