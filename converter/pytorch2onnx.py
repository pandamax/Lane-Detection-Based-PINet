'''
Convert trained model into onnx.
envs:
pytorch 1.6
onnxsim

'''
import sys
sys.path.insert(0,'..')
import torch
import torch.onnx
from hourglass_network import lane_detection_network
# from pytorch_caffemodel_inference import param
from thop import profile
from thop import clever_format
from onnxsim import simplify
import onnx
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--onnx_models_dir', default='onnx_models', type=str, help='the dir path to save converted onnx model')
    parser.add_argument('--model_dir', default='savefile', type=str, help='the dir path to save trained pytorch model')
    parser.add_argument('--model_index', default= 13, type=int, help='the trained pytorch model index')
    parser.add_argument('--model_loss', default= 0.7866, type=float, help='the trained pytorch model loss value')
    parser.add_argument('--y_size', default= 256, type=int, help='net input size y')
    parser.add_argument('--x_size', default= 256, type=int, help='net input size x')

    return parser.parse_args()

# get flops.
def GetFlops(model,input):
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.1f")
    return macs, params

# simplifier onnx model.
def OnnxSim(onnx_model):
    # load your predefined ONNX model
    model = onnx.load(onnx_model)
    # convert model
    model_simp, check = simplify(model)
    model_simp_path = onnx_model.split('.onnx')[0] + '_sim.onnx'
    onnx.save_model(model_simp, model_simp_path)

def convert2onnx(args):

    save_dir = './{}'.format(args.onnx_models_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = lane_detection_network()
    
    weights_path = '../{}/{}_tensor({})_lane_detection_network.pkl'.format(args.model_dir, args.model_index, args.model_loss)

    # Load the weights from a file (.pth or .pkl usually)
    state_dict = torch.load(weights_path)

    # Load the weights now into a model net architecture.
    model.load_state_dict(state_dict)

    # set the model to inference mode
    model.eval() # sometimes it's useless.
    
    # Create the right input shape.
    sample_batch_size = 1
    channel = 3
    # height = 256#256
    # width = 256#256 model
    dummy_input = torch.randn(sample_batch_size, channel, args.y_size, args.x_size)

    # dummy_input = torch.randn(sample_batch_size, channel, height, width)

    macs, params = GetFlops(model, dummy_input)
    input_size = '{}x{}'.format(args.y_size, args.x_size) # hw
    # model_path = save_dir + "pinet_{}_{}_{}_{}.onnx".format(macs, input_size, args.model_index, args.model_loss)
    model_path =  "{}/pinet_{}_{}_{}.onnx".format(save_dir, input_size, args.model_index, args.model_loss)


    torch.onnx.export(model, dummy_input, model_path, verbose = True)
    print('macs:{}\nparams:{}'.format(macs, params))

    # simplifier model using onnx-sim，merge bn.
    OnnxSim(model_path)

    return model_path


if __name__ =='__main__':
    # (True)Convert to onnx mode.
    # (False)Check converted onnx model mode.
    convert = True 

    args = parse_args()
    if convert == True:

       _ = convert2onnx(args)
       print("FINISHED!")

    if convert == False:

        import onnx
        model_path = convert2onnx(args)

        # Load the onnx model
        model = onnx.load(model_path)

        # Check that the IR is well formed
        onnx.checker.check_model(model)

        # Print a human readable representation of the graph.
        print(onnx.helper.printable_graph(model.graph))

