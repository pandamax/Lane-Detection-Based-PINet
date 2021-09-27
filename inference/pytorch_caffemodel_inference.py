'''
inference test of the base pytorch model and corresponding converted caffe model.
'''
import sys
sys.path.insert(0,'..')
import caffe
import numpy as np  
import os
import cv2
from copy import deepcopy
import pdb
from hourglass_network import lane_detection_network
import torch

#set parameters.
class param:
    def __init__(self):
        self.color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),
                    (100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]

        self.x_size = 256 
        self.y_size = 256 
        self.resize_ratio = 8
        self.grid_x = self.x_size//self.resize_ratio 
        self.grid_y = self.y_size//self.resize_ratio 

        self.threshold_point = 0.85
        self.threshold_instance = 0.08
        self.use_cuda = False


        self.grid_location = np.zeros((self.grid_y, self.grid_x, 2))
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                self.grid_location[y][x][0] = x
                self.grid_location[y][x][1] = y

p = param() 

# crop params should be same in data_loader.
def crop_img(temp_image):
    crop_start_h = temp_image.shape[0]//4
    crop_end_h = temp_image.shape[0]
    crop_image = temp_image[crop_start_h:crop_end_h,:,:]
    return crop_image, crop_start_h

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

# draw_points on original img.
def draw_points(x, y, image, w_ratio, h_ratio, crop_start_h):
    color_index = 0
    for id in range(len(x)):
        color_index += 1
        if color_index > 12:
            color_index = 12 
        x_l = x[id]
        x_list = [int(x / w_ratio) for x in x_l]
        y_l = y[id]
        y_list = [int(y / h_ratio)+crop_start_h for y in y_l]
        for pts in zip(x_list, y_list):
            image = cv2.circle(image, (int(pts[0]), int(pts[1])), 8, p.color[color_index], -1)  # 5
    return image

def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]

    offset = offsets[mask]

    feature = instance[mask]
   
    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y


def test(model_output, test_image, w_ratio, h_ratio ,crop_start_h, thresh=p.threshold_point):

    confidence, offset, instance = model_output[0],model_output[1],model_output[2]
    
    out_x = []
    out_y = []
    out_image = []
    
    confidence = np.squeeze(confidence)
  
    offset = np.squeeze(offset)
    offset = np.rollaxis(offset, axis=2, start=0)
    offset = np.rollaxis(offset, axis=2, start=0)
    
    instance = np.squeeze(instance)
    instance = np.rollaxis(instance, axis=2, start=0)
    instance = np.rollaxis(instance, axis=2, start=0)
    
    # generate point and cluster
    raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

    # eliminate fewer points
    in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
            
    # sort points along y 
    in_x, in_y = sort_along_y(in_x, in_y) 

    result_image = draw_points(in_x, in_y,test_image, w_ratio, h_ratio,crop_start_h)

    out_x.append(in_x)
    out_y.append(in_y)
        
    return out_x, out_y,  result_image

def to_np(test_image):
    test_image = np.rollaxis(test_image, axis=2, start=0)
    inputs = test_image.astype(np.float32)
    inputs = inputs[np.newaxis,:,:,:] 
    return inputs

def load_caffemodel(model_path):
    caffe.set_mode_cpu()
    protofile = model_path + '.prototxt'
    weightfile = model_path + '.caffemodel'
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    return net

def caffemodel_out(net, image):
    net.blobs['blob1'].data[...] = image
    output = net.forward()
    return [output['conv113'],output['conv116'],output['conv119']]


def load_pytorchmodel(model_dir):
    net = lane_detection_network()   
    state_dict = torch.load(model_dir, map_location='cpu')
    net.load_state_dict(state_dict)
    return net

def pytorchmodel_out(net, image):
    image = torch.from_numpy(image).float() 
    with torch.no_grad():
        result, _ = net(image)
        
    return [result[-1][0].numpy(), result[-1][1].numpy() ,result[-1][2].numpy()]
    
def inference(pth_model, caffe_model, test_images,save_test_dir):

    pytorch_net = load_pytorchmodel(pth_model)

    caffe_net = load_caffemodel(caffe_model)

    img_list = os.listdir(test_images)
    img_list = [img for img in img_list if 'png' in img or '.jpg' in img]
    use_ori = True

    for img in img_list:
        print("Now Dealing With:",img)
        ori_image = cv2.imread(test_images + '/' + img) #hw, cv2.IMREAD_UNCHANGED
        caffe_img = ori_image.copy()
        pytorch_img = ori_image.copy()

        crop_image, crop_start_h = crop_img(ori_image)

        test_image = cv2.resize(crop_image, (p.x_size, p.y_size)) / 255.0
        
        test_image = to_np(test_image)

        pred_pytorch = pytorchmodel_out(pytorch_net.eval(), test_image)
        pred_caffe = caffemodel_out(caffe_net, test_image)
        
        w_ratio = p.x_size * 1.0 / crop_image.shape[1]
        h_ratio = p.y_size* 1.0 / crop_image.shape[0]

        _, _, ti_pytorch = test(pred_pytorch, pytorch_img, w_ratio, h_ratio ,crop_start_h, thresh=p.threshold_point)
        _, _, ti_caffe = test(pred_caffe, caffe_img, w_ratio, h_ratio ,crop_start_h, thresh=p.threshold_point)

        cv2.putText(ti_pytorch, 'pytorch', (ori_image.shape[1]//2, ori_image.shape[0]//8), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(ti_caffe, 'caffe',  (ori_image.shape[1]//2, ori_image.shape[0]//8), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        concat_img = cv2.hconcat([ti_pytorch, ti_caffe])
        cv2.imwrite(save_test_dir + '/' + "{}_tested.jpg".format(img.split('.jpg')[0]), concat_img)


if __name__ == '__main__':

    index = 16
    loss = 0.7522
    pth_model = '../savefile/{}_tensor({})_lane_detection_network.pkl'.format(index, loss) 
    caffe_model = '../converter/caffe_models/pinet_256x256_{}_{}'.format(index, loss) 

    save_test_dir = './inference_test_images/test_pytorch_caffe_result_{}_{}'.format(index, loss)
    test_dataset_list = ['9_Road028_Trim005_frames'] 

    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)
    print("======= MODEL INFERENCE =======")

    test_dataset_list = ['9_Road028_Trim005_frames'] 

    for test_dateset in test_dataset_list:
        test_images = './inference_test_images/{}'.format(test_dateset)
        inference(pth_model, caffe_model, test_images, save_test_dir)

    print("finished~~")


