#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
from tqdm import tqdm
import csaps
import os
import pdb

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Testing():
    print('Testing')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    if p.mode != 2:
        print("Get dataset")
        loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(11, "tensor(0.3863)")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()

    if p.mode == 0 : # test on testdatas 
        for _, _, _, test_image in loader.Generate():
            _, _, ti = test(lane_agent, np.array([test_image]))
            cv2.imshow("test", ti[0])
            cv2.waitKey(0) 

    elif p.mode == 1: # test video
        cap = cv2.VideoCapture("./inference/inference_test_videos/test_vil.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        size = (640,360)
        out = cv2.VideoWriter('./inference/inference_test_videos/test_vil_out.avi',fourcc, 10.0, size,True)
        while(cap.isOpened()):
            ret, ori_frame = cap.read()
            if ret == True:
                torch.cuda.synchronize()
                prevTime = time.time()

                # crop
                crop_image ,crop_start_h,_ ,_ ,_  = Generator().crop_img(ori_frame)

                frame = cv2.resize(crop_image, (p.x_size, p.y_size))/255.0
                frame = np.rollaxis(frame, axis=2, start=0)
                _, _, ti = test(lane_agent, np.array([frame])) 
                curTime = time.time()
                sec = curTime - prevTime
                fps = 1/(sec)
                s = "FPS : "+ str(fps)

                w_ratio = p.x_size * 1.0 / crop_image.shape[1]
                h_ratio = p.y_size* 1.0 / crop_image.shape[0]
                _, _, ti = test_ori(lane_agent, ori_frame, np.array([frame]), w_ratio, h_ratio, crop_start_h, draw_type= 'point',thresh=p.threshold_point)
                ti[0] = cv2.resize(ti[0],size)
                out.write(ti[0])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:break
        cap.release()

    elif p.mode == 2: # test on test images folder.
        test_images = '' # path to test images folder.
        save_test_dir = ''# save test results.

        if not os.path.exists(save_test_dir):
            os.makedirs(save_test_dir)

        img_list = os.listdir(test_images)
        # img_list = [img for img in img_list if '.jpg' in img]
        use_ori = True
        print("image test")
        for img in img_list:
            print("Now Dealing With:",img)

            ori_image = cv2.imread(test_images + '/' + img) #hw, cv2.IMREAD_UNCHANGED

            crop_image , crop_start_h,_ ,_ ,_ = Generator().crop_img(ori_image)
            test_image = cv2.resize(crop_image, (p.x_size, p.y_size)) / 255.0
            test_image = np.rollaxis(test_image, axis=2, start=0)

            if not use_ori:
                _, _, ti = test(lane_agent, np.array([test_image]))
            if use_ori:
                w_ratio = p.x_size * 1.0 / crop_image.shape[1]
                h_ratio = p.y_size* 1.0 / crop_image.shape[0]
                _, _, ti = test_ori(lane_agent, ori_image, np.array([test_image]), w_ratio, h_ratio, crop_start_h, draw_type= 'point',thresh=p.threshold_point)
            cv2.imwrite(save_test_dir + '/' + "{}_tested.jpg".format(img.split('.jpg')[0]), ti[0])

    elif p.mode == 3: #evaluation
        print("evaluate")
        evaluation(loader, lane_agent)
############################################################################
## evaluate on the test dataset
############################################################################
def evaluation_base(loader, lane_agent, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    for test_image, target_h, ratio_w, ratio_h, testset_index in loader.Generate_Test():
        x, y, _ = test(lane_agent, np.array([test_image]), thresh)
        x, y = util.convert_to_original_size(x[0], y[0], ratio_w, ratio_h)
        # x, y = find_target(x, y, target_h, ratio_w, ratio_h) # remove find target.
        x, y = find_target_limit(x, y, target_h, ratio_w, ratio_h) 
        result_data = write_result_json(result_data, x, target_h, testset_index)
    if name == None:
        save_result(result_data, "pred_result_{}.json".format(thresh))
    else:
        save_result(result_data, name)
############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, index= -1, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    progressbar = tqdm(range(loader.size_test//4))
    for test_image, target_h, ratio_w, ratio_h, testset_index, gt in loader.Generate_Test():
       
        # pdb.set_trace()

        x, y, _ = test(lane_agent, test_image, thresh, index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        #x_, y_ = find_target(x_, y_, target_h, ratio_w, ratio_h)
        x_, y_ = fitting(x_, y_, target_h, ratio_w, ratio_h)
        # pdb.set_trace()
        result_data = write_result_json(result_data, x_, target_h, testset_index)
        # result_data = write_result_json(result_data, x_, y_, testset_index)

        #util.visualize_points_origin_size(x_[0], y_[0], test_image[0], ratio_w, ratio_h)
        #print(gt.shape)
        #util.visualize_points_origin_size(gt[0], y_[0], test_image[0], ratio_w, ratio_h)

        progressbar.update(1)
    progressbar.close()
    if name == None:
        save_result(result_data, "test_result.json")
    else:
        save_result(result_data, name)

###########################################################################################################
## linear interpolation for fixed y value on the test dataset, if you want to use python2, use this code
###########################################################################################################
def find_target(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    count = 0
    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    for k in range(len(j)-1):
                        if j[k] >= h and h >= j[k+1]:
                            #linear regression
                            if i[k] < i[k+1]:
                                temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            else:
                                temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            break
                else:
                    if i[0] < i[1]:
                        l = int(i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)                            
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch)
        count += 1
    
    return out_x, out_y

def find_target_limit(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = int(p.x_size/ratio_w)
    y_size = int(p.y_size/ratio_h)
    for i, j in zip(x,y):
        min_y = min(j)
        max_y = max(j)
        temp_x = []
        temp_y = []
        for h in target_h:
            temp_y.append(h)
            if h < min_y or h > max_y:
                temp_x.append(-2)
            elif min_y <= h and h <= max_y:
                for k in range(len(j)-1):
                    if j[k] >= h and h >= j[k+1]:
                        #linear regression
                        if i[k] < i[k+1]:
                            temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        else:
                            temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        break
    #            else:
    #                if i[0] < i[1]:
    #                    l = int(i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
    #                    if l > x_size or l < 0 :
    #                        temp_x.append(-2)
    #                    else:
    #                        temp_x.append(l)
    #                else:
    #                    l = int(i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
    #                    if l > x_size or l < 0 :
    #                        temp_x.append(-2)
    #                    else:
    #                        temp_x.append(l)
        out_x.append(temp_x)
        out_y.append(temp_y)

    return out_x, out_y

def fitting(x, y, target_h, ratio_w, ratio_h):
    out_x = []
    out_y = []
    count = 0
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 
        count += 1

    return out_x, out_y

############################################################################
## write result
############################################################################
def write_result_json(result_data, x, y, testset_index):
    for index, batch_idx in enumerate(testset_index):
        result_data[batch_idx]['lanes'] = []
        result_data[batch_idx]['h_samples'] = y[index].tolist()
        for i in x[index]:
            result_data[batch_idx]['lanes'].append(i)
        if not result_data[batch_idx]['lanes']:
            result_data[batch_idx]['lanes'].append([-2] * len(y[index].tolist()))
    return result_data

############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):

    result = lane_agent.predict_lanes_test(test_images)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[index]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)
        
    return out_x, out_y,  out_images

############################################################################
## test on the input test image,and show result on original output.
############################################################################
def test_ori(lane_agent, ori_image, test_images,w_ratio, h_ratio,crop_start_h, draw_type, thresh=p.threshold_point):  # p.threshold_point:0.81

    result = lane_agent.predict_lanes_test(test_images)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[-1]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()
    
        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
       

        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        if draw_type == 'line':
            result_image = util.draw_lines_ori(in_x, in_y, ori_image,w_ratio, h_ratio, crop_start_h)  # line
        elif draw_type == 'point':
            result_image = util.draw_point_ori(in_x, in_y, ori_image,w_ratio, h_ratio, crop_start_h)  # point
        else:
            result_image = util.draw_points(in_x, in_y,deepcopy(image))  

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)
        
    return out_x, out_y,  out_images

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
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

if __name__ == '__main__':
    Testing()
