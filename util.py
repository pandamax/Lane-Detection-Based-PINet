import torch.nn as nn
import cv2
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
from parameters import Parameters
import math

p = Parameters()

def cross_entropy2d(inputs, target, weight=None, size_average=True):
    loss = torch.nn.CrossEntropyLoss()

    n, c, h, w = inputs.size()
    prediction = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    gt =target.transpose(1, 2).transpose(2, 3).contiguous().view(-1)

    return loss(prediction, gt)

###############################################################
##
## visualize
## 
###############################################################

def visualize_points(image, x, y):
    image = image
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for k in range(len(y)):
        for i, j in zip(x[k], y[k]):
            if i > 0:
                image = cv2.circle(image, (int(i), int(j)), 2, p.color[1], -1)

    cv2.imshow("test2", image)
    cv2.waitKey(0)  

def visualize_points_origin_size(x, y, test_image, ratio_w, ratio_h):
    color = 0
    image = deepcopy(test_image)
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    image = cv2.resize(image, (int(p.x_size/ratio_w), int(p.y_size/ratio_h)))

    for i, j in zip(x, y):
        color += 1
        for index in range(len(i)):
            cv2.circle(image, (int(i[index]), int(j[index])), 10, p.color[color], -1)
    cv2.imshow("test2", image)
    cv2.waitKey(0)  

    return test_image

def visualize_gt(self, gt_point, gt_instance, ground_angle, image):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for y in range(self.p.grid_y):
        for x in range(self.p.grid_x):
            if gt_point[0][y][x] > 0:
                xx = int(gt_point[1][y][x]*self.p.resize_ratio+self.p.resize_ratio*x)
                yy = int(gt_point[2][y][x]*self.p.resize_ratio+self.p.resize_ratio*y)
                image = cv2.circle(image, (xx, yy), 10, self.p.color[1], -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)

def visualize_regression(image, gt):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for i in gt:
        for j in range(p.regression_size):#gt
            y_value = p.y_size - (p.regression_size-j)*(220/p.regression_size)
            if i[j] >0:
                x_value = int(i[j]*p.x_size)
                image = cv2.circle(image, (x_value, y_value), 5, p.color[1], -1)
    cv2.imwrite("./image.png", image)
    # cv2.waitKey(0)   

def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 5, p.color[color_index], -1)

    return image


# draw_points on original img.
def draw_point_ori(x, y, image, w_ratio, h_ratio, crop_start_h):
    # image_w = image.shape[1]
    color_index = 0
    for id in range(len(x)):
        # print("id", id)
        color_index += 1
        if color_index > 12:
            color_index = 12  # 
        x_l = x[id]
        x_list = [int(x / w_ratio) for x in x_l]
        
        y_l = y[id]
        y_list = [int(y / h_ratio)+crop_start_h for y in y_l]
        # print("x_list", x_list)
        # print("y_list", y_list)
        for pts in zip(x_list, y_list):
            image = cv2.circle(image, (int(pts[0]), int(pts[1])), 5, p.color[color_index], -1)  # 5
    return image

######################################################
### add draw_lines function,back to oringal img.
########################################################
# method 1：color choice defferent
def curve_fit(image, x_list, y_list, color):

    # print(type(image))
    x = np.array(x_list)
    y = np.array(y_list)
    #calculate the coefficients
    z = np.polyfit(y, x, 2) # 
    # start = min(min(y_list), image.shape[0] / 2)
    start = min(y_list)
    end   = max(max(y_list), image.shape[0] / 2)
    lspace = np.linspace(start, end, 20) # 
    draw_y = lspace
    draw_x = np.polyval(z, draw_y)   # evaluate the polynomial

    # base on y,choose h
    x = np.polyval(z, image.shape[0]/1.5)

    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed

    cv2.polylines(image, [draw_points], False, color, 5)
    return image

# method 2
# color choice different
def curve_fit(image, x_list, y_list, color):

    # print(type(image))
    x = np.array(x_list)
    y = np.array(y_list)
    #calculate the coefficients.
    z = np.polyfit(y, x, 2) # 

    # start = min(min(y_list), image.shape[0] / 2)
    start = min(y_list)
    end   = max(max(y_list), image.shape[0] / 2)
    lspace = np.linspace(start, end, 20) # 
    # print("test",np.max(lspace))
    draw_y = lspace
    draw_x = np.polyval(z, draw_y)   # evaluate the polynomial

    # base on y,choose h
    x = np.polyval(z, image.shape[0]/1.5)

    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed

    cv2.polylines(image, [draw_points], False, color, 5)
    return image

def draw_lines_ori(x, y, image, w_ratio, h_ratio, crop_start_h):

    method = 2
    for id in range(len(x)):
        x_l = x[id]
        x_list = [int(x / w_ratio) for x in x_l]
        # print("x_list", x_list)
        y_l = y[id]
        y_list = [int(y / h_ratio)+ crop_start_h for y in y_l]

        if method == 1:
            curve_fit(image, x_list, y_list) # before:color = id.
        else:
            curve_fit(image, x_list, y_list, p.color[id])
    return image

###############################################################
##
## calculate
## 
###############################################################
def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x,y):
        out_x.append((np.array(i)/ratio_w).tolist())
        out_y.append((np.array(j)/ratio_h).tolist())

    return out_x, out_y

def get_closest_point_along_angle(x, y, point, angle):
    index = 0
    for i, j in zip(x, y): 
        a = get_angle_two_points(point, (i,j))
        if abs(a-angle) < 0.1:
            return (i, j), index
        index += 1
    return (-1, -1), -1

def get_angle_two_points(p1, p2):
    del_x = p2[0] - p1[0]
    del_y = p2[1] - p1[1] + 0.000001    
    if p2[0] >= p1[0] and p2[1] > p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta /= 360.0
    elif  p2[0] > p1[0] and p2[1] <= p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta += 180
        theta /= 360.0
    elif  p2[0] <= p1[0] and p2[1] < p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta += 180
        theta /= 360.0
    elif  p2[0] < p1[0] and p2[1] >= p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta += 360
        theta /= 360.0
    
    return theta
    
def get_num_along_point(x, y, point1, point2, image=None): # point1 : source
    x = np.array(x)
    y = np.array(y)

    x = x[y<point1[1]]
    y = y[y<point1[1]]

    dis = np.sqrt( (x - point1[0])**2 + (y - point1[1])**2 )

    count = 0
    shortest = 1000
    target_angle = get_angle_two_points(point1, point2)
    for i in range(len(dis)):
        angle = get_angle_two_points(point1, (x[i], y[i]))
        diff_angle = abs(angle-target_angle)
        distance = dis[i] * math.sin( diff_angle*math.pi*2 )
        if distance <= 12:
            count += 1
            if distance < shortest:
                shortest = distance

    return count, shortest

def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y<point[1]]
    y = y[y<point[1]]

    dis = (x - point[0])**2 + (y - point[1])**2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i,j))

    return points

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

def sort_along_x(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(i, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y
