'''
visualization.

testd on windows.

datasets structure:
VIL-100
    |----Annotations
    |----data
           |----train_converted.json
           |----test_converted.json
           |----...
                                  
    |----JPEGImages
    |----Json
    |----train.json
'''
import random
import cv2
import numpy as np
import json
import os


def vis_converted(dataset_dir, color):
    
    test_image = cv2.imdecode(np.fromfile(dataset_dir + test_data[i]['raw_file'], dtype=np.uint8), -1)
    y_samples = test_data[i]['h_samples']
    for id, lane in enumerate(test_data[i]['lanes']):
        for pts in zip(lane, y_samples):
            if pts[0] != -2: #remove -2
                test_image = cv2.circle(test_image, pts, 3, color[id], -1)
    return test_image         
       
        
if __name__ == '__main__':
    # choose datasets category from:'train','test'
    datasets_category = 'train'  
    # dataset_dir = '../dataset/VIL-100' #path to VIL dataset.
    dataset_dir = '../VIL-100' # path to VIL dataset.

    save_dir = '{}/vis_converted'.format(dataset_dir)
    label_file = '{}/data/{}_converted.json'.format(dataset_dir, datasets_category)

    # color
    color = [(218,112,214), (255, 0, 0), (0, 255, 0), (0, 0, 255), 
             (255, 255, 0), (255, 0, 255),(255, 0, 100), (0, 255, 100),(0, 255, 100)]
    
    sample_num = 0.01 # 0-1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    test_data = []
    print("Data Loading...")

    with open(label_file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if random.random() < sample_num:
                jsonString = json.loads(line)
                test_data.append(jsonString)

    size_test = len(test_data)

    for i in range(size_test):
        print("Now deal with {}".format(test_data[i]['raw_file']))
        image = vis_converted(dataset_dir, color)
        cv2.imwrite(save_dir + '/vis_{}.png'.format(test_data[i]['raw_file'].split('/')[-1].split('.jpg')[0]), image)
        # cv2.imencode('.png',mask)[1].tofile('{}\{}.png'.format(mask_dir,label_file.split('.txt')[0]))
    
    print("Done!")
    






