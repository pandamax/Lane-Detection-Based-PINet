#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from util_hourglass import *

class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        self.channel = 128 # 
        self.resizing = resize_layer(3, self.channel)
        #feature extraction
        self.layer1 = hourglass_block(self.channel, self.channel)
        self.layer2 = hourglass_block(self.channel, self.channel)
        # self.layer3 = hourglass_block(channel, channel)
        # self.layer4 = hourglass_block(channel, channel)

    def forward(self, inputs):
        #feature extraction
        out = self.resizing(inputs)
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)   
        # result3, out, feature3 = self.layer3(out)
        # result4, out, feature4 = self.layer4(out)

        return [result1, result2], [feature1, feature2]
