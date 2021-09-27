from hourglass_network import lane_detection_network
import torch
from thop import profile
from thop import clever_format

model = lane_detection_network()

input = torch.randn(1, 3, 256, 256)

macs, params = profile(model, inputs=(input, ))
print(macs, params)
macs, params = clever_format([macs, params], "%.3f")

print('macs:{}\nparams:{}'.format(macs, params))
print("FINISHED!")