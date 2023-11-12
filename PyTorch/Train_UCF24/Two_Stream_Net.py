import torch.nn as nn
import torchvision.models as models
import LoadUCF101Data
from simplenet import simplenet
from utils import *


class OpticalFlowStreamNet(nn.Module):
    def __init__(self, in_chans=10*2, num_classes=24, qat=True):
        super(OpticalFlowStreamNet, self).__init__()

        #self.OpticalFlow_stream = models.resnet50()
        self.OpticalFlow_stream = simplenet(network_idx=0, mode=0, num_classes=num_classes, in_chans=in_chans, qat=bool(qat))
        if(bool(qat)):
            self.OpticalFlow_stream = prepare_qmodel(self.OpticalFlow_stream, fuse_model=qat, qat = qat)  #From utils
        # ~ self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # ~ self.OpticalFlow_stream.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        streamOpticalFlow_out = self.OpticalFlow_stream(x)
        return streamOpticalFlow_out



class RGBStreamNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=24, qat=True):
        super(RGBStreamNet, self).__init__()

        # ~ self.RGB_stream = models.resnet50(pretrained=True)
        self.RGB_stream = simplenet(network_idx=0, mode=0, num_classes=num_classes, in_chans=in_chans, qat=bool(qat))
        if(bool(qat)):
            self.RGB_stream = prepare_qmodel(self.RGB_stream, fuse_model=qat, qat = qat) #From utils
        # ~ self.RGB_stream.fc = nn.Linear(in_features=2048, out_features=24)

    def forward(self, x):
        streamRGB_out = self.RGB_stream(x)
        return streamRGB_out



class TwoStreamNet(nn.Module):
    def __init__(self, num_classes=24, qat=True):
        super(TwoStreamNet, self).__init__()

        self.rgb_branch = RGBStreamNet(num_classes=num_classes, qat=qat)
        self.opticalFlow_branch = OpticalFlowStreamNet(num_classes=num_classes, qat=qat)

    def forward(self, x_rgb, x_opticalFlow):
        rgb_out = self.rgb_branch(x_rgb)
        opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
        return rgb_out + opticalFlow_out
