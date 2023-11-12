import cv2, time, configparser
import ctypes
import numpy as np

from pynq import Overlay
from pynq import allocate
import nnet.fpga_nn as fpga_nn
from nnet.accelerator import CNN_accelerator

def make_layers_new(config, in_channel=3, accelerator=None):
    assert config is not None    
    in_height = int(config["DataConfig"]["image_height"])
    in_width = int(config["DataConfig"]["image_width"])
    in_channel = in_channel

    assert accelerator is not None
    acc = accelerator

    layers = []
    #Conv(output channel, input channel, input height, input width, kerSize, stride)
    layers += [fpga_nn.Conv2D(64, in_channel, in_height, in_width, ker = 3, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height), int(in_width), ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2D(64, 64, int(in_height/2), int(in_width/2), ker = 3, accelerator=acc)]
    
    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/2), int(in_width/2), ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2D(64, 64, int(in_height/4), int(in_width/4), ker = 3, accelerator=acc)]

#     layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/32), int(in_width/32), ker = 3, poolWin = 2, accelerator=acc)]

    # conv output size = (8,8,512)
    layers += [fpga_nn.Flatten(int(in_height/4),int(in_width/4), 64)]
    layers += [fpga_nn.Linear(1024,int(in_height/4)*int(in_width/4)*64)]
    layers += [fpga_nn.Linear(24,1024, quantize = False)]
    

    return layers

def make_layers(config, in_channel=3, accelerator=None):
    assert config is not None    
    in_height = int(config["DataConfig"]["image_height"])
    in_width = int(config["DataConfig"]["image_width"])
    in_channel = in_channel

    assert accelerator is not None
    acc = accelerator

    layers = []
    #Conv(output channel, input channel, input height, input width, kerSize, stride)
    layers += [fpga_nn.Conv2DPool(32, in_channel, in_height, in_width, ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 32, int(in_height/2), int(in_width/2), ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/4), int(in_width/4), ker = 3, poolWin = 2, accelerator=acc)]
    
    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/8), int(in_width/8), ker = 3, poolWin = 2, accelerator=acc)]

    layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/16), int(in_width/16), ker = 3, poolWin = 2, accelerator=acc)]

#     layers += [fpga_nn.Conv2DPool(64, 64, int(in_height/32), int(in_width/32), ker = 3, poolWin = 2, accelerator=acc)]

    # conv output size = (8,8,512)
    layers += [fpga_nn.Flatten(int(in_height/32), int(in_width/32), 64)]
    layers += [fpga_nn.Linear(1024,int(in_height/32)*int(in_width/32)*64)]
    layers += [fpga_nn.Linear(24,1024, quantize = False)]
    
  
    return layers

class UCF24SimpNetSpatial(CNN_accelerator):
    def __init__(self, config, layers, params_path = None):
        super(UCF24SimpNetSpatial, self).__init__(config, is_spatial=True)
        self.layers = layers
        self.params_path = params_path

        # initialize weight for each layer
        self.init_weight(params_path)

        # copy weight data to hardware buffer 
        self.load_parameters();

def ucf24_simpnet_spatial(model_path, config, accelerator):
    layers = make_layers(config, in_channel=3, accelerator=accelerator)
    params_path = model_path
    model = UCF24SimpNetSpatial(config, layers, params_path = params_path)
    return model        
        


if __name__=='__main__':
    config_path = './files/config.config'
    model_path = './files/params/ucf24_simpnet/model.pickle'
    config = configparser.ConfigParser()   
    config.read(config_path)

    overlay = Overlay(config["FPGAConfig"]["bitstream_path"])

    (in_height, in_width, in_channel) = \
            (int(config["DataConfig"]["image_height"]),int(config["DataConfig"]["image_width"]),3)


    cnn_acc0 = CNN_accelerator(config, overlay.DoCompute_0)

    ucf24_spatial_model = ucf24_simpnet_spatial(model_path, config, cnn_acc0)

    #########################################################################
