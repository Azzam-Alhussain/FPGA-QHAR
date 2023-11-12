import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import math
import collections
import numpy as np
import torch

from simplenet import simplenet
from Two_Stream_Net import TwoStreamNet

import dill
import pickle
import json
import copy

from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=10_000_000)

def model_dict2json(model):
    
    state_dict = model.state_dict()
    # as requested in comment

    # ~ with open('model.json', 'w') as file:
        # ~ file.write(json.dumps()) # use `json.loads` to do the reverse
        
    # ~ with open('file.txt', 'w') as file:
        # ~ file.write(pickle.dumps(state_dict)) # use `pickle.loads` to do the reverse
        
    with open('stat_dump.txt', 'w') as f:
        print(state_dict, file=f)
        
        
    with open('formatedDict_rgb.txt', 'w') as f:
        print("{",file=f)
        for i in [0,1,3,4,6]:
            print(f'module.features.Conv2d{i}', ":{scale:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.scale'].numpy(),
                  ", zero_point:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.zero_point'].numpy(), 
                  ", weight:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.weight'].int_repr().numpy(), "},\n\n",  file=f)
            print(['*']*400, file=f)
        print("module.classifier:0.{ scale:", state_dict["rgb_branch.RGB_stream.classifier.1.scale"].numpy(), 
               ", zero_point:", state_dict["rgb_branch.RGB_stream.classifier.1.zero_point"].numpy(), 
               ", weights:", state_dict["rgb_branch.RGB_stream.classifier.1._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        print("}",file=f)
        
        print("module.classifier:2.{ scale:", state_dict["rgb_branch.RGB_stream.classifier.3.scale"].numpy(), 
               ", zero_point:", state_dict["rgb_branch.RGB_stream.classifier.3.zero_point"].numpy(), 
               ", weights:", state_dict["rgb_branch.RGB_stream.classifier.3._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        print("}",file=f)
        
        
    
    with open('formatedDict_opflow.txt', 'w') as f:
        print("{",file=f)
        for i in [0,1,3,4,6]:
            print(f'.features.Conv2d{i}', ":{scale:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.scale'].numpy(),
                  ", zero_point:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.zero_point'].numpy(), 
                  ", weight:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.weight'].int_repr().numpy(), "},\n\n",  file=f)
            print(['*']*400, file=f)
        print("module.classifier.1:{ scale:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1.scale"].numpy(), 
               ", zero_point:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1.zero_point"].numpy(), 
               ", weights:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        print("}",file=f)
        
        print("module.classifier.3:{ scale:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3.scale"].numpy(), 
               ", zero_point:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3.zero_point"].numpy(), 
               ", weights:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        print("}",file=f)


def make_temporal_pickle(state_dict):
    new_dict = {}
    with open('extracted_models/temporal_fm.txt', 'w') as f:
        print("{",file=f)
        for i in [0,2,4,6,8]:
            print(f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}', ":{scale:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.scale'].item(),
                  ", zero_point:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.zero_point'].item(), 
                  ", weight:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.weight'].int_repr().numpy(), "},\n\n",  file=f)
            
            print(['*']*400, file=f)
            print(f"opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i} layer size:", state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.weight'].int_repr().numpy().shape)   
               
            new_dict[f'features.Conv2d{i}']={}
            new_dict[f'features.Conv2d{i}']['xscale'] = state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.scale'].item()
            new_dict[f'features.Conv2d{i}']['wscale'] = state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.weight'].q_scale()
            new_dict[f'features.Conv2d{i}']['scale'] = new_dict[f'features.Conv2d{i}']['xscale'] * new_dict[f'features.Conv2d{i}']['wscale']
            new_dict[f'features.Conv2d{i}']['zero_point'] = state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.zero_point'].item()
            new_dict[f'features.Conv2d{i}']['weight'] = state_dict[f'opticalFlow_branch.OpticalFlow_stream.features.Conv2d{i}.weight'].int_repr().numpy()
            new_dict[f'features.Conv2d{i}']['w_zeropoint'] = 0
            # ~ minimum = np.min(new_dict[f'features.Conv2d{i}']['weight'])
            # ~ if minimum < 0:
                # ~ print(f"The minimum for features.Conv2d{i} is:", minimum)
                # ~ new_dict[f'features.Conv2d{i}']['weight'] = (new_dict[f'features.Conv2d{i}']['weight'] - minimum).astype('uint8')
                # ~ new_dict[f'features.Conv2d{i}']['w_zeropoint'] = minimum * (-1)
            # ~ else:
                # ~ new_dict[f'features.Conv2d{i}']['w_zeropoint'] = 0
                
                
            if(i>1):
                new_dict[f'features.Conv2d{i-2}']['xnext_zeropoint'] = new_dict[f'features.Conv2d{i}']['zero_point']
            
            
                
            
            
        print("opticalFlow_branch.OpticalFlow_stream.classifier.1:{ scale:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1.scale"].item(), 
               ", zero_point:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1.zero_point"].item(), 
               ", weights:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        print("}",file=f)
        print("opticalFlow_branch.OpticalFlow_stream.classifier 1 layer size:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.1._packed_params._packed_params"][0].int_repr().numpy().shape)
        
        # ~ print("opticalFlow_branch.OpticalFlow_stream.classifier.3:{ scale:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3.scale"].item(), 
               # ~ ", zero_point:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3.zero_point"].item(), 
               # ~ ", weights:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        # ~ print("}",file=f)
        
        print("opticalFlow_branch.OpticalFlow_stream.classifier.3:{ scale:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3.scale"].item(), 
               ", zero_point:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3.zero_point"].item(), 
               ", weights:", torch.dequantize(state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params"][0]).numpy(), "},\n\n",  file=f)
        print("}",file=f)
        
        print("opticalFlow_branch.OpticalFlow_stream.classifier 3 layer size:", state_dict["opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params"][0].int_repr().numpy().shape)   
               
        new_dict['classifier.1']={}
        new_dict['classifier.1']['xscale'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.1.scale'].item()
        new_dict['classifier.1']['wscale'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.1._packed_params._packed_params'][0].q_scale()
        new_dict['classifier.1']['scale'] = new_dict['classifier.1']['xscale'] * new_dict['classifier.1']['wscale']
        new_dict['classifier.1']['zero_point'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.1.zero_point'].item()
        new_dict['classifier.1']['w_zeropoint'] = 0

        new_dict['classifier.1']['weight'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.1._packed_params._packed_params'][0].int_repr().numpy()
        # ~ new_dict['classifier.1']['weight'] = torch.dequantize(state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.1._packed_params._packed_params'][0])
        
        # ~ minimum = np.min(new_dict['classifier.1']['weight'])
        # ~ if minimum < 0:
            # ~ print("The minimum for class 1 is:", minimum)
            # ~ new_dict['classifier.1']['weight'] = (new_dict['classifier.1']['weight'] - minimum).astype('uint8')
            # ~ new_dict['classifier.1']['w_zeropoint'] = -minimum
        # ~ else:
            # ~ new_dict['classifier.1']['w_zeropoint'] = 0
               
               
        new_dict['classifier.3']={}
        new_dict['classifier.3']['wscale'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params'][0].q_scale()
        new_dict['classifier.3']['xscale'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.3.scale'].item()
        new_dict['classifier.3']['scale'] = new_dict['classifier.3']['xscale'] * new_dict['classifier.3']['wscale'] 
        new_dict['classifier.3']['zero_point'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.3.zero_point'].item()
        new_dict['classifier.3']['w_zeropoint'] = 0
        new_dict['classifier.3']['xnext_zeropoint'] = 0
        #new_dict['classifier.3']['weight'] = state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params'][0].int_repr().numpy()
        new_dict['classifier.3']['weight'] = torch.dequantize(state_dict['opticalFlow_branch.OpticalFlow_stream.classifier.3._packed_params._packed_params'][0]).numpy()
               
        new_dict[f'features.Conv2d8']['xnext_zeropoint'] = new_dict['classifier.1']['zero_point']
        new_dict['classifier.1']['xnext_zeropoint'] = new_dict['classifier.3']['zero_point']
        
    
    
    with open("extracted_models/temperal_model.txt", 'w') as f:
        print(new_dict, file=f)
    
    with open("extracted_models/temperal_model.pickle", "wb") as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        
def make_spatial_pickle(state_dict):
    new_dict = {}
    with open('extracted_models/spatial_fm.txt', 'w') as f:
        print("{",file=f)
        for i in [0,2,4,6,8]:
            print(f'rgb_branch.RGB_stream.features.Conv2d{i}', ":{scale:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.scale'].item(),
                  ", zero_point:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.zero_point'].item(), 
                  ", weight:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.weight'].int_repr().numpy(), "},\n\n",  file=f)
            
            print(['*']*400, file=f)
            print(f"rgb_branch.RGB_stream.features.Conv2d{i} layer size:", state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.weight'].int_repr().numpy().shape)   
               
            new_dict[f'features.Conv2d{i}']={}
            new_dict[f'features.Conv2d{i}']['xscale'] = state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.scale'].item()
            new_dict[f'features.Conv2d{i}']['wscale'] = state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.weight'].q_scale()
            new_dict[f'features.Conv2d{i}']['scale'] = new_dict[f'features.Conv2d{i}']['xscale'] * new_dict[f'features.Conv2d{i}']['wscale']
            new_dict[f'features.Conv2d{i}']['zero_point'] = state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.zero_point'].item()
            new_dict[f'features.Conv2d{i}']['weight'] = state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.weight'].int_repr().numpy()
            new_dict[f'features.Conv2d{i}']['w_zeropoint'] = 0
            # ~ minimum = np.min(new_dict[f'features.Conv2d{i}']['weight'])
            # ~ if minimum < 0:
                # ~ print(f"The minimum for features.Conv2d{i} is:", minimum)
                # ~ new_dict[f'features.Conv2d{i}']['weight'] = (new_dict[f'features.Conv2d{i}']['weight'] - minimum).astype('uint8')
                # ~ new_dict[f'features.Conv2d{i}']['w_zeropoint'] = -minimum
            # ~ else:
                # ~ new_dict[f'features.Conv2d{i}']['w_zeropoint'] = 0
            
            if(i>1):
                new_dict[f'features.Conv2d{i-2}']['xnext_zeropoint'] = new_dict[f'features.Conv2d{i}']['zero_point']
                
            print(f'rgb_branch.RGB_stream.features.Conv2d{i}.weight zeropint::', state_dict[f'rgb_branch.RGB_stream.features.Conv2d{i}.weight'].q_zero_point())
            
            
        print("rgb_branch.RGB_stream.classifier.1:{ scale:", state_dict["rgb_branch.RGB_stream.classifier.1.scale"].item(), 
               ", zero_point:", state_dict["rgb_branch.RGB_stream.classifier.1.zero_point"].item(), 
               ", weights:", state_dict["rgb_branch.RGB_stream.classifier.1._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
        print("}",file=f)
        print("rgb_branch.RGB_stream.classifier 1 layer size:", state_dict["rgb_branch.RGB_stream.classifier.1._packed_params._packed_params"][0].int_repr().numpy().shape)  
        
        # ~ print("rgb_branch.RGB_stream.classifier.3:{ scale:", state_dict["rgb_branch.RGB_stream.classifier.3.scale"].item(), 
               # ~ ", zero_point:", state_dict["rgb_branch.RGB_stream.classifier.3.zero_point"].item(), 
               # ~ ", weights:", state_dict["rgb_branch.RGB_stream.classifier.3._packed_params._packed_params"][0].int_repr().numpy(), "},\n\n",  file=f)
               
        print("rgb_branch.RGB_stream.classifier.3:{ scale:", state_dict["rgb_branch.RGB_stream.classifier.3.scale"].item(), 
               ", zero_point:", state_dict["rgb_branch.RGB_stream.classifier.3.zero_point"].item(), 
               ", weights:", torch.dequantize(state_dict["rgb_branch.RGB_stream.classifier.3._packed_params._packed_params"][0]).numpy(), "},\n\n",  file=f)
               
        print("}",file=f)
        print("rgb_branch.RGB_stream.classifier 3 layer size:", state_dict["rgb_branch.RGB_stream.classifier.3._packed_params._packed_params"][0].int_repr().numpy().shape)  
        
               
        new_dict['classifier.1']={}
        new_dict['classifier.1']['xscale'] = state_dict['rgb_branch.RGB_stream.classifier.1.scale'].item()
        new_dict['classifier.1']['wscale'] = state_dict['rgb_branch.RGB_stream.classifier.1._packed_params._packed_params'][0].q_scale()
        new_dict['classifier.1']['scale'] = new_dict['classifier.1']['xscale'] * new_dict['classifier.1']['wscale']
        new_dict['classifier.1']['zero_point'] = state_dict['rgb_branch.RGB_stream.classifier.1.zero_point'].item()
        new_dict['classifier.1']['w_zeropoint'] = 0
        
        new_dict['classifier.1']['weight'] = state_dict['rgb_branch.RGB_stream.classifier.1._packed_params._packed_params'][0].int_repr().numpy()
        # ~ new_dict['classifier.1']['weight'] = torch.dequantize(state_dict['rgb_branch.RGB_stream.classifier.1._packed_params._packed_params'][0])
        
        # ~ minimum = np.min(new_dict['classifier.1']['weight'])
        # ~ if minimum < 0:
            # ~ print("The minimum for class 1 is:", minimum)
            # ~ new_dict['classifier.1']['weight'] = (new_dict['classifier.1']['weight'] - minimum).astype('uint8')
            # ~ new_dict['classifier.1']['w_zeropoint'] = -minimum
        # ~ else:
            # ~ new_dict['classifier.1']['w_zeropoint'] = 0
        
        new_dict['classifier.3']={}
        new_dict['classifier.3']['xscale'] = state_dict['rgb_branch.RGB_stream.classifier.3.scale'].item()
        new_dict['classifier.3']['wscale'] = state_dict['rgb_branch.RGB_stream.classifier.3._packed_params._packed_params'][0].q_scale()
        new_dict['classifier.3']['scale'] = new_dict['classifier.3']['xscale'] * new_dict['classifier.3']['wscale']
        new_dict['classifier.3']['zero_point'] = state_dict['rgb_branch.RGB_stream.classifier.3.zero_point'].item()
        new_dict['classifier.3']['w_zeropoint'] = 0
        new_dict['classifier.3']['xnext_zeropoint'] = 0
        # ~ new_dict['classifier.3']['weight'] = state_dict['rgb_branch.RGB_stream.classifier.3._packed_params._packed_params'][0].int_repr().numpy()  
        new_dict['classifier.3']['weight'] = torch.dequantize(state_dict['rgb_branch.RGB_stream.classifier.3._packed_params._packed_params'][0]).numpy()     
        
        new_dict[f'features.Conv2d8']['xnext_zeropoint'] = new_dict['classifier.1']['zero_point']
        new_dict['classifier.1']['xnext_zeropoint'] = new_dict['classifier.3']['zero_point']
        
        
    
    
    with open("extracted_models/spatial_pickle.txt", 'w') as f:
        print(new_dict, file=f)
    
    with open("extracted_models/spatial_model.pickle", "wb") as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

           
    
def model2dict_pickle(model):
    
    state_dict = model.state_dict()
    new_dict = {}
    with open("extracted_models/combo_model_dict_dump.txt", 'w') as f:
        print(state_dict, file=f)
        
    make_spatial_pickle(state_dict)    
    make_temporal_pickle(state_dict)  
    
    
def make_pickels(param_path="temperal_model.pickle"):
    stat_dict = pickle.load(open(param_path, "rb"))
    param_list = list(stat_dict.values())
    key_list  = list(stat_dict.keys())
    #{'conv2': {'qweight': array([[[[xx]]]], dtype=float32), 'scale': 0.001070737955160439, 'w_zeropoint': 116, 'x_zeropoint': 7, 'xnext_zeropoint': 6}, ... }
    #print("Vars:::", stat_dict.values()) 
    #exit()
    # ~ l_idx = 0
    with open("newfiles/"+param_path+'extracted.txt', 'w') as f:
        print(stat_dict, file=f)
    
    
    for key, value in stat_dict.items():
        scaler = random()*0.01
        # ~ stat_dict[key]['qweight'] = stat_dict[key]['qweight']*scaler 
        stat_dict[key]['w_scale'] = scaler 
        
        value = stat_dict[key]['qweight']*scaler        
        stat_dict[key]['weight'] = value
        del stat_dict[key]['qweight']
        
        
        value = stat_dict[key]['x_zeropoint']   
        stat_dict[key]['zeropoint'] = value
        del stat_dict[key]['x_zeropoint']
        
        
        # ~ value = stat_dict[key]['xnext_zeropoint']        
        # ~ stat_dict[key]['xnext_zeropoint'] = value
        if 'xnext_zeropoint' not in stat_dict[key].keys():
            continue
        
        del stat_dict[key]['xnext_zeropoint']
        
        
        
        
        

    
    # ~ pickle.save(stat_dict, open("new_mod_file.pickle", "wb")) pickle only dumps
    
    
    with open("newfiles/"+param_path+'.txt', 'w') as f:
        print(stat_dict, file=f)
    
    with open("newfiles/"+param_path, "wb") as handle:
        pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
       
    


def convert_model(model):
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 20, 256, 256)

    torch.onnx.export(model, (x, y), "./int8_model.onnx", opset_version=18)
    #model_ir = mo.convert_model(input_model="./int8_model.onnx", input_shape=[-1, 3, 224, 224])

    #serialize(model_ir, "./int8_model.xml")
    
    
def convert_caffe2_onnx(model):
    
    torch.backends.quantized.engine = "qnnpack"
    qconfig = torch.quantization.default_qconfig
    # ~ model = ConvModel()
    model.qconfig = qconfig
    model = torch.quantization.prepare(model)
    model = torch.quantization.convert(model)

    # ~ x_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
    # ~ x = torch.from_numpy(x_numpy).to(dtype=torch.float)
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 20, 256, 256)
    outputs = model(x,y)
    input_names = ["rgb", "opf"]
    outputs = model(x,y)

    traced = torch.jit.trace(model, (x, y))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    buf.seek(0)

    model = torch.jit.load(buf)
    f = io.BytesIO()
    torch.onnx.export(model, (x, y), f, input_names=input_names, example_outputs=outputs,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    f.seek(0)

    onnx_model = onnx.load(f)
    
    
model = TwoStreamNet(num_classes=24, qat=True).to('cpu')
checkpoint = torch.load("/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/BizhuWu/Two-Stream-Network-PyTorch-Simpnet-24/best_checkpoint_new.pth", pickle_module=dill, map_location=torch.device('cpu'))
        
        

    
model_state = checkpoint['model']

model.load_state_dict(model_state)
# ~ convert_model(model)
# ~ convert_caffe2_onnx(model)
from torchsummary import summary
input_shape1 = (3,256,256)
input_shape2 = (20,256,256)
summary(model, [input_shape1, input_shape2])

model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')

    

    
print("Qconfig::", model.qconfig)    
model = torch.ao.quantization.convert(model)

#model_dict2json(model)
model2dict_pickle(model)

