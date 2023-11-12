import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import dill
import copy
# other util


import torch
from torch import distributed as dist


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def fuse_conv_bn_relu(model):
    
    # Example of fuse_custom_config_dict
    # ~ fuse_custom_config_dict = {
        # ~ # Additional fuser_method mapping
        # ~ "additional_fuser_method_mapping": {
            # ~ (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
        # ~ },
    # ~ }
    modules_to_fuse = [ ['features.Conv2d0', 'features.BatchNorm2d0'], 
                        # ['features.Conv2d1', 'features.BatchNorm2d1'],
                        ['features.Conv2d2', 'features.BatchNorm2d2'],
                        # ['features.Conv2d3', 'features.BatchNorm2d3'],
                        ['features.Conv2d4', 'features.BatchNorm2d4'],
                        # ['features.Conv2d5', 'features.BatchNorm2d5'],
                        ['features.Conv2d6', 'features.BatchNorm2d6'],
                        # ['features.Conv2d7', 'features.BatchNorm2d7'],
                        ['features.Conv2d8', 'features.BatchNorm2d8'],
                        # ['features.Conv2d9', 'features.BatchNorm2d9'],
                        # ['features.Conv2d10', 'features.BatchNorm2d10'],
                        ]
    model.eval()
    fused_model = copy.deepcopy(model)
    
    fused_model.eval()
    fused_model = torch.quantization.fuse_modules(fused_model, modules_to_fuse, inplace=False)
    
    return fused_model

def prepare_qmodel(model, fuse_model=True, qat=True): #Do not fuse the model before training

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'x86' for server inference and 'qnnpack'
    # for mobile inference. Other quantization configurations such as selecting
    # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
    # can be specified here.
    # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
    # for server inference.
    # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    # set the qengine to control weight packing
    torch.backends.quantized.engine = 'qnnpack'
    # fuse the activations to preceding layers, where applicable
    # this needs to be done manually depending on the model architecture
    # model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32,
    #     [['conv', 'bn', 'relu']])
    if(fuse_model):
        model = fuse_conv_bn_relu(model)

    # Prepare the model for QAT. This inserts observers and fake_quants in
    # the model needs to be set to train for QAT logic to work
    # the model that will observe weight and activation tensors during calibration.
    model = torch.ao.quantization.prepare_qat(model.train())

    # run the training loop (not shown)
    # training_loop(model_fp32_prepared)
    return model
    


def accuracy_og(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # ~ correct.contiguous() #Fixes view issues
    for k in topk:
        # ~ correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        # ~ correct_k = correct[:k].reshape(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1, 1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))      
        print (result)

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print (result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)   


