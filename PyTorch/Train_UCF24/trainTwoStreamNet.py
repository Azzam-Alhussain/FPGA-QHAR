import os
import time
import argparse
import shutil
import dill
# ~ from LoadUCF101Data import trainset_loader, testset_loader
from LoadUCF101Data import UCF101Data
from Two_Stream_Net import TwoStreamNet
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from random import seed
from random import random
# seed random number generator
seed(100)




parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--data-rgb', metavar='DIR', default='/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/DATASET_UFC101/RGB/jpegs_256/', help='path to dataset')
parser.add_argument('--data-flow', metavar='DIR', default='/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/DATASET_UFC101/OpFlow/tvl1_flow/', help='path to dataset')
                    
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='full',
                    choices=["full", "rgb", "flow"],
                    help='modality: full | rgb | flow')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')

parser.add_argument('--device', default='cpu',
                    choices=["cpu", "cuda"],
                    help='device: cpu | cuda')
                    
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=4, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 8)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=12, type=int,
                    metavar='N', help='print frequency (default: 12)')
parser.add_argument('--save-freq', default=72, type=int,
                    metavar='N', help='save frequency (default: 20)')
parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--isz', default=256, type=int, metavar='N', help='image square dim (default: 32)')
parser.add_argument('--qat', default=False, type=bool, metavar='N', help='Quantization (default: True)')
parser.add_argument('--num-classes', default=24, type=int, metavar='N', help='Quantization (default: 24)')

args = parser.parse_args()

EPOCH = args.epochs
LEARNING_RATE = args.lr
MOMENTUM = args.momentum
SAVE_INTERVAL = args.save_freq

TRAIN_BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.batch_size


# ~ parser.add_argument('--data', metavar='DIR', default='/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/DATASET_UFC101/RGB/jpegs_256/', help='path to dataset')
# ~ parser.add_argument('--data', metavar='DIR', default='/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/DATASET_UFC101/OpFlow/tvl1_flow/', help='path to dataset')
      
      
      
 
 
      



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# define the transformation
# PIL images -> torch tensors [0, 1]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])


# load the UCF101 training dataset
trainset = UCF101Data(
    RBG_root= args.data_rgb,
    OpticalFlow_root= args.data_flow,
    isTrain=True,
    transform=transform
)

# divide the dataset into batches
trainset_loader = DataLoader(
    trainset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)



# load the UCF101 testing dataset
testset = UCF101Data(
    RBG_root= args.data_rgb,
    OpticalFlow_root= args.data_flow,
    isTrain=False,
    transform=transform
)

# divide the dataset into batches
testset_loader = DataLoader(
    testset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


    
twoStreamNet = TwoStreamNet(num_classes=args.num_classes, qat=True).to(device)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # ~ print("Size of correct::", correct.size())
        # ~ correct_k = correct[:k].view(-1).float().sum(0)
        # ~ correct_k = correct[:k].float().sum(0)
        # ~ correct_k = correct[:k].reshape(1, -1).float().sum(0) wrong
        correct_k = correct[:k].reshape(-1, 1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        #print("Res::", res, "correctk size", correct_k.size(), "k::", k)
    return res

optimizer = optim.SGD(
    params=twoStreamNet.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)



def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

    

def train(epoch, save_interval):
    global args
    iteration = 0
    best_prec1 = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()
    for i in range(epoch):
        twoStreamNet.train()
        end = time.time()
        for index, data in enumerate(trainset_loader):
            data_time.update(time.time() - end)
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = twoStreamNet(RGB_images, OpticalFlow_images)
            loss = F.cross_entropy(output, label)
            
            
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            prec1, prec3 = accuracy(output.data, label, topk=(1, 3))

            losses.update(loss.item(), RGB_images.size(0))
            top1.update(prec1[0], RGB_images.size(0))
            top3.update(prec3[0], RGB_images.size(0))

            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint(args.resume +'/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)     # OpticalFlow_ResNetModel

            iteration += 1

            #print("Loss: " + str(loss.item()), "Acc@1:", prec1, "Acc@3:", prec3)
            
            if index % args.print_freq == 0:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, epoch, index, len(trainset_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top3=top3))
            
            # with open('log.txt', 'a') as f:
            #     f.write('Epoch: [{0}/{1}][{2}/{3}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #       'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
            #        i, epoch, index, len(trainset_loader), batch_time=batch_time,
            #        data_time=data_time, loss=losses, top1=top1, top3=top3) + "\n")

        prec1 = test(i+1)
        
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (i + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint(args.resume + '/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)
        elif(is_best):
          torch.save(twoStreamNet, args.resume+"/model_best_full.pth", pickle_module=dill) #only saves dicts
          save_checkpoint(args.resume + '/best_checkpoint.pth', twoStreamNet, optimizer)

    save_checkpoint(args.resume + '/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)


def test(i_epoch):

    twoStreamNet.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    correct = 0
    with torch.no_grad():
        for index, data in enumerate(testset_loader):
            end = time.time()
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            output = twoStreamNet(RGB_images, OpticalFlow_images)

            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()
            prec1, prec3 = accuracy(output.data, label, topk=(1, 3))

            top1.update(prec1[0], RGB_images.size(0))
            top3.update(prec3[0], RGB_images.size(0))            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # print("Accuracy: Prec@1:" + str(correct*1.0*100/len(testset_loader.dataset)) + " Prec@3:" + str(prec3.item()))
    print("Accuracy: Prec@1:" + str(top1.avg.item()) + " Prec@3:" + str(top3.avg.item()))
    # with open('log.txt', 'a') as f:
    #     f.write("Epoch " + str(i_epoch) + "'s Accuracy: Prec@1:" + str(correct*1.0*100/len(testset_loader.dataset)) + " Prec@3:" + str(prec3.item()) +"\n")
        
    return top1.avg.item()

if __name__ == '__main__':
    train(EPOCH, SAVE_INTERVAL)
