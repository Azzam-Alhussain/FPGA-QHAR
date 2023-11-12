import os
import time
import argparse
import shutil
import dill

from PIL import Image
import matplotlib.pyplot as plt
from Two_Stream_Net import TwoStreamNet
from LoadUCF101Data import UCF101Data
#from LoadUCF101Data import testset
import LoadUCF101Data
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal

from random import seed
from random import random


import math
from scipy import signal
from PIL import Image
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
import random
# seed random number generator
seed(100)




parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--data-rgb', metavar='DIR', default='/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/BizhuWu/Two-Stream-Network-PyTorch-Simpnet-24/ucf24_all/RGB/jpegs_256/', help='path to dataset')
parser.add_argument('--data-flow', metavar='DIR', default='/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/BizhuWu/Two-Stream-Network-PyTorch-Simpnet-24/ucf24_all/OpFlow/tvl1_flow/', help='path to dataset')
                    
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

# setting gpu or cpu
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

# new a Two Stream model
# ~ twoStreamNet = TwoStreamNet().to(device)
twoStreamNet = TwoStreamNet(num_classes=24, qat=True).to('cpu')



import cv2, io, PIL
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from IPython.display import  Image as IImage
from IPython.display import display, clear_output

#################################################################################### 
# dictionary of actions
action_map = {
1 : 'BaseballPitch',
2 : 'Basketball',
3 : 'BasketballDunk',
4 : 'Biking',
5 : 'CricketBowling',
6 : 'Diving',
7 : 'Fencing',
8 : 'FloorGymnastics',
9 : 'GolfSwing',
10 : 'HorseRiding',
11 : 'LongJump',
12 : 'PoleVault',
13 : 'RopeClimbing',
14 : 'SalsaSpin',
15 : 'SkateBoarding',
16 : 'Skiing',
17 : 'Skijet',
18 : 'SoccerJuggling',
19 : 'Surfing',
20 : 'TennisSwing',
21 : 'TrampolineJumping',
22 : 'VolleyballSpiking',
23 : 'WalkingWithDog',
24 : 'WallPushups',
}


def your_model(capture_frame):
    # put your model here
    pred_result = np.random.randint(65535)
    return pred_result

font = ImageFont.truetype("Arial.ttf", 11)
import textwrap
def draw_multiple_line_text(image, text, font, text_color, text_start_height):

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    y_text = text_start_height
    lines = textwrap.wrap(text, width=40)
    for line in lines:
        # ~ line_width, line_height = font.getsize(line)
        _, _, line_width, line_height = font.getbbox(line)
        draw.text(((image_width - line_width) / 2, y_text), 
                  line, font=font, fill=text_color)
        y_text += line_height

def show_frame(capture_frame, pred_result, fps, show_meta, topk=None):
    
    #font = ImageFont.load_default()
    print("Shape is::", capture_frame.shape)
    h, w, _ = capture_frame.shape
    if show_meta:
        capture_frame[h-23:, :, :] = 0
    
    # predict value to string
    topk_class = []
    class_str = ""
    for c in topk:
        topk_class.append(f"{action_map[c]}")
        #class_str += f"{action_map[c]} "
    topk_class.sort()
    for c in topk_class:
        class_str += f"{c} "
    result = f' fps:{"{:.1f}".format(fps)}, Actions: {class_str}'
    #result = str(textwrap.wrap(result, width=20))
    frame = PIL.Image.fromarray(capture_frame)
    if show_meta:
        #draw = ImageDraw.Draw(frame)
        #draw.text((10,h-23), result, (255,255,0), font=font)
        draw_multiple_line_text(image=frame, text=result, font=font, text_color=(255,255,0), text_start_height=h-23)
    return frame

def showarray(capture_frame, fps=0, fmt='jpeg', show_meta=True, topk=None):    
    f = io.BytesIO()
    
    # put your model here
    #pred_result = your_model(capture_frame)
    
    frame = show_frame(capture_frame, 0, fps, show_meta, topk)
    
    clear_output(wait=True)
    frame.save(f, fmt)
    display(IImage(data=f.getvalue()))
    
    plt.figure()
    plt.axis('off')
    plt.imshow(frame)
    plt.show()
    





def optical_flow(I1g, I2g, window_size, tau=1e-2):
 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = window_size/2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade 
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(int(w), int(I1g.shape[0]-w)):
        for j in range(int(w), int(I1g.shape[1]-w)):
            Ix = fx[int(i-w):int(i+w+1), int(j-w):int(j+w+1)].flatten()
            Iy = fy[int(i-w):int(i+w+1), int(j-w):int(j+w+1)].flatten()
            It = ft[int(i-w):int(i+w+1), int(j-w):int(j+w+1)].flatten()
            #b = ... # get b here
            #A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = ... # get velocity here
            u[i,j]=nu[0]
            v[i,j]=nu[1]
 
    return (u,v)

'''
    This function implements the LK optical flow estimation algorithm with two frame data and without the pyramidal approach.
'''
def LK_OpticalFlow(Image1, Image2):
    
    I1 = np.array(Image1)
    I2 = np.array(Image2)
    S = np.shape(I1)
    
    t = 0.3 # choose threshold value

    #applying Gaussian filter of size 3x3 to eliminate any noise
    I1_smooth = cv2.GaussianBlur(I1 #input image
                                ,(3,3)	#shape of the kernel
                                ,0      #lambda
                                )
    I2_smooth = cv2.GaussianBlur(I2, (3,3), 0)

    '''
    let the filter in x-direction be Gx = 0.25*[[-1,1],[-1,1]]
    let the filter in y-direction be Gy = 0.25*[[-1,-1],[1,1]]
    let the filter in xy-direction be Gt = 0.25*[[1,1],[1, 1]]
    **1/4 = 0.25** for a 2x2 filter
    '''
        
    # First Derivative in X direction
    Ix = signal.convolve2d(I1_smooth,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,0.25],[-0.25,0.25]],'same')
    # First Derivative in Y direction
    Iy = signal.convolve2d(I1_smooth,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,-0.25],[0.25,0.25]],'same')
    # First Derivative in XY direction
    It = signal.convolve2d(I1_smooth,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,-0.25],[-0.25,-0.25]],'same')

    # finding the good features
    features = cv2.goodFeaturesToTrack(I1_smooth # Input image
    ,10000 # max corners
    ,0.01 # lambda 1 (quality)
    ,10 # lambda 2 (quality)
    )	

    feature = np.int0(features)

    plt.subplot(1,3,1)
    plt.title('Frame 1')
    plt.imshow(I1_smooth, cmap = cm.gray)
    plt.subplot(1,3,2)
    plt.title('Frame 2')
    plt.imshow(I2_smooth, cmap = cm.gray)#plotting the features in frame1 and plotting over the same
    for i in feature:
        x,y = i.ravel()
        cv2.circle(I1_smooth #input image
            ,(x,y) 			 #centre
            ,3 				 #radius
            ,0 			 #color of the circle
            ,-1 			 #thickness of the outline
            )

    #creating the u and v vector
    u = v = np.nan*np.ones(S)

    # Calculating the u and v arrays for the good features obtained n the previous step.
    for l in feature:
        j,i = l.ravel()
        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.
        
        IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]]) #The x-component of the gradient vector
        IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]]) #The Y-component of the gradient vector
        IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]]) #The XY-component of the gradient vector
        
        # Using the minimum least squares solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK)) # transpose of A
        LK = np.array(np.matrix.transpose(LK)) 
        
        A1 = np.dot(LK_T,LK) #Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2,LK_T)
        
        (u[i,j],v[i,j]) = np.dot(A3,IT) # we have the vectors with minimized square error

   
    return (u, v)  
    
    
def run_model_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed



if __name__ == '__main__':
    # set the test video id in testset
    

    # load the chekpoint file
    #state = torch.load('model/checkpoint-9000.pth')
    state = torch.load("/home/thanx/HDL-Workspace/Xilinx_SDK_Workspace/MyAcc/TWO_STREAM_SOC/PYTHON_MODELS/BizhuWu/Two-Stream-Network-PyTorch-Simpnet-24/best_checkpoint_new.pth", pickle_module=dill, map_location=torch.device('cpu'))
        
    #twoStreamNet.load_state_dict(state['model'])



    # send the model into the device, set its mode to eval
    twoStreamNet = twoStreamNet.to(device)
    twoStreamNet.eval()
    
    from torchsummary import summary
    input_shape1 = (3,256,256)
    input_shape2 = (20,256,256)
    summary(twoStreamNet, [input_shape1, input_shape2])
    
    for t in range(0, 500, 5):
        test_video_id = t
        print('Video Name:', LoadUCF101Data.TestVideoNameList[test_video_id])
        fps_list=[]
        max_fps = 0


        start = time.time()
        # get demo video's img and stacked optical flow images, show them in plt
        demo_RGB_img_path, demo_StackedOpticalFlow_imgs_path, label = testset.filenames[test_video_id]

        RGB_img = Image.open(demo_RGB_img_path)



        # send demo video's img and stacked optical flow images into the model
        RGB_img, opticalFlowStackedImg, actual_label = testset[test_video_id]

        RGB_img = RGB_img.to(device)
        opticalFlowStackedImg = opticalFlowStackedImg.to(device)

        RGB_img = RGB_img.unsqueeze(0)
        opticalFlowStackedImg = opticalFlowStackedImg.unsqueeze(0)

        output = twoStreamNet(RGB_img, opticalFlowStackedImg)



        # get the most possible result
        prob = F.softmax(output, dim=1)
        max_value, max_index = torch.max(prob, 1)
        pred_class = max_index.item()
        print('actual class is', LoadUCF101Data.classIndName[actual_label])
        print('actual class is::', actual_label)
        print('predicted class is',  LoadUCF101Data.classIndName[pred_class], "ClassID:", pred_class, ', probability is', round(max_value.item(), 6) * 100)
        
        
        time.sleep(4)
        end = time.time()
        sec = end-start
        # ~ fpga_sec = end_fpga - start
        fps_val = 1.0/sec
        # ~ fpga_fps_val = 1.0/fpga_sec
        fps_list.append(fps_val)
        # ~ fpga_fps_list.append(fpga_fps_val)
        if(max_fps < fps_val):
            max_fps = fps_val
        
        
      

        print(f"avg fps:{sum(fps_list)/len(fps_list)} the highest fps is: {max_fps}")
        # ~ print(f"avg fpga fps:{sum(fpga_fps_list)/len(fpga_fps_list)} the highest fpga fps is: {fpga_max_fps}")
        curr_frame = cv2.resize(np.array(Image.open(demo_RGB_img_path)), (256, 256))
        showarray(curr_frame, 100/sec, topk=[max_index.item()])
        
        
