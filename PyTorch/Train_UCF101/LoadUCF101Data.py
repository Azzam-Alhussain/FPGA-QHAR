import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image
import natsort




SAMPLE_FRAME_NUM = 10

# ~ v_BaseballPitch_g09_c06 66 1

classInd = {}
classIndName = {}
with open('classInd.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        line_info = line.split()
        idx = int(line_info[0])
        className = line_info[1]
        classInd[className] = idx
        classIndName[idx] = className

TrainVideoNameList = []
with open('trainlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        line_info = line.split()
        video_name = line_info[0]
        # ~ video_name = video_name.split('/')[1]
        TrainVideoNameList.append(video_name)

TestVideoNameList = []
with open('testlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        line_info = line.split()
        video_name = line_info[0]
        # ~ video_name = video_name.split('/')[1]
        TestVideoNameList.append(video_name)





class UCF101Data(Dataset):  # define a class named MNIST
    # read all pictures' filename
    def __init__(self, RBG_root, OpticalFlow_root, isTrain, transform=None, num_classes=24):
        # root: Dataset's filepath
        # classInd: dictionary (1 -> ApplyEyeMakeup)
        self.filenames = []
        self.transform = transform
        
        print("Initilizing data loader:RBG_root:", RBG_root)
        print("Initilizing data loader:OpticalFlow_root:", OpticalFlow_root)
        OpticalFlow_v_class_path = OpticalFlow_root + '/v/' 
        OpticalFlow_u_class_path = OpticalFlow_root + '/u/' 
        RGB_class_path = RBG_root + '/' 
        
        
        # only load train/test data using TrainVideoNameList/TestVideoNameList
        if isTrain:
            TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_v_class_path)).intersection(set(TrainVideoNameList)))
            print("Training data loader::")
        else:
            TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_v_class_path)).intersection(set(TestVideoNameList)))
            print("Testing data loader::")
        
        # ~ for i in range(0, num_classes):  
             
        # ~ for video_dir in os.listdir(OpticalFlow_v_class_path):
        video_dir_list = natsort.natsorted(os.listdir(OpticalFlow_v_class_path))
        for video_dir in video_dir_list:
            if video_dir in TrainOrTest_VideoNameList:
                single_OpticalFlow_v_video_path = OpticalFlow_v_class_path + '/' + video_dir
                single_OpticalFlow_u_video_path = OpticalFlow_u_class_path + '/' + video_dir
                signel_RGB_video_path = RGB_class_path + '/' + video_dir

                # load Optical Flow data
                v_frame_list = natsort.natsorted(os.listdir(single_OpticalFlow_v_video_path))
                u_frame_list = natsort.natsorted(os.listdir(single_OpticalFlow_u_video_path))
                #frame_list.sort(key=lambda x:int(x.split("_")[-2]))
                # generate a random frame idx (Notes: it must start from x)
                ran_frame_idx = np.random.randint(0, len(v_frame_list) - SAMPLE_FRAME_NUM + 1)
                while ran_frame_idx % 2 != 0:
                    ran_frame_idx = np.random.randint(0, len(v_frame_list) - SAMPLE_FRAME_NUM + 1)

                stacked_OpticalFlow_image_path = []
                for j in range(ran_frame_idx, ran_frame_idx + SAMPLE_FRAME_NUM):
                    OpticalFlow_v_image_path = single_OpticalFlow_v_video_path + '/' + v_frame_list[j]
                    stacked_OpticalFlow_image_path.append(OpticalFlow_v_image_path)
                    OpticalFlow_u_image_path = single_OpticalFlow_u_video_path + '/' + u_frame_list[j]
                    stacked_OpticalFlow_image_path.append(OpticalFlow_u_image_path)


                # load RGB data
                RGB_image_path = str()
                # ~ for image_fileName in os.listdir(signel_RGB_video_path):
                    # ~ RGB_image_path = signel_RGB_video_path + '/' + image_fileName
                # ~ if isTrain is False:
                    # ~ print("Class name:", video_dir.split("_")[1], "Class idx:", classInd[video_dir.split("_")[1]])
                
                rgb_frame_list = natsort.natsorted(os.listdir(signel_RGB_video_path))
                # ~ RGB_image_path = signel_RGB_video_path + '/' + rgb_frame_list[ran_frame_idx + int((SAMPLE_FRAME_NUM)/2)]
                RGB_image_path = signel_RGB_video_path + '/' + rgb_frame_list[ran_frame_idx + int((SAMPLE_FRAME_NUM - 1))]


                # (RGB_image_path, stacked_OpticalFlow_image_path, label)
                self.filenames.append((RGB_image_path, stacked_OpticalFlow_image_path, classInd[video_dir.split("_")[1]]))

        self.len = len(self.filenames)
        print("Data loader ready::", self.len)



    # Get a sample from the dataset & Return an image and it's label
    def __getitem__(self, index):
        RGB_image_path, stacked_OpticalFlow_image_path, label = self.filenames[index]


        # open the optical flow image
        stacked_OpticalFlow_image = torch.empty(SAMPLE_FRAME_NUM * 2, 256, 256)
        idx = 0

        for i in stacked_OpticalFlow_image_path:
            OpticalFlow_image = Image.open(i)

            # May use transform function to transform samples
            if self.transform is not None:
                OpticalFlow_image = self.transform(OpticalFlow_image)
            stacked_OpticalFlow_image[idx, :, :] = OpticalFlow_image[0, :, :]
            idx += 1


        # open the RGB image
        RGB_image = Image.open(RGB_image_path)

        # May use transform function to transform samples
        if self.transform is not None:
            RGB_image = self.transform(RGB_image)

        return RGB_image, stacked_OpticalFlow_image, label



    # get the length of dataset
    def __len__(self):
        return self.len







