from LK_optical_flow.utils import showarray
from two_stream import Two_stream

import os, sys, cv2, time
old_stdout = sys.stdout

import configparser
import numpy as np

from contextlib import contextmanager
import ctypes
import multiprocessing as mp

from multiprocessing import Process

import random
import glob




output = mp.Array(ctypes.c_float, 101)
ps_two_stream = None #Process(target=self.spatial_job, args=(input_frame, 0))        
        
## configs
config = configparser.ConfigParser()
config.read('./files/config.config')

size = (ch,h,w) = (config['OpticalFlow']['channel'],
                   int(config['DataConfig']['image_height']),int(config['DataConfig']['image_width'])) #(20,256,256)




two_stream = Two_stream('./files/config.config')


@contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target
        




def action_recogn_job(curr_frame, prev_frame, output):
    global two_stream
    
    with silence_stdout():
        output[:] = two_stream(curr_frame, prev_frame)
        
    


delay_list = []
def demo(video_path='LK_optical_flow/files/Big.Buck.Bunny.mp4'):
    global output, ps_two_stream, h, w
    
    fps_list = []
    start,end,sec = 0,0,1e-4
    
    # put demo video
    cap = cv2.VideoCapture(video_path)

    prev_frame = np.zeros((h,w,3)).astype(np.uint8)
    prev_frame_gray = np.zeros((h,w)).astype(np.uint8)
    while(True):
        start = time.time()

        ret,frame = cap.read()
        if ret:            
            curr_frame = cv2.resize(frame, (h, w))
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            start = time.time()
            
            # luccas kannade , fbank
            vx,vy = two_stream.lucas_kanade_acc.compute(prev_frame_gray, curr_frame_gray)
            
            end = time.time()
            delay_list.append(end-start)
            
            two_stream.feature_bank.push(vx,vy)
            
#             action_recogn_job(curr_frame, prev_frame, output)
            if ps_two_stream==None or not ps_two_stream.is_alive():
                ps_two_stream = Process(target=action_recogn_job, args=(curr_frame, prev_frame, output))
                ps_two_stream.start()

            ## class 
            k = 3
            pred_action = np.argpartition(np.array(output), -k)[-k:]+1 #top-k

            prev_frame = curr_frame
            prev_frame_gray = curr_frame_gray

            ####################
            ##### just show ####
            vx = np.expand_dims(vx, axis=2)
            vy = np.expand_dims(vy, axis=2)
            #vv = np.sqrt((vx**2+vy**2))
            vv = np.zeros((h,w,1))
            v = np.concatenate((vv,vx,vy),axis=2).astype(np.uint8)
            showarray(np.concatenate((curr_frame,v),axis=1), 1/sec, topk=pred_action)
            ####################
            #showarray(curr_frame, 1/sec, show_meta=False)
        else:
            break

        end = time.time()
        sec = end-start
        fps_list.append(1/sec)
        #print(sec)

    print(f"avg fps:{sum(fps_list)/len(fps_list)}")






video_candidate0 = ['LK_optical_flow/files/v_BaseballPitch_g17_c05.avi',
                    'LK_optical_flow/files/v_BaseballPitch_g17_c01.avi',
                   'LK_optical_flow/files/v_BaseballPitch_g17_c02.avi',
                   'LK_optical_flow/files/v_BaseballPitch_g17_c03.avi']
for vd in video_candidate0:
    demo(vd)
    
video_candidate1 = glob.glob('LK_optical_flow/files/*.avi')
random.shuffle(video_candidate1)

for vd in video_candidate1:
    demo(vd)
