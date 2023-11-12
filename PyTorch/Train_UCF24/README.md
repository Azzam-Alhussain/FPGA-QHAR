# Two-Stream Network implemented in PyTorch

The backbone of each stream is **`SimpNet`**

&nbsp;


## Performance
Stream     | Accuracy
:-----------:|:-----------:
RGB  | -
Optical Flow  | -
Fusion (Two Stream)  | **`53%-67%`** (only stack 20 optical flow images：10 x_direction 10 y_direction)

&nbsp;


## Data Preparation
### The First Way
Original Dataset：[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)



## Train
Before training, you should new a directory named **`model`** to save checkpoint file.
```python
python3 trainTwoStreamNet.py
```
&nbsp;


## demo
This is a demo video for test. I randomly set the **`test_video_id = 1000`** from **`testset`** to run this demo python file. What's more, I use the checkpoint file saved in **`9000-th`** iteration as the demo model.

You can change the **`test_video_id`** at here:
```python
# set the test video id in testset
test_video_id = 1000
print('Video Name:', LoadUCF101Data.TestVideoNameList[test_video_id])
```

You can change the **`checkpoint_file_path`** at here:
```python
# load the chekpoint file
state = torch.load('model/checkpoint-9000.pth')
twoStreamNet.load_state_dict(state['model'])
```

run **`demo.py`** file
```python
CUDA_VISIBLE_DEVICES=0 python3 demo.py
```
