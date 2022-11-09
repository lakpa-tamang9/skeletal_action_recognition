# Realtime Skeleton based Action Recognition using Spatio-temporal Graph Convolutional Networks

A simplified action recognition project to detect performed dance actions in real time. 

## Usage
### 1. Prepare and extract data
- The first step to realizing action recognition is to collect the desired data. This can be done by repeatedly recording the actions of multiple persons and creating a large sample of dataset. You can also use opensource datasets or create your own custom dataset consisting of the any type of actions.
- After you have the video files of the dataset, then the next step is to extract pose data from those videos. In this project, both Mediapipe and Openpose frameworks are used to extract the pose keypoints. Run extract_data.py to extract the pose data from your videos. To run this file, provide the run time arguments as explained in the file.

### 2. Train the model
After preparation of the dataset is completed, you can proceed for training the model. The deep learning model used in this project is a Graph Convolutional Network. This model is responsible to capture the spatial and temporal dependencies of the performed actions. To train the model run the file train_stgcn.py. 
### 3. Perform real time recognition
After training the model, you can proceed for the real time inference. For inference, you can either use pre-recorded video or directly use web-camera feed to capture the actions of the performer in real time. The captured actions will then be passed to pose data extraction, followed by real time prediction of the performed actions.
There are two programs for realizing real time recogntiion; one by using Mediapipe and the other by using Openpose with run_recog_mp.py and run_recognition.py files respectively.
References:
1. https://github.com/opendr-eu/opendr/blob/d8b1572e742bab6a2edc5ef8195cbf487c042a4d/src/opendr/perception/skeleton_based_action_recognition/spatio_temporal_gcn_learner.py

2. https://github.com/open-mmlab/mmskeleton
3. https://github.com/yysijie/st-gcn