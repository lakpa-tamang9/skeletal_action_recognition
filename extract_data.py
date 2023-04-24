import argparse
import json
import os
import pickle
import time

import cv2
import natsort
import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser("Data argument parser")
parser.add_argument("--pose_estimation", default=None, choices = ["Mediapipe", "Openpose"], help = "Pose estimation type Openpose \
    or Mediapipe", required=True)
parser.add_argument("--channels", default = 2, help = "Coordinates of the pose estimation \
    2 if 2D without confidence score, 3 if 3D or 2D with confidence score")
parser.add_argument("--visualize", default = False, help = "Visualize the pose detection while \
    data extraction")
parser.add_argument("--no_persons", default=1, help="Total number of person to estimate")
parser.add_argument("--skip_frames", default=False, help = "Uses frame skipping technique \
    while creating dataset")
parser.add_argument("--json_output_path", help="The path to the json file where the extracted keypoints are saved.")
parser.add_argument("--total_frames", default=300, type = int, help = "Total number of frames \
    to process for a single action")
parser.add_argument("--videos_path", default=None, type = str, help = "Root path to the video files")
parser.add_argument("--output_path", default = None, type = str, help = "Output path of the \
                    extracted data")
parser.add_argument("--labels_path", default=None, type=str, help = "Path to label json file")
parser.add_argument("--landmarks", default= None, type = int, help = "Total number of nodes\
    in the skeleton. For Mediapipe set to 33, for openpose set to 18")
parser.add_argument("--save_keypoints", default=False)
parser.add_argument("--remove_bg", default= False, help = "Removes the background and uses \
    blank white background when set to true")
args= parser.parse_args()

# Read the labels which is stored in json format
with open(args.labels_path, "r") as f:
    global class_names
    class_names = json.load(f)

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

class ExtractionUtils(object):
    @staticmethod
    def tile(a, dim, n_tile):
        """Creates a tile of arrays by concatenating in a specified dimension

        Args:
            a (ndarray): The numpy array obtained from pose2numpy method
            dim (int): Dimension of the extracted pose data coordinates. eg: 2 for (x, y)
            n_tile (int): How many tiles to create

        Returns:
            ndarray: Tiles of array
        """        
        a = torch.from_numpy(a)
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        )
        tiled_a = torch.index_select(a, dim, order_index)
        return tiled_a.numpy()

    def pose2numpy(num_current_frames, landmark_list):
        """Converts the list of estimated poses to numpy array

        Args:
            num_current_frames (int): The current frame number of the video data
            landmark_list (list): The list of eestimated poses from Mediapipe or Openpose

        Returns:
            ndarray: Numpy array of the poses in the form (1, C, F, L, P)
            C --> Number of coordinate channels of the pose eg: 2 for (x, y), 3 for (x, y, confidence) etc.
            F --> Current frame in the loop of the video frames
            L --> Total landmarks obtained from pose estimation tool
            P --> Total number of persons to estimate in the frame
        """        
        data_numpy = np.zeros((1, args.channels, num_current_frames, args.landmarks, args.no_persons))
        skeleton_seq = np.zeros((1, args.channels, args.total_frames, args.landmarks, args.no_persons))
        
        for t in range(num_current_frames):
            m = 0 # Only predicted single pose
            if args.pose_estimation == "Mediapipe":
                data_numpy[0, 0:2, t, :, m] = np.transpose(np.array(landmark_list.get(f"frame_{t}")))
            elif args.pose_estimation == "Openpose":
                data_numpy[0, 0:2, t, :, m] = np.transpose(landmark_list[t][m].data)

        # If we have less than num_frames, repeat frames to reach total_frames
        diff = args.total_frames - num_current_frames
        if diff == 0:
            skeleton_seq = data_numpy
        while diff > 0:
            num_tiles = int(diff / num_current_frames)
            if num_tiles > 0:
                data_numpy = ExtractionUtils.tile(data_numpy, 2, num_tiles + 1)
                num_current_frames = data_numpy.shape[2]
                diff = args.total_frames - num_current_frames
            elif num_tiles == 0:
                skeleton_seq[:, :, :num_current_frames, :, :] = data_numpy
                for j in range(diff):
                    skeleton_seq[:, :, num_current_frames + j, :, :] = data_numpy[
                        :, :, -1, :, :
                    ]
                break
        return skeleton_seq
    
    @staticmethod
    def save_labels(sample_names, class_names, out_path, part):
        """Saves the label of the data according to the train and validation set

        Args:
            sample_names (list): The names of the files in the dataset to use \ 
                as training or validation sample.
            class_names (list): List of names of the action classes to classify
            out_path (str): Path of the output path to save the labels
            part (str): Argument to identify training or validation data
        """        
        sample_labels = []
        classnames = sorted(list(class_names.keys()))
        for sample_name in sample_names:
            actioname = ("_").join(sample_name.split("_")[:-1])
            
            for classname in classnames:
                if classname == actioname:
                    new_label_name = sample_name.replace(sample_name, str(class_names[actioname]))
                    sample_labels.append(int(new_label_name))

        with open("{}/{}_label.pkl".format(out_path, part), "wb") as f:
            pickle.dump((sample_names, list(sample_labels)), f)
            
    @staticmethod
    def skip_n_frames(poses_list, required_frame = 60, skip_frame_val = 3):
        """Skips certain number of frames from the list of multiple frames.
        This is optional and can be set to true if:
        --> You have long actions
        --> You want to trim into short sequences by randomly skipping certain \
            frames in between

        Args:
            poses_list (list): _description_
            required_frame (int, optional): The desired frame count. Defaults to 60.
            skip_frame_val (int, optional): How many frames to skip . Defaults to 3.
            :: For example, If 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10 frames
            :: selects 1, 5, 9 frames with 3 frames skio interval

        Returns:
            list: List of skipped frames
        """        
        skipped_frames = []
        for i in range(0, len(poses_list), skip_frame_val):
            skipped_frames.append(poses_list[i])

        extra_frames = required_frame - len(skipped_frames)
        
        if extra_frames < 0:
            remove_index = [i for i in range(1, int(abs(extra_frames) / 2) + 1)]
            for i in remove_index:
                del skipped_frames[i]
                del skipped_frames[-i]

        elif extra_frames > 0:
            copy_index = [i for i in range(1, abs(extra_frames) + 1)]
            for i in copy_index:
                copied_frame = skipped_frames[-i].copy()
                skipped_frames.append(copied_frame)

        return skipped_frames

    @staticmethod
    def split_dataset():
        """Split dataset into training and validation samples

        Returns:
            list: List of filename of training samples and validation samples
        """
        training_subjects = [i for i in range(1, 301)]
        sample_nums = []
        train_sample_names = []
        val_sample_names = []
        
        files = natsort.natsorted(os.listdir(args.videos_path))
        for file in files:
            sample_name = (file.split("."))[0]
            filenum = int(("_").join((file.split(".")[0]).split("_")[-1]))
            sample_nums.append(filenum)

            istraining = filenum in training_subjects
            if istraining:
                train_sample_names.append(sample_name)
            else:
                val_sample_names.append(sample_name)

        return train_sample_names, val_sample_names
    
if args.pose_estimation == "Openpose":
    print("Data extraction using Openpose.")
    from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
        LightweightOpenPoseLearner
    from inference.pose_estimation.lightweight_open_pose.utilities import draw
    
    assert args.landmarks == 18
    class OpenposeExtractor(object):
        def __init__(self):  
            self.pose_estimator = LightweightOpenPoseLearner()
            self.pose_estimator.load("openpose_default")
        
        def extract_openpose_data(self, sample_names, total_frames, out_path, part):
            skeleton_data = np.zeros(
                (len(sample_names), 2, total_frames, 18, 1), dtype=np.float32
            )
            bg_img = cv2.resize(cv2.imread("./resources/bg_image.jpg"), (1080, 720))
            for i, s in enumerate(tqdm(sample_names)):
                video_path = os.path.join(args.videos_path, s + ".mp4")
                image_provider = VideoReader(video_path)
                
                counter = 0
                poses_list = []        
                for img in image_provider:
                    start_time = time.perf_counter()
                    poses = self.pose_estimator.infer(img)
                    
                    if args.remove_bg:
                        img = bg_img.copy()
                        
                    for pose in poses:
                            draw(img, pose)

                    if len(poses) > 0:
                        counter += 1
                        poses_list.append(poses)
                    
                    if args.visualize:
                        cv2.imshow("Result", img)
                        key = cv2.waitKey(1)
                        if key == ord("q"):
                            break
                
                if args.skip_frames:
                    poses_list = self.skip_n_frames(poses_list=poses_list)
                    counter = len(poses_list)
                else:
                    if counter > total_frames:
                        for cnt in range(counter - total_frames):
                            poses_list.pop(0)
                        counter = total_frames
                
                if args.save_keypoints:
                    # TODO: Use arguments for the output path
                    json_output_path = "./resources/json_keypoints_data"
                    if not os.path.exists(json_output_path):
                        os.makedirs(json_output_path)
                    video_keypoints = {}
                    for index, _ in enumerate(poses_list):
                        video_keypoints[f"frame_{index}"] = (poses_list[index][0].data).tolist()
                    with open(f"{json_output_path}/{s}.json", "w") as f:
                        json.dump(video_keypoints, f, indent=3)
                    
                if counter > 0:
                    frame_skeleton_seq = ExtractionUtils.pose2numpy(counter, poses_list)
                    skeleton_data[i, :, :, :, :] = frame_skeleton_seq
                    
            np.save("{}/{}_data_joint.npy".format(out_path, part), skeleton_data)

        def run_extraction(self):
            train_sample_names, validation_sample_names = ExtractionUtils.split_dataset()
            try:
                self.extract_openpose_data(train_sample_names, args.total_frames, args.output_path, "train")
                ExtractionUtils.save_labels(train_sample_names, class_names, args.output_path, "train")
                print("Training data extraction completed \n Starting validation data extraction...")
                self.extract_openpose_data(validation_sample_names, args.total_frames, args.output_path, "val")
                ExtractionUtils.save_labels(validation_sample_names, class_names, args.output_path, "train")
            except Exception:
                raise ValueError
            print("Data extraction Finished")
            
    if __name__ == "__main__":
        estimator = OpenposeExtractor()
        estimator.run_extraction()
            
elif args.pose_estimation == "Mediapipe":
    print("Data extraction using Mediapipe")
    assert args.landmarks == 33
    import mediapipe as mp

    class MediapipeEstimator(object):
        def __init__(self):  
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
        def extract_mediapipe_data(self, sample_names, total_frames, out_path, part):
            skeleton_data = np.zeros(
                (len(sample_names), 2, total_frames, 33, 1), dtype=np.float32
            )
            for i, s in enumerate(tqdm(sample_names)):
                video_path = os.path.join(args.videos_path, s + ".mp4")

                cap = cv2.VideoCapture(video_path)
                
                counter = 0
                landmarks = []
                frame_keypoints = {}
                
                with self.mp_pose.Pose(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                ) as pose:
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            # If loading a video, use 'break' instead of 'continue'.
                            break

                        # Flip the image horizontally for a later selfie-view display, and convert
                        # the BGR image to RGB.
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        image.flags.writeable = False
                        results = pose.process(image)
                        if not results.pose_landmarks:
                            continue
                        counter += 1

                        landmark = results.pose_landmarks.landmark
                        landmarks.append(landmark)
                        if counter > total_frames:
                            for cnt in range(counter - total_frames):
                                landmarks.pop(0)
                            counter = total_frames
                            
                    cap.release()                    
                    for index, landmark in enumerate(landmarks):
                        frame_landmarks = []
                        for keypoint in landmark:
                            frame_landmarks.append([keypoint.x, keypoint.y]) 
                        frame_keypoints[f"frame_{index}"] = frame_landmarks
                    
                    if counter > 0:
                        frame_skeleton_seq = ExtractionUtils.pose2numpy(counter, frame_keypoints)
                        skeleton_data[i, :, :, :, :] = frame_skeleton_seq
                
            np.save("{}/{}_data_joint.npy".format(out_path, part), skeleton_data)
        
        def run_extraction(self):
            train_sample_names, validation_sample_names = ExtractionUtils.split_dataset()
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            try:
                print("Starting extraction!!!")
                self.extract_mediapipe_data(train_sample_names, args.total_frames, args.output_path, "train")
                ExtractionUtils.save_labels(train_sample_names, class_names, args.output_path, "train")
                print("*** Training data extraction completed *** \n Starting validation data extraction...")
                self.extract_mediapipe_data(validation_sample_names, args.total_frames, args.output_path, "val")
                ExtractionUtils.save_labels(validation_sample_names, class_names, args.output_path, "val")
            except:
                raise ValueError
            print("Data extraction Finished")
            
    if __name__ == "__main__":
        estimator = MediapipeEstimator()
        estimator.run_extraction()
            



    
    
    
    