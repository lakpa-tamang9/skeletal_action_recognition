import argparse
import json
import time

import cv2
import numpy as np
import torch

from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import (
    LightweightOpenPoseLearner,
)
from inference.pose_estimation.lightweight_open_pose.utilities import draw
from train_stgcn import SpatioTemporalGCNLearner


class RecognitionDemo(object):
    def __init__(
        self,
        video_path,
        model_path,
        model_name,
        labels_path,
        channels=2,
        total_frames=300,
        landmarks=18,
        no_persons=1,
    ):

        self.channels = channels
        self.total_frames = total_frames
        self.landmarks = landmarks
        self.no_persons = no_persons
        self.pose_estimator = LightweightOpenPoseLearner()
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

        self.action_classifier = SpatioTemporalGCNLearner(
            in_channels=2,
            num_point=landmarks,
            graph_type="openpose",
        )
        print(model_path, model_name)
        self.action_classifier.load(model_path, model_name)

        self.image_provider = VideoReader(video_path)
        self.no_frames = 0
        with open(labels_path) as f:
            class_labels = json.load(f)
        self.action_labels = {v: k for k, v in class_labels.items()}
        # self.action_labels = {
        #     0: "big_wind",
        #     1: "bokbulbok",
        #     2: "chalseok_chalseok_phaldo",
        #     3: "chulong_chulong_phaldo",
        #     4: "crafty_tricks",
        #     5: "flower_clock",
        #     6: "seaweed_in_the_swell_sea",
        #     7: "sowing_corn_and_driving_pigeons",
        #     8: "waves_crashing",
        #     9: "wind_that_shakes_trees",
        # }

    def preds2label(self, confidence):
        """Converts the predictions to the corresponding label based on the confidence value

        Args:
            confidence (array): Confidence value which is the output of the model

        Returns:
            labels(str): The action labels
        """
        k = 10
        class_scores, class_inds = torch.topk(confidence, k=k)
        labels = {
            self.action_labels[int(class_inds[j])]: float(class_scores[j].item())
            for j in range(k)
        }
        return labels

    def tile(self, a, dim, n_tile):
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

    def pose2numpy(self, num_current_frames, landmark_list):

        data_numpy = np.zeros(
            (1, self.channels, num_current_frames, self.landmarks, self.no_persons)
        )
        skeleton_seq = np.zeros(
            (1, self.channels, self.total_frames, self.landmarks, self.no_persons)
        )

        for t in range(num_current_frames):
            m = 0  # Only predicted single pose
            data_numpy[0, 0:2, t, :, m] = np.transpose(landmark_list[t][m].data)

        # If we have less than num_frames, repeat frames to reach total_frames
        diff = self.total_frames - num_current_frames
        if diff == 0:
            skeleton_seq = data_numpy
        while diff > 0:
            num_tiles = int(diff / num_current_frames)
            if num_tiles > 0:
                data_numpy = self.tile(data_numpy, 2, num_tiles + 1)
                num_current_frames = data_numpy.shape[2]
                diff = self.total_frames - num_current_frames
            elif num_tiles == 0:
                skeleton_seq[:, :, :num_current_frames, :, :] = data_numpy
                for j in range(diff):
                    skeleton_seq[:, :, num_current_frames + j, :, :] = data_numpy[
                        :, :, -1, :, :
                    ]
                break
        return skeleton_seq

    def draw_preds(self, frame, preds):
        """Draws the skeletal structure of the lightweight openpose learner

        Args:
            frame (ndarray): Image frame from the feed / video
            preds (_type_): The prediction values of the model
        """
        for i, (cls, prob) in enumerate(preds.items()):
            cv2.putText(
                frame,
                f"{prob:04.3f} {cls}",
                (10, 40 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    def prediction(self):
        """Runs real time prediction based on lightweight openpose learner and trained GCN model"""
        counter = 0
        frame_count = 0
        poses_list = []
        pred_list = []
        for img in self.image_provider:
            start_time = time.perf_counter()
            poses = self.pose_estimator.infer(img)

            if not len(poses) == 0:
                for pose in poses:
                    draw(img, pose)

                if len(poses) > 0:
                    counter += 1
                    frame_count += 1
                    poses_list.append(poses)

                if counter > self.total_frames:
                    poses_list.pop(0)
                    counter = self.total_frames

                if counter > 0 and frame_count > 60:
                    skeleton_seq = self.pose2numpy(counter, poses_list)

                    prediction = self.action_classifier.infer(skeleton_seq)
                    skeleton_seq = []
                    category_labels = self.preds2label(prediction.confidence)
                    if max(list(category_labels.values())) > 0.60:
                        predicted_label = torch.argmax(prediction.confidence)
                        if counter > 20:
                            pred_text = self.action_labels[predicted_label.item()]
                            pred_list.append(pred_text)
                        else:
                            pred_text = ""

                        if len(pred_list) > 40:
                            final_pred = max(pred_list[35:], key=pred_list[35:].count)
                            pred_list.clear()
                            print(final_pred)
                            img = cv2.putText(
                                img,
                                final_pred,
                                (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 0, 255),
                                2,
                            )

                        else:
                            img = cv2.putText(
                                img,
                                "",
                                (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 0, 255),
                                2,
                            )

            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            avg_fps = 0.8 * fps + 0.2 * fps
            img = cv2.putText(
                img,
                "FPS: %.2f" % (avg_fps,),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Result", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments for running real time inference")
    parser.add_argument(
        "--video_path",
        default=None,
        type=str,
        help="Path to the video. Set to 0 for camera feed.",
    )
    parser.add_argument(
        "--model_name", default=None, type=str, help="Name of the trained model (.pt)"
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained model.")
    parser.add_argument(
        "--labels_path",
        required=True,
        type=str,
        help="Path to the json file containing class names and it labels.",
    )
    path = "./videofile.avi"
    args = parser.parse_args()
    recdem = RecognitionDemo(
        video_path=args.video_path,
        model_path=args.model_path,
        model_name=args.model_name,
        labels_path=args.labels_path,
    )
    recdem.prediction()

# path = "./resources/test_videos/wholeaction_v2.mp4"
# model_name = yagr_all_class_60_frames_v2 - 44 - 945
# model_path = "./temp/yagr_all_class_60_frames_v2_checkpoints"
