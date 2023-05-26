import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch

from train_stgcn import SpatioTemporalGCNLearner
import json


class RecognitionDemo(object):
    def __init__(
        self,
        labels_path,
        model_path,
        model_name,
        total_frames=300,
        channels=2,
        landmarks=33,
        no_person=1,
    ):
        self.action_classifier = SpatioTemporalGCNLearner(
            in_channels=channels,
            num_point=landmarks,
            graph_type="mediapipe",
        )
        self.channels = channels
        self.landmarks, self.num_persons = landmarks, no_person
        self.total_frames = total_frames
        # self.model_saved_path = "./temp/230526_01_checkpoints"
        self.action_classifier.load(model_path, model_name)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        with open(labels_path) as f:
            class_names = json.load(f)
        self.action_labels = {v: k for k, v in class_names.items()}

    def preds2label(self, confidence):
        k = 10
        class_scores, class_inds = torch.topk(confidence, k=k)
        labels = {
            self.action_labels[int(class_inds[j])]: float(class_scores[j].item())
            for j in range(k)
        }
        return labels

    def draw_preds(self, frame, preds):
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
            (1, self.channels, num_current_frames, self.landmarks, self.num_persons)
        )
        skeleton_seq = np.zeros(
            (1, self.channels, self.total_frames, self.landmarks, self.num_persons)
        )

        for t in range(num_current_frames):
            m = 0  # Only predicted single pose
            # kps = np.array(landmark_list.get(f"frame_{t}"))
            data_numpy[0, 0:2, t, :, m] = np.transpose(
                np.array(landmark_list.get(f"frame_{t}"))
            )

        # if we have less than num_frames, repeat frames to reach num_frames
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

    def draw_boundingbox(self, image, pose_skeleton_flattened):
        """Draws bounding box around the image
        Args:
            image (tensor): Input image where bounding box is to be drawn
            skeleton (list): Exctracted pose skeleton from mediapipe
        Returns:
            ints: minimum and maximum values of x and y for bounding boxes
        """
        # Set initial values for min and max of x and y
        minx = 999
        miny = 999
        maxx = -999
        maxy = -999
        i = 0
        NaN = 0

        while i < len(pose_skeleton_flattened):
            if not (
                pose_skeleton_flattened[i] == NaN
                or pose_skeleton_flattened[i + 1] == NaN
            ):
                minx = min(minx, pose_skeleton_flattened[i])
                maxx = max(maxx, pose_skeleton_flattened[i])
                miny = min(miny, pose_skeleton_flattened[i + 1])
                maxy = max(maxy, pose_skeleton_flattened[i + 1])
            i += 2

        # Scale the min and max value according to image shape
        minx = int(minx * image.shape[1])
        miny = int(miny * image.shape[0])
        maxx = int(maxx * image.shape[1])
        maxy = int(maxy * image.shape[0])

        return minx, miny, maxx, maxy

    def prediction(self, path):
        counter = 0
        frame_keypoints = {}
        poses = []
        pred_list = []
        final_preds = []
        cap = cv2.VideoCapture(0 if path == "0" else path)  # Setting zero for camera

        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                start_time = time.perf_counter()
                height, width, _ = image.shape
                # print()

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if not results.pose_landmarks:
                    continue

                image.flags.writeable = True
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                pred_pose = results.pose_landmarks.landmark
                poses.append(pred_pose)
                counter += 1

                if not len(poses) == 0:
                    if counter > self.total_frames:
                        poses.pop(0)
                        counter = self.total_frames

                    for index, frame_pose in enumerate(poses):
                        frame_landmarks = []
                        for keypoint in frame_pose:
                            frame_landmarks.append([keypoint.x, keypoint.y])
                        frame_landmarks_flattened = [
                            keypoint
                            for landmark in frame_landmarks
                            for keypoint in landmark
                        ]
                        frame_keypoints[f"frame_{index}"] = frame_landmarks

                    # Calculate the min and max value for bounding box creation
                    minx, miny, maxx, maxy = self.draw_boundingbox(
                        image, frame_landmarks_flattened
                    )
                    if counter > 0:
                        skeleton_seq = self.pose2numpy(counter, frame_keypoints)

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
                                final_pred = max(
                                    pred_list[35:], key=pred_list[35:].count
                                )
                                pred_list.clear()

                                # Save predictions in a list
                                final_preds.append(final_pred)

                                # Check consecutive prediction of the frame
                                # If two consecutive frames same then, show second last frame
                                # else show the last frame
                                if len(final_preds) > 1:
                                    if final_preds[-1] == final_preds[-2]:
                                        pred_to_show = final_preds[-2]
                                        # Perform fifo ops
                                        final_preds.pop(0)
                                    else:
                                        pred_to_show = final_preds[-1]
                                        print(pred_to_show)
                                    image = cv2.putText(
                                        image,
                                        pred_to_show,
                                        (minx, miny),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        2,
                                        (0, 0, 255),
                                        2,
                                    )
                            else:

                                image = cv2.putText(
                                    image,
                                    "",
                                    (100, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (0, 0, 255),
                                    2,
                                )

                    # Create the bounding box
                    image = cv2.rectangle(
                        image, (minx, miny), (maxx, maxy), (0, 255, 0), 4
                    )

                    end_time = time.perf_counter()
                    fps = 1.0 / (end_time - start_time)
                    avg_fps = 0.8 * fps + 0.2 * fps
                    image = cv2.putText(
                        image,
                        "frames per sec: %.2f" % (avg_fps,),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Result", image)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments for running real time inference")
    parser.add_argument(
        "--video_path",
        default=None,
        type=str,
        help="Path to the video. Set to 0 for camera feed.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        help="Path to the json file containing class names and it labels.",
    )
    parser.add_argument("--model_path", help="Path to the trained model directory.")

    parser.add_argument(
        "--model_name", help="Name of the trained model (.pt extension)"
    )

    args = parser.parse_args()
    # args.labels_path = "./class_names.json"
    # args.model_path = "./temp/230526_01_checkpoints"
    # args.model_name = "230526_01-44-945"
    # args.video_path = "./resources/evaluation_files/test_video.mp4"
    recdem = RecognitionDemo(
        labels_path=args.labels_path,
        model_path=args.model_path,
        model_name=args.model_name,
    )
    recdem.prediction(path=args.video_path)

# Model name: mediapipe_model-44-945
