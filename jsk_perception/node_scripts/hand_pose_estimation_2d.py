#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Reference:
# @inproceedings{cao2017realtime,
#   author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
#   booktitle = {CVPR},
#   title = {Realtime Multi-Person 2D Pose Estimation
#            using Part Affinity Fields},
#   year = {2017}
#   }
# @online{Realtime_Multi-Person_Pose_Estimation,
#   title = {Reaprioriry-queueltime_Multi-Person_Pose_Estimation},
#   url = {https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation},
#   timestamp = {2016-12-29},
#   urldate = {2017-11-08},
#  }

import math

import chainer
import chainer.functions as F
from chainer import cuda
import cv2
import matplotlib
import numpy as np
import pylab as plt  # NOQA

import cv_bridge
import message_filters
import rospy
from jsk_topic_tools import ConnectionBasedTransport
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from jsk_recognition_msgs.msg import PeoplePose
from jsk_recognition_msgs.msg import PeoplePoseArray
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from openpose.net import OpenPoseHandNet


class HandPoseDetector2D(ConnectionBasedTransport):

    def __init__(self):
        super(HandPoseDetector2D, self).__init__()

        self.backend = rospy.get_param('~backend', 'chainer')
        self.gpu = rospy.get_param('~gpu', -1)  # -1 is cpu mode
        self.with_depth = rospy.get_param('~with_depth', False)
        self.width = rospy.get_param('~width', None)
        self.height = rospy.get_param('~height', None)
        self.hand_heatmap_peak_thresh = rospy.get_param('~hand_heatmap_peak_thresh', 0.1)
        self.ksize = rospy.get_param('~kernel_size', 17)
        self.gaussian_sigma = rospy.get_param('~gaussian_sigma', 2.5)

        kernel = self.create_gaussian_kernel(
            sigma=self.gaussian_sigma,
            ksize=self.ksize)
        if self.gpu >= 0:
            kernel = chainer.cuda.to_gpu(kernel)
        self.gaussian_kernel = kernel
        self._load_model()
        self.image_pub = self.advertise('~output', Image, queue_size=1)
        self.pose_pub = self.advertise('~pose', PeoplePoseArray, queue_size=1)
        self.sub_info = None
        if self.with_depth is True:
            self.pose_2d_pub = self.advertise('~pose_2d', PeoplePoseArray, queue_size=1)

    @property
    def visualize(self):
        return self.image_pub.get_num_connections() > 0

    def _load_model(self):
        if self.backend == 'chainer':
            self._load_chainer_model()
        else:
            raise RuntimeError('Unsupported backend: %s', self.backend)

    def _load_chainer_model(self):
        model_file = rospy.get_param('~model_file')
        self.model = OpenPoseHandNet()
        chainer.serializers.load_npz(model_file, self.model)
        rospy.loginfo('Finished loading trained model: {0}'.format(model_file))
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def subscribe(self):
        if self.with_depth:
            queue_size = rospy.get_param('~queue_size', 10)
            sub_img = message_filters.Subscriber(
                '~input', Image, queue_size=1, buff_size=2**24)
            sub_depth = message_filters.Subscriber(
                '~input/depth', Image, queue_size=1, buff_size=2**24)
            self.subs = [sub_img, sub_depth]

            # NOTE: Camera info is not synchronized by default.
            # See https://github.com/jsk-ros-pkg/jsk_recognition/issues/2165
            sync_cam_info = rospy.get_param("~sync_camera_info", False)
            if sync_cam_info:
                sub_info = message_filters.Subscriber(
                    '~input/info', CameraInfo, queue_size=1, buff_size=2**24)
                self.subs.append(sub_info)
            else:
                self.sub_info = rospy.Subscriber(
                    '~input/info', CameraInfo, self._cb_cam_info)

            if rospy.get_param('~approximate_sync', True):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            if sync_cam_info:
                sync.registerCallback(self._cb_with_depth_info)
            else:
                self.camera_info_msg = None
                sync.registerCallback(self._cb_with_depth)
        else:
            sub_img = rospy.Subscriber(
                '~input', Image, self._cb, queue_size=1, buff_size=2**24)
            self.subs = [sub_img]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        if self.sub_info is not None:
            self.sub_info.unregister()
            self.sub_info = None

    def _cb_cam_info(self, msg):
        self.camera_info_msg = msg
        self.sub_info.unregister()
        self.sub_info = None
        rospy.loginfo("Received camera info")

    def _cb_with_depth(self, img_msg, depth_msg):
        if self.camera_info_msg is None:
            return
        self._cb_with_depth_info(img_msg, depth_msg, self.camera_info_msg)

    def _cb_with_depth_info(self, img_msg, depth_msg, camera_info_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth_img = br.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth_msg.encoding == '16UC1':
            depth_img = np.asarray(depth_img, dtype=np.float32)
            depth_img /= 1000  # convert metric: mm -> m
        elif depth_msg.encoding != '32FC1':
            rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)

        key_points = self.pose_estimate(img)
        pose_estimated_img = draw_hand_keypoints(img, key_points, (0, 0))
        pose_estimated_msg = br.cv2_to_imgmsg(
            pose_estimated_img.astype(np.uint8), encoding='bgr8')
        pose_estimated_msg.header = img_msg.header

        # people_pose_msg = PeoplePoseArray()
        # people_pose_msg.header = img_msg.header
        # people_pose_2d_msg = self._create_2d_people_pose_array_msgs(
        #     people_joint_positions,
        #     img_msg.header)

        # calculate xyz-position
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]
        for person_joint_positions in key_points:
            pose_msg = PeoplePose()
            for joint_pos in person_joint_positions:
                if joint_pos['score'] < 0:
                    continue
                z = float(depth_img[int(joint_pos['y'])][int(joint_pos['x'])])
                if np.isnan(z):
                    continue
                x = (joint_pos['x'] - cx) * z / fx
                y = (joint_pos['y'] - cy) * z / fy
                pose_msg.limb_names.append(joint_pos['limb'])
                pose_msg.scores.append(joint_pos['score'])
                pose_msg.poses.append(Pose(position=Point(x=x, y=y, z=z),
                                           orientation=Quaternion(w=1)))
            people_pose_msg.poses.append(pose_msg)

        self.pose_2d_pub.publish(people_pose_2d_msg)
        self.pose_pub.publish(people_pose_msg)
        if self.visualize:
            self.image_pub.publish(pose_estimated_msg)

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        key_points = self.pose_estimate(img)
        pose_estimated_img = draw_hand_keypoints(img, key_points, (0, 0))

        pose_estimated_msg = br.cv2_to_imgmsg(
            pose_estimated_img.astype(np.uint8), encoding='bgr8')
        pose_estimated_msg.header = img_msg.header

        # people_pose_msg = self._create_2d_people_pose_array_msgs(
        #     people_joint_positions,
        #     img_msg.header)

        # self.pose_pub.publish(people_pose_msg)
        if self.visualize:
            self.image_pub.publish(pose_estimated_msg)

    def _create_2d_people_pose_array_msgs(self, people_joint_positions, header):
        people_pose_msg = PeoplePoseArray(header=header)
        for person_joint_positions in people_joint_positions:
            pose_msg = PeoplePose()
            for joint_pos in person_joint_positions:
                if joint_pos['score'] < 0:
                    continue
                pose_msg.limb_names.append(joint_pos['limb'])
                pose_msg.scores.append(joint_pos['score'])
                pose_msg.poses.append(Pose(position=Point(x=joint_pos['x'],
                                                          y=joint_pos['y'],
                                                          z=0)))
            people_pose_msg.poses.append(pose_msg)
        return people_pose_msg

    def pose_estimate(self, bgr):
        if self.backend == 'chainer':
            return self._pose_estimate_chainer_backend(bgr)
        raise ValueError('Unsupported backend: {0}'.format(self.backend))

    def _pose_estimate_chainer_backend(self, bgr_image, hand_type="right"):
        xp = self.model.xp

        if hand_type == "left":
            bgr_image = cv2.flip(bgr_image, 1)

        original_height, original_width, _ = bgr_image.shape

        if self.width is None or self.height is None:
            resized_image = bgr_image
        else:
            resized_image = cv2.resize(
                bgr_image, (self.height, self.width))
        x = xp.array(resized_image[np.newaxis], dtype=np.float32).\
            transpose(0, 3, 1, 2) / 256.0 - 0.5

        hs = self.model(x)
        heatmaps = F.resize_images(hs[-1],
            (original_height, original_width)).data[0]

        if hand_type == "left":
            heatmaps = cv2.flip(heatmaps.transpose(1, 2, 0), 1).transpose(2, 0, 1)

        keypoints = self.compute_peaks_from_heatmaps(heatmaps)

        return keypoints

    def create_gaussian_kernel(self, sigma=1, ksize=5):
        center = int(ksize / 2)
        kernel = np.zeros((1, 1, ksize, ksize), dtype=np.float32)
        for y in range(ksize):
            distance_y = abs(y-center)
            for x in range(ksize):
                distance_x = abs(x-center)
                kernel[0][0][y][x] = 1/(sigma**2 * 2 * np.pi) * np.exp(-(distance_x**2 + distance_y**2)/(2 * sigma**2))
        return kernel

    def compute_peaks_from_heatmaps(self, heatmaps):
        keypoints = []
        xp = chainer.cuda.get_array_module(heatmaps)

        heatmaps = F.convolution_2d(heatmaps[:, None],
            self.gaussian_kernel, stride=1, pad=int(self.ksize/2)).data.squeeze()
        for heatmap in heatmaps[:-1]:
            max_value = heatmap.max()
            if max_value > self.hand_heatmap_peak_thresh:
                coords = np.array(np.where(chainer.cuda.to_cpu(heatmap==max_value))).flatten().tolist()
                keypoints.append([coords[1], coords[0], max_value])
            else:
                keypoints.append(None)
        return keypoints

    def _visualize(self, img, joint_cands_indices, all_peaks, candidate):

        cmap = matplotlib.cm.get_cmap('hsv')
        for i in range(len(self.index2limbname)-1):
            rgba = np.array(cmap(1 - i / 18. - 1. / 36))
            rgba[0:3] *= 255
            for j in range(len(all_peaks[i])):
                cv2.circle(img, (int(all_peaks[i][j][0]), int(
                    all_peaks[i][j][1])), 4, self.colors[i], thickness=-1)

        stickwidth = 4
        for i in range(len(self.index2limbname) - 2):
            for joint_cand_indices in joint_cands_indices:
                index = joint_cand_indices[np.array(self.limb_sequence[i],
                                                    dtype=np.int32) - 1]
                if -1 in index:
                    continue
                cur_img = img.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(
                    length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_img, polygon, self.colors[i])
                img = cv2.addWeighted(img, 0.4, cur_img, 0.6, 0)

        return img


fingers_indices = [
    [[0, 1], [1, 2], [2, 3], [3, 4]],
    [[0, 5], [5, 6], [6, 7], [7, 8]],
    [[0, 9], [9, 10], [10, 11], [11, 12]],
    [[0, 13], [13, 14], [14, 15], [15, 16]],
    [[0, 17], [17, 18], [18, 19], [19, 20]]]


def draw_hand_keypoints(orig_img, hand_keypoints, left_top):
    img = orig_img.copy()
    left, top = left_top

    finger_colors = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]

    for i, finger_indices in enumerate(fingers_indices):
        for finger_line_index in finger_indices:
            keypoint_from = hand_keypoints[finger_line_index[0]]
            keypoint_to = hand_keypoints[finger_line_index[1]]

            if keypoint_from:
                keypoint_from_x, keypoint_from_y, _ = keypoint_from
                cv2.circle(img, (keypoint_from_x + left, keypoint_from_y + top), 3, finger_colors[i], -1)

            if keypoint_to:
                keypoint_to_x, keypoint_to_y, _ = keypoint_to
                cv2.circle(img, (keypoint_to_x + left, keypoint_to_y + top), 3, finger_colors[i], -1)

            if keypoint_from and keypoint_to:
                cv2.line(img, (keypoint_from_x + left, keypoint_from_y + top), (keypoint_to_x + left, keypoint_to_y + top), finger_colors[i], 1)

    return img


if __name__ == '__main__':
    rospy.init_node('hand_pose_estimation_2d')
    HandPoseDetector2D()
    rospy.spin()
