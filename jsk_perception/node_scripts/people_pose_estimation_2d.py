#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import pickle

import chainer
from chainer import cuda
import chainer.functions as F
import cupy

import cv2
import matplotlib
import pylab as plt
import numpy as np
import chainer.links.caffe
import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import PeoplePose2D
from jsk_recognition_msgs.msg import PeoplePose2DArray


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


class PeoplePoseEstimation2D(ConnectionBasedTransport):
    # find connection in the specified sequence,
    # center 29 is in the position 15
    limb_sequence = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],
                     [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],
                     [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    # the middle joints heatmap correpondence
    map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
               [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
               [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38],
               [45, 46]]

    index2limbname = ["Nose",
                      "Neck",
                      "RShoulder",
                      "RElbow",
                      "RWrist",
                      "LShoulder",
                      "LElbow",
                      "LWrist",
                      "RHip",
                      "RKnee",
                      "RAnkle",
                      "LHip",
                      "LKnee",
                      "LAnkle",
                      "REye",
                      "LEye",
                      "REar",
                      "LEar",
                      "Bkg"]


    def __init__(self):
        super(self.__class__, self).__init__()
        self.backend = rospy.get_param('~backend', 'chainer')
        self.scales = rospy.get_param('~scales', [0.38])
        self.stride = rospy.get_param('~stride', 8)
        self.pad_value = rospy.get_param('~pad_value', 128)
        self.thre1 = rospy.get_param('~thre1', 0.1)
        self.thre2 = rospy.get_param('~thre2', 0.05)
        self.gpu = rospy.get_param('~gpu', -1)  # -1 is cpu mode
        self._load_model()
        self.pub = self.advertise('~output', Image, queue_size=1)
        self.pose_pub = self.advertise('~pose', PeoplePose2DArray, queue_size=1)

    def _load_model(self):
        if self.backend == 'chainer':
            self._load_chainer_model()
        else:
            raise RuntimeError('Unsupported backend: %s', self.backend)

    def _load_chainer_model(self):
        # model_name = rospy.get_param('~model_name')
        # model_h5 = rospy.get_param('~model_h5')
        model_file = rospy.get_param('~model_file')
        # model = dict(caffemodel="/home/iory/workspace/caffe_rtpose/model/coco/pose_iter_440000.caffemodel")
        # rospy.loginfo('Loading trained model: {0}'.format(model_h5))
        # S.load_hdf5(model_h5, self.model)
        # rospy.loginfo('Finished loading trained model: {0}'.format(model_h5))
        # self.func = chainer.functions.caffe.CaffeFunction(model['caffemodel'])
        self.func = pickle.load(open(model_file, 'rb'))
        if self.gpu != -1:
            self.func.to_gpu(self.gpu)

    def subscribe(self):
        sub_img = rospy.Subscriber(
            '~input', Image, self._cb, queue_size=1, buff_size=2**24)
        self.subs = [sub_img]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        pose_estimated_img, poses_msg = self.pose_estimate(img)
        pose_estimated_msg = br.cv2_to_imgmsg(pose_estimated_img.astype(np.uint8))
        pose_estimated_msg.header = img_msg.header
        pose_estimated_msg.encoding = "bgr8"
        poses_msg.heapq = img_msg.header
        self.pose_pub.publish(poses_msg)
        self.pub.publish(pose_estimated_msg)

    def pose_estimate(self, bgr):
        if self.backend == 'chainer':
            return self._pose_estimate_chainer_backend(bgr)
        raise ValueError('Unsupported backend: {0}'.format(self.backend))

    def _pose_estimate_chainer_backend(self, bgr_img):
        xp = cuda.cupy if self.gpu != -1 else np

        heatmap_avg = xp.zeros((bgr_img.shape[0], bgr_img.shape[1], 19),
                               dtype=np.float32)
        paf_avg = xp.zeros((bgr_img.shape[0], bgr_img.shape[1], 38),
                           dtype=np.float32)

        for scale in self.scales:
            img = cv2.resize(bgr_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            padded_img, pad = padRightDownCorner(img, self.stride, self.pad_value)
            # for chainer
            x = np.transpose(np.float32(padded_img[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
            if self.gpu != -1:
                x = chainer.cuda.to_gpu(x)
            x = chainer.Variable(x)
            y = self.func(inputs={'image': x},
                          outputs=['Mconv7_stage6_L2', 'Mconv7_stage6_L1'])
            # extract outputs, resize, and remove padding
            y0 = F.resize_images(y[0], (184, 248))
            heatmap = y0[:, :, :padded_img.shape[0]-pad[2], :padded_img.shape[1]-pad[3]]
            heatmap = F.resize_images(heatmap, (bgr_img.shape[0], bgr_img.shape[1]))
            heatmap = xp.transpose(xp.squeeze(heatmap.data), (1, 2, 0))
            y1 = F.resize_images(y[1], (184, 248))
            paf = y1[:, :, :padded_img.shape[0]-pad[2], :padded_img.shape[1]-pad[3]]
            paf = F.resize_images(paf, (bgr_img.shape[0], bgr_img.shape[1]))
            paf = xp.transpose(xp.squeeze(paf.data), (1, 2, 0))

            coeff = 1.0 / len(self.scales)
            paf_avg += paf * coeff
            heatmap_avg += heatmap * coeff
        all_peaks = []
        peak_counter = 0

        heatmav_left = xp.zeros_like(heatmap_avg)
        heatmav_left[1:, :] = heatmap_avg[:-1, :]
        heatmav_right = xp.zeros_like(heatmap_avg)
        heatmav_right[:-1, :] = heatmap_avg[1:, :]
        heatmav_up = xp.zeros_like(heatmap_avg)
        heatmav_up[:, 1:] = heatmap_avg[:, :-1]
        heatmav_down = xp.zeros_like(heatmap_avg)
        heatmav_down[:, :-1] = heatmap_avg[:, 1:]
        peaks_binary = (heatmap_avg >= heatmav_left) & \
                       (heatmap_avg >= heatmav_right) & \
                       (heatmap_avg >= heatmav_up) & \
                       (heatmap_avg >= heatmav_down) & \
                       (heatmap_avg > self.thre1)

        for part in range(18):
            # peaks = xp.array(xp.nonzero(peaks_binary[..., part]),
            #                  dtype=np.int32)
            tmp0, tmp1 = xp.nonzero(peaks_binary[...,part])
            peaks = xp.array(zip(tmp1, tmp0),
                             dtype=np.int32)
            peaks_with_score_and_id = \
                [xp.concatenate([xp.array(x, dtype=np.float32),
                                 xp.array([heatmap_avg[x[0], x[1], part]],
                                          dtype=np.float32),
                                 xp.array([id_value], dtype=np.float32)])
                 for id_value, x in enumerate(peaks, peak_counter)]
            all_peaks.append(xp.array(peaks_with_score_and_id,
                                      dtype=np.float32))
            peak_counter += len(peaks)

        connection_all = []
        mid_num = 10
        eps = 1e-8
        score_mid = paf_avg[:, :, [[x-19 for x in self.map_idx[k]] for k in range(len(self.map_idx))]]
        cands = np.array(all_peaks)[np.array(self.limb_sequence) - 1]
        candAs = cands[:, 0]
        candBs = cands[:, 1]
        nAs = np.array([len(candA) for candA in candAs])
        nBs = np.array([len(candB) for candB in candBs])
        target_indices = np.nonzero(np.logical_and(nAs != 0, nBs != 0))[0]
        if len(target_indices) == 0:
            return bgr_img
        candB = [np.tile(np.array(chainer.cuda.to_cpu(candB),
                                  dtype=np.float32), (nA, 1)).astype(np.float32) for candB, nA in zip(candBs[target_indices], nAs[target_indices])]
        candA = [np.repeat(np.array(chainer.cuda.to_cpu(candA),
                                    dtype=np.float32), nB, axis=0).astype(np.float32) for candA, nB  in zip(candAs[target_indices], nBs[target_indices])]
        vec = np.vstack(candB)[:,:2] - np.vstack(candA)[:,:2]
        vec = chainer.cuda.to_gpu(vec)
        norm = xp.sqrt(xp.sum(vec ** 2, axis=1)) + eps
        vec = vec / norm[:, None]
        startend = zip(np.round(np.mgrid[np.vstack(candA)[:,1].reshape(-1, 1):np.vstack(candB)[:,1].reshape(-1, 1):(mid_num*1j)]).astype(np.int32),
                    np.round(np.mgrid[np.vstack(candA)[:,0].reshape(-1, 1):np.vstack(candB)[:,0].reshape(-1, 1):(mid_num*1j)]).astype(np.int32),
                    np.concatenate([[[index] * mid_num for i in range(len(c))] for index, c in zip(target_indices, candB)]),)

        v = score_mid[np.concatenate(startend, axis=1).tolist()].reshape(-1, mid_num, 2)
        score_midpts = xp.sum(v * xp.repeat(vec, (mid_num), axis=0).reshape(-1, mid_num, 2), axis=2)
        score_with_dist_prior = xp.sum(score_midpts, axis=1) / mid_num + \
                                                        xp.minimum(0.5 * bgr_img.shape[0] / norm - 1, xp.zeros_like(norm))
        c1 = xp.sum(score_midpts > self.thre2, axis=1) > 0.8 * mid_num
        c2 = score_with_dist_prior > 0.0
        criterion = xp.logical_and(c1, c2)
        bins = np.concatenate([np.zeros(1), np.cumsum(nAs * nBs)]).astype(np.float32)
        tmp_index = xp.nonzero(criterion)[0]
        tmp_index = chainer.cuda.to_cpu(tmp_index)
        # tmp_indexは有効な(c1 c2を満たす)index
        k_s = np.digitize(tmp_index, bins)
        k_s -= 1
        i_s = (tmp_index - (bins[k_s])) // nBs[k_s] # k_s-1
        j_s = (tmp_index - (bins[k_s])) % nBs[k_s] # k_s-1

        ccandA = [xp.repeat(xp.array(tmp_candA, dtype=xp.float32), nB, axis=0) for tmp_candA, nB  in zip(candAs, nBs)]
        ccandB = [xp.tile(xp.array(tmp_candB, dtype=xp.float32), (nA, 1)) for tmp_candB, nA in zip(candBs, nAs)]
        score_with_dist_prior = chainer.cuda.to_cpu(score_with_dist_prior)
        connection_candidate = np.concatenate([k_s.reshape(-1, 1),
                                            i_s.reshape(-1, 1),
                                            j_s.reshape(-1, 1),
                                            score_with_dist_prior[tmp_index][None,].T,
                                            (score_with_dist_prior[tmp_index][None,] + \
                                                np.concatenate(candA)[tmp_index, 2] + np.concatenate(candB)[tmp_index, 2]).T], axis=1)

        def mycmp(x, y):
            if x[0] > y[0]:
                return -1
            elif x[0] == y[0]:
                if x[3] > y[3]:
                    return 1
                elif x[3] == y[3]:
                    return 0
                else:
                    return -1
            else:
                return 1

        connection_all = []
        connection_candidate = sorted(connection_candidate, cmp=mycmp, reverse=True)
        for _ in range(0, 19):
            connection = np.zeros((0,5), dtype=np.float32)
            connection_all.append(connection)

        for c_candidate in connection_candidate:
            k, i, j, s = c_candidate[0:4]
            k = int(k)
            i = int(i)
            j = int(j)
            if(len(connection_all[k]) >= min(nAs[k], nBs[k])):
                continue
            i *= nBs[k]
            if(i not in connection_all[k][:,3] and j not in connection_all[k][:,4]):
                connection_all[k] = np.vstack([connection_all[k], np.array([ccandA[k][i][3], ccandB[k][j][3], float(s), i, j], dtype=np.float32)])

        subset = -1 * np.ones((0, 20), dtype=np.float32)
        candidate = xp.array([item for sublist in all_peaks for item in sublist],
                             dtype=np.float32)
        special_k = []
        for k in range(len(self.map_idx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(self.limb_sequence[k]) - 1
                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == float(partAs[i]) or subset[j][indexB] == float(partBs[i]):
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != float(partBs[i])):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print "found = 2"
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # visualize
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        cmap = matplotlib.cm.get_cmap('hsv')

        canvas = bgr_img[:]
        for i in range(18):
            rgba = np.array(cmap(1 - i/18. - 1./36))
            rgba[0:3] *= 255
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, (int(all_peaks[i][j][0]), int(all_peaks[i][j][1])), 4, colors[i], thickness=-1)

        # visualize 2
        stickwidth = 4

        poses_msg = PeoplePose2DArray()
        # for people_id, indices in enumerate(subset):
        #     for index in indices:
        #         if index == -1:
        #             continue

        #         pose_msg = PeoplePose2D()
        #         pose_msg.id = index
        #         pose_msg.x = mX
        #         pose_msg.y = mY
        #         pose_msg.string = self.index2limbname[index]
        #         poses_msg.append(pose_msg)
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(self.limb_sequence[i])-1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

                for i in index:
                    pose_msg = PeoplePose2D()
                    pose_msg.id = i
                    pose_msg.x = mX[i]
                    pose_msg.y = mY[i]
                    pose_msg.string = self.index2limbname[i]
                    poses_msg.append(pose_msg)
        return canvas, poses_msg


if __name__ == '__main__':
    rospy.init_node('people_pose_estimation_2d')
    PeoplePoseEstimation2D()
    rospy.spin()
