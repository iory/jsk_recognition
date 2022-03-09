#!/usr/bin/env python

from __future__ import print_function

import rospy
from hark_msgs.msg import HarkWave, HarkSource, HarkSourceVal
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import pyroomacoustics as pra
import time
import message_filters
import sensor_msgs.point_cloud2 as pc2

import tf


class WaveDataNode():

    def __init__(self):
        self.audio_prefetch = rospy.get_param("~audio_prefetch", 0.5)
        self.sampling_rate = rospy.get_param("~sampling_rate", 16000)
        self.window_function = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0.0, 1.0, 1.0 / (self.audio_prefetch/8 * self.sampling_rate)))

        self.HEADER = None
        self.FIELDS =[
            PointField(name="x", offset=0, datatype=7, count=1),
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
            PointField(name="pfh", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.t_data = np.empty((8,0))
        self.t_data2 = np.empty((8,0))
        self.t_data_len = 8000

        self.hsrc = HarkSource()
        self.c = 343.    # speed of sound
        self.fs = 16000  # sampling frequency
        self.nfft = 128  # FFT size
        self.freq_range = [800, 4000]

        self.flag = False

        self.hsrc_list = []
        for i in range(16200):
            self.hsrc_list.append(HarkSourceVal())
        distance = 2.
        snr_db = 5.
        10 ** (-snr_db /10)/(4. * np.pi * distance) **2

        #tamago1
        mic_array = np.array([[30.0, 0.0, 0.0],
                              [17.3, 17.3, 0.0],
                              [0.0, 30.0, 0.0],
                              [-17.3, 17.3, 0.0],
                              [-30.0, 0.0, 0.0],
                              [-17.3, -17.3, 0.0],
                              [0.0, -30.0, 0.0],
                              [17.3, -17.3, 0.0]])
        mic_loc = mic_array
        mic_loc /= 1000.
        mic_loc = mic_loc.T

        self.doa = pra.doa.algorithms["MUSIC"](
            mic_loc, self.fs, self.nfft,
            c=self.c, num_src=1, dim=3)

        self.spatial_resp = np.empty((360,))

        self.subscribe()
        self.pub_src = rospy.Publisher("~output", HarkSource, queue_size=1)
        self.pub_src_max = rospy.Publisher("~max", HarkSource, queue_size=1)
        self.pub_pc = rospy.Publisher("~output_pointcloud", PointCloud2, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self.timer_cb)

    def subscribe(self):
        self.sub_mic1 = rospy.Subscriber(
            "~input", HarkWave, queue_size=1, buff_size=2**24,
            callback=self._callback)

    def unsubscribe(self):
        self.sub_mic1.unsubscribe()

    def timer_cb(self, timer):
        start = time.time()
        X = pra.transform.stft.analysis(self.t_data3.T, self.nfft, self.nfft // 2)
        X = X.transpose([2, 1, 0])
        self.doa.locate_sources(X, num_src=1, freq_range=self.freq_range)
        end=time.time()
        elapsed_time = end - start
        rospy.loginfo("elapsed_time: {}".format(elapsed_time))

        test1, test2, test3 = self.doa.grid.regrid()
        test1 = test1.flatten()
        test2 = test2.flatten()
        test3 = test3.flatten()
        min = np.min(test3)
        max = np.max(test3)
        test3 = (test3 - min) / (max - min)

        max_idx = self.doa.grid.find_peaks(k=1)[0]
        print(max_idx)
        second_max_idx = self.doa.grid.find_peaks(k=2)[0]
        print(second_max_idx)

        mmax_az = self.doa.grid.azimuth[max_idx]
        mmax_el = self.doa.grid.colatitude[max_idx]
        max_az = test1
        max_el = test2

        max_az = np.where(max_az < 0, max_az + 2*np.pi, max_az)
        max_az *= 180./np.pi
        mmax_az = np.where(mmax_az < 0, mmax_az + 2*np.pi, mmax_az)
        mmax_az *= 180./np.pi
        print("azimuth:{}".format(mmax_az))

        max_el = np.pi/2. - max_el
        max_el *= 180./np.pi
        mmax_el = np.pi/2. - mmax_el
        mmax_el *= 180./np.pi
        print("elevation:{}".format(mmax_el))

        self.hsrc.src = []
        self.hsrc2.src = []
        j = 0

        for i in range(16200):
            self.hsrc_list[i].id = i
            self.hsrc_list[i].power = test3[i]
            self.hsrc_list[i].azimuth = max_az[i]
            self.hsrc_list[i].elevation = max_el[i]
            j+=1

        self.hsrc.src.extend(self.hsrc_list)
        print(len(self.hsrc.src))
        self.pub_src.publish(self.hsrc)

        self.hsrc2.src.append(HarkSourceVal(id=0, azimuth=mmax_az, elevation=mmax_el))
        self.pub_src_max.publish(self.hsrc2)

        POINTS = []
        x_mic = np.cos(np.radians(max_el))* np.cos(np.radians(max_az))
        y_mic = np.cos(np.radians(max_el))* np.sin(np.radians(max_az))
        z_mic = np.sin(np.radians(max_el))
        rgb = test3 * 255.0
        g = np.vstack((x_mic, y_mic, z_mic, rgb)).T
        POINTS.extend(g.tolist())
        print(("point", len(POINTS)))
        point_cloud = pc2.create_cloud(self.HEADER, self.FIELDS, POINTS)
        self.pub_pc.publish(point_cloud)

    def _callback(self, msg1):
        self.hsrc = HarkSource()
        self.hsrc.header = msg1.header
        self.hsrc.count = msg1.count

        self.HEADER = msg1.header
        self.HEADER.frame_id = "tamago1"

        data_list = np.array([])
        for i in range(len(msg1.src)):
            data_list = np.append(data_list , msg1.src[i].wavedata[-160:])

        data_list = np.reshape(data_list, (len(msg1.src), -1))
        self.t_data = np.hstack((self.t_data, np.array(data_list)))
        self.t_data = self.t_data[:,-self.t_data_len:]
        self.u_data = np.array(data_list)

        self.t_data3 = self.t_data
        self.u_data3 = self.u_data


if __name__ == "__main__":
    rospy.init_node("sound_localize")
    listener = tf.TransformListener()
    wavedatanode = WaveDataNode()
    rospy.spin()
