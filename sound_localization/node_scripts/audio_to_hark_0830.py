#!/usr/bin/env python

from __future__ import print_function
import rospy
from audio_common_msgs.msg import AudioData
from hark_msgs.msg import HarkWave, HarkWaveVal
import numpy as np

import os

class AudioDataToHarkWave():
    def __init__(self):
        self.mean_flag = False
        
        self.audio_prefetch = rospy.get_param("~audio_prefetch", 0.5)
        self.rate = rospy.get_param("~rate", 16000)
        self.bitwidth = rospy.get_param("~bitwidth", 2)
        self.bitdepth = rospy.get_param("~bitdepth", 16)

        self.sampling_rate = rospy.get_param("~sampling_rate", 16000)
        self.audio_prefetch_bytes = int(
            self.audio_prefetch * self.rate * self.bitdepth/8)
        self.audio_prefetch_buffer = str()
        self.audio_prefetch_buffer_sparse = np.array([], dtype='int16')
        self.audio_prefetch_sparse_bytes = int(
            self.audio_prefetch_bytes * self.sampling_rate // self.rate // 2)
        self.freq = np.linspace(0, self.sampling_rate, self.audio_prefetch_sparse_bytes)
        self.window_function = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0.0, 1.0, 1.0 / (self.audio_prefetch/8 * self.sampling_rate)))
        #self.window_function = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0.0, 1.0, 1.0 / 160))

        self.first_100 = 0
        self.first = True
        self.a_lis = np.empty((0, 8))
        self.mean_amplitude = 0
        
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.cur_dir , "../data/")

        self.sub = rospy.Subscriber("~input", AudioData, self._callback, queue_size=1000, buff_size=2**24)
        self.pub = rospy.Publisher("~output", HarkWave, queue_size=1)
        #rospy.Rate(100)

    def _callback(self, msg):
        data = msg.data
        data16 = np.frombuffer(data, dtype="int16")
        data16 = np.array(data16)
        data16.reshape(-1,8)

        #print(data16.max())

        self.audio_prefetch_buffer_sparse = np.append(
            self.audio_prefetch_buffer_sparse,
            data16)
        self.audio_prefetch_buffer_sparse = self.audio_prefetch_buffer_sparse[-self.audio_prefetch_sparse_bytes:]
        #print(data16.shape) #1280
        #print(self.audio_prefetch_sparse_bytes) #8000
        #print(self.audio_prefetch_buffer_sparse.shape) #8000

        if len(self.audio_prefetch_buffer_sparse) != self.audio_prefetch * self.sampling_rate:
            return

        # if self.first_100 <= 300:
        #     self.a_lis = np.append(self.a_lis, data8, axis=0)
        # else:
        #     if self.first:
        #         for i in range(8):
        #             wavio.write(os.path.join(self.save_dir + "out_{}.wav".format(i)), self.a_lis.T[i], 16000 ,sampwidth=3)
        #         #wavio.write("out.wav", self.a_lis, 16000 , sampwidth=3)
        #         self.first = False
        #         print("ok")
        #     pass

        #part2
        #self.reshaped_buffer = data16.reshape(-1,8).T
        #part1
        self.reshaped_buffer = self.audio_prefetch_buffer_sparse.reshape(-1,8).T
        #print(self.reshaped_buffer.shape) #(8,1000)
        #print(self.window_function.shape) #(1000,)

        #fft
        #self.f = self.audio_prefetch_buffer_sparse[0::8] * self.window_function
        #print("before:{}".format(self.reshaped_buffer))
        self.f = self.reshaped_buffer * self.window_function
        #print("before:{}".format(self.f))
        self.F = np.fft.fft(self.f, axis=1)
        phase = np.angle(self.F)
        self.Amp = np.log(np.abs(self.F))
        # print(self.Amp.shape) (8,1000)

        if self.mean_flag:
            self.Amp -= self.mean_amplitude
            #print(self.Amp)

        if self.first_100 < 200:
            self.mean_amplitude += self.Amp
            #self.a_lis = np.append(self.a_lis, data8, axis=0)
        else:
            if self.first:
                self.first = False
                self.mean_amplitude /= 200
                #print(self.mean_amplitude.shape)
                #self.mean_amplitude = np.average(self.mean_amplitude, axis=0)
                print("ok")
                self.mean_flag = True
                
                # for i in range(8):
                #    wavio.write(os.path.join(self.save_dir + "out_{}.wav".format(i)), self.a_lis.T[i], 16000 ,sampwidth=3)
        self.first_100 += 1

        reverse = np.exp(self.Amp)
        reverse = np.array(reverse * np.cos(phase) + (reverse * np.sin(phase))*1.j)
        reverse = np.fft.ifft(reverse)
        reverse = reverse.real
        reverse = reverse / self.window_function
        #print("after:{}".format(reverse))
        #print(reverse.shape) #(8,1000)
        
        self.dhw = HarkWave()
        self.dhw.header.stamp = rospy.Time.now()
        self.dhw.header.frame_id = "tamago"
        #int_array = [ord(n) for n in msg.data]

        #print(reverse.shape) #(8,1000)
        for i in range(8):
            #harkwaveval = HarkWaveVal(wavedata = self.audio_prefetch_buffer_sparse[i::8])
            #harkwaveval = HarkWaveVal(wavedata = self.reshaped_buffer[i])
            harkwaveval = HarkWaveVal(wavedata = reverse[i])
            self.dhw.src.append(harkwaveval)
        self.dhw.nch = 8 # 8 channels
        self.dhw.length = 320
        self.dhw.data_bytes = 320 * 8 * 4 # float(4bytes) times length of data
        self.pub.publish(self.dhw)

if __name__ == "__main__":
    rospy.init_node("audiodata_to_harkwave")
    atoh = AudioDataToHarkWave()
    rospy.spin()
