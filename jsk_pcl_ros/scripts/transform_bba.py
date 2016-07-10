#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import tf
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import BoundingBoxArray
import numpy as np

class transform_bba(ConnectionBasedTransport):
    def __init__(self):
        super(transform_bba, self).__init__()
        self.pub_bba_ = self.advertise('~output', BoundingBoxArray, queue_size=1)
    def subscribe(self):
        self.sub_bba_ = rospy.Subscriber('~input', BoundingBoxArray, self.callback)
        self.subs_ = [self.sub_bba_]
    def unsubscribe(self):
        for sub in self.subs_:
            sub.unregister()
    def callback(self, msg):
        bba = BoundingBoxArray()
        bba.header = msg.header
        bba.header.stamp = rospy.Time.now()
        for bb in msg.boxes:
            quaternion = tf.transformations.quaternion_from_euler(np.deg2rad(90), 0, 0)
            bb.pose.orientation.x = quaternion[0]
            bb.pose.orientation.y = quaternion[1]
            bb.pose.orientation.z = quaternion[2]
            bb.pose.orientation.w = quaternion[3]
            bba.boxes.append(bb)
        self.pub_bba_.publish(bba)

def main():
    rospy.init_node('transform_bba')
    app = transform_bba()
    rospy.spin()

if __name__ == "__main__":
    main()
