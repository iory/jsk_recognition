#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import BoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import Point32
from jsk_recognition_msgs.msg import PolygonArray

import numpy as np

class PolygonToBoundingBox(ConnectionBasedTransport):
    def __init__(self):
        super(PolygonToBoundingBox, self).__init__()
        self.pub_bba_ = self.advertise('~output', BoundingBoxArray, queue_size=1)
    def subscribe(self):
        self.sub_polygon_ = rospy.Subscriber('~input', PolygonArray, self.callback)
        self.subs = [self.sub_polygon_]
    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
    def callback(self, msg):
        bba = BoundingBoxArray()
        bba.header = msg.header
        bba.header.stamp = rospy.Time.now()
        for polygon in msg.polygons:
            center_point_x = 0
            center_point_y = 0
            center_point_z = 0
            N = len(polygon.polygon.points)

            for point in polygon.polygon.points:
                center_point_x += point.x
                center_point_y += point.y
                center_point_z += point.z

            maximum_length_edge = 0.0
            for pa, pb in zip(polygon.polygon.points, polygon.polygon.points[1:]):
                maximum_length_edge = max(maximum_length_edge, \
                                          np.sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2))

            center_point_x /= N
            center_point_y /= N
            center_point_z /= N

            bb = BoundingBox()
            bb.header = bba.header

            bounding_box_height = 1.0
            offset = 0.4
            center_point_z += bounding_box_height / 2 + offset
            bb.pose.position = Point32(center_point_x,
                                       center_point_y,
                                       center_point_z)
            bb.dimensions.x = maximum_length_edge
            bb.dimensions.y = maximum_length_edge
            bb.dimensions.z = bounding_box_height
            bba.boxes.append(bb)
        self.pub_bba_.publish(bba)

def main():
    rospy.init_node('polygon_to_boundingbox')
    app = PolygonToBoundingBox()
    rospy.spin()

if __name__ == "__main__":
    main()
