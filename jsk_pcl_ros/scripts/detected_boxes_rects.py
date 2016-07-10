#!/usr/bin/env python
import rospy


from jsk_recognition_msgs.msg import DetectedRectArray
from jsk_recognition_msgs.msg import DetectedRect
from jsk_recognition_msgs.msg import DetectedRectArrayWithBoundingBoxArray
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import BoundingBoxArrayWithRectArray
from jsk_recognition_msgs.msg import BoundingBoxArray

def boxarray_cb(boxarray):
    bba = boxarray.boxes
    bba.header =boxarray.header
    BoxPub.publish(bba)

if __name__ == "__main__":
    rospy.init_node('decompose_detected_boxes_rects')
    BoxPub = rospy.Publisher('~output', BoundingBoxArray, queue_size=1)
    rospy.Subscriber("~input", DetectedRectArrayWithBoundingBoxArray, boxarray_cb)
    rospy.spin()
