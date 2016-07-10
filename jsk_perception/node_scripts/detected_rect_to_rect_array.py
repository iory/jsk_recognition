#!/usr/bin/env python
import rospy

from jsk_recognition_msgs.msg import DetectedRectArray
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32

def detected_rect_array_cb(detected_rect_array):
    # rect_array = RectArray()
    # rect_array.header = detected_rect_array.header
    polygon = PolygonStamped()
    current_width = 0
    current_height = 0
    a, b = Point32(), Point32()
    for detected_rect in detected_rect_array.rects:
        if detected_rect.label == 'person':
            x = detected_rect.x
            y = detected_rect.y
            width = detected_rect.width
            height = detected_rect.height
            if current_height < height:
                current_height = height
                a = Point32(x=x, y=y, z=0.0)
                b = Point32(x=x+width, y=y+height, z=0.0)
            # rect_array.rects.append(Rect(x=x, y=y, width=width, height=height))
    polygon.header = detected_rect_array.header
    p = Polygon()
    p.points.append(a)
    p.points.append(b)
    polygon.polygon = p
    rect_array_pub.publish(polygon)

if __name__ == "__main__":
    rospy.init_node('detected_rect_array_to_rect_array')
    # rect_array_pub = rospy.Publisher('~output', RectArray, queue_size=1)
    rect_array_pub = rospy.Publisher('~output', PolygonStamped, queue_size=1)
    rospy.Subscriber("~input", DetectedRectArray, detected_rect_array_cb, queue_size=1)
    rospy.spin()
