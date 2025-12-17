#!/usr/bin/env python
import rospy, numpy as np, struct
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

def pub_depth(pub, h=270, w=480, dist=5.0):
    depth = np.full((h, w), dist, dtype=np.float32)  # meters
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "camera"
    msg.height, msg.width = h, w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = w * 4
    msg.data = depth.tobytes()
    pub.publish(msg)

def pub_odom(pub, t):
    odom = Odometry()
    odom.header.stamp = rospy.Time.now()
    odom.header.frame_id = "world"
    # Position: circle path example
    x, y, z = np.cos(t)*0.0, np.sin(t)*0.0, 2.0
    odom.pose.pose.position.x = x
    odom.pose.pose.position.y = y
    odom.pose.pose.position.z = z
    # Orientation: yaw only
    q = quaternion_from_euler(0.0, 0.0, 0.0)  # roll,pitch,yaw (rad)
    odom.pose.pose.orientation = Quaternion(*q)
    # Linear velocity
    odom.twist.twist.linear.x = 0.0
    odom.twist.twist.linear.y = 0.0
    odom.twist.twist.linear.z = 0.0
    pub.publish(odom)

def pub_goal(pub, x=50.0, y=0.0, z=2.0):
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "world"
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation = Quaternion(0,0,0,1)
    pub.publish(goal)

if __name__ == "__main__":
    rospy.init_node("yopo_input_publisher")
    depth_pub = rospy.Publisher("/depth_image", Image, queue_size=1)
    odom_pub = rospy.Publisher("/sim/odom", Odometry, queue_size=1)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    rate_depth = rospy.Rate(30)  # ~30 Hz depth
    rate_ctrl = rospy.Rate(50)   # optional odom ~50 Hz
    t = 0.0
    sent_goal = False
    while not rospy.is_shutdown():
        pub_depth(depth_pub)
        pub_odom(odom_pub, t)
        t += 0.02
        if not sent_goal:
            pub_goal(goal_pub)  # send once
            sent_goal = True
        rate_depth.sleep()