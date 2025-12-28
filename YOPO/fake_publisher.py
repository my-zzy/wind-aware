#!/usr/bin/env python3
"""
Fake ROS publisher for testing test_yopo_ros.py
Publishes simulated odometry and depth images
To see the results, run:
# Watch odometry feedback
rostopic echo /sim/odom

# Watch depth images
rostopic echo /depth_image

# Watch velocity commands published to AirSim
rostopic echo /yopo/cmd_vel

# Watch control commands
rostopic echo /so3_control/pos_cmd

# List all active topics
rostopic list

# Check publishing rates
rostopic hz /depth_image
rostopic hz /sim/odom
rostopic hz /yopo/cmd_vel
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2


class FakePublisher:
    def __init__(self):
        rospy.init_node('fake_publisher', anonymous=False)
        
        # Publishers
        self.odom_pub = rospy.Publisher('/sim/odom', Odometry, queue_size=10)
        self.depth_pub = rospy.Publisher('/depth_image', Image, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # Parameters
        self.depth_width = 160
        self.depth_height = 96
        self.bridge = CvBridge()
        
        # Simulation state
        self.position = np.array([0.0, 0.0, 2.5])  # Start at (0, 0, 2.5)
        self.velocity = np.array([0.5, 0.0, 0.0])  # Moving forward
        self.yaw = 0.0
        self.time_step = 0.0
        
        print("Fake Publisher Ready!")
        print("Publishing:")
        print("  - Odometry on /sim/odom (50 Hz)")
        print("  - Depth images on /depth_image (30 Hz)")
        print("  - Goal on /move_base_simple/goal (once at start)")
        print("\nSimulating drone moving from (0,0,2.5) toward (50,0,2.5)")
        
        # Publish initial goal
        rospy.sleep(1.0)
        self.publish_goal()
        
        # Timers
        self.odom_timer = rospy.Timer(rospy.Duration(0.02), self.publish_odom)  # 50 Hz
        self.depth_timer = rospy.Timer(rospy.Duration(0.033), self.publish_depth)  # ~30 Hz
        
        rospy.spin()
    
    def publish_odom(self, event):
        """Publish odometry at 50 Hz - simulates moving drone"""
        # Update position (simple integration)
        dt = 0.02
        self.time_step += dt
        
        # Simple trajectory: move forward with slight sinusoidal motion
        self.position[0] += self.velocity[0] * dt
        self.position[1] = 0.3 * np.sin(self.time_step * 0.5)  # Slight side-to-side
        self.yaw = 0.1 * np.sin(self.time_step * 0.3)  # Slight yaw oscillation
        
        # Create Odometry message
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        
        # Position
        odom.pose.pose.position.x = self.position[0]
        odom.pose.pose.position.y = self.position[1]
        odom.pose.pose.position.z = self.position[2]
        
        # Orientation (quaternion from yaw)
        cy = np.cos(self.yaw * 0.5)
        sy = np.sin(self.yaw * 0.5)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = sy
        odom.pose.pose.orientation.w = cy
        
        # Velocity
        odom.twist.twist.linear.x = self.velocity[0]
        odom.twist.twist.linear.y = 0.1 * np.cos(self.time_step * 0.5)
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.03 * np.cos(self.time_step * 0.3)
        
        self.odom_pub.publish(odom)
    
    def publish_depth(self, event):
        """Publish depth image at ~30 Hz - simulates depth camera"""
        # Generate synthetic depth image
        # Create a simple scene with some obstacles
        depth_image = np.ones((self.depth_height, self.depth_width), dtype=np.float32) * 10.0
        
        # Add some random obstacles (closer regions)
        num_obstacles = 5
        for _ in range(num_obstacles):
            cx = np.random.randint(20, self.depth_width - 20)
            cy = np.random.randint(20, self.depth_height - 20)
            radius = np.random.randint(10, 25)
            distance = np.random.uniform(2.0, 6.0)
            
            y, x = np.ogrid[:self.depth_height, :self.depth_width]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            depth_image[mask] = distance
        
        # Add ground (bottom of image should be closer)
        ground_gradient = np.linspace(3.0, 8.0, self.depth_height)
        for i in range(self.depth_height):
            if i > self.depth_height * 0.6:  # Bottom 40% is ground
                depth_image[i, :] = np.minimum(depth_image[i, :], ground_gradient[i])
        
        # Add some noise
        noise = np.random.normal(0, 0.1, depth_image.shape).astype(np.float32)
        depth_image = np.clip(depth_image + noise, 0.5, 15.0)
        
        # Convert to ROS Image message
        # Depth images are typically encoded as 32FC1 or 16UC1
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header.stamp = rospy.Time.now()
        depth_msg.header.frame_id = "camera"
        
        self.depth_pub.publish(depth_msg)
    
    def publish_goal(self):
        """Publish goal once at startup"""
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "world"
        goal.pose.position.x = 50.0
        goal.pose.position.y = 0.0
        goal.pose.position.z = 2.5
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
        print(f"Published goal: ({goal.pose.position.x}, {goal.pose.position.y}, {goal.pose.position.z})")


if __name__ == "__main__":
    try:
        FakePublisher()
    except rospy.ROSInterruptException:
        pass
