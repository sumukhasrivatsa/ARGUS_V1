# in sam_test_env
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class Saver(Node):
    def __init__(self):
        super().__init__('saver')
        self.bridge = CvBridge()
        self.done = False
        self.create_subscription(Image, '/rgbd_camera/image', self.cb, 10)
    def cb(self, msg):
        if self.done: return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite('/tmp/fresh_frame1.png', img)
        print(f'Saved: {img.shape}')
        self.done = True

rclpy.init()
node = Saver()
rclpy.spin(node)