#!/usr/bin/env python3
"""
perception_node.py

Subscribes to the overhead RGBD camera, runs YOLO-World to detect objects
by name, projects detections into 3D world-frame coordinates using camera
intrinsics + a static transform, and publishes CollisionObjects to MoveIt2.

Change detection: only publishes a new CollisionObject when an object has
moved more than CHANGE_THRESHOLD metres since the last publish. This means
MoveIt2 only gets interrupted when the scene genuinely changes.

Labels and weights are hardcoded here for now.
The LLM policy node (built separately) will publish to /policy_update
and this node will update its labels and weights from there.
"""

import math
import os
import subprocess

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PointStamped, PoseStamped, Quaternion, TransformStamped
from moveit_msgs.msg import CollisionObject
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header, String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from ultralytics import YOLOWorld

# ─────────────────────────────────────────────────────────────────────────────
# Camera constants — must match table_scene.sdf <pose> exactly
# Format: x y z roll pitch yaw (radians)
# If I move the camera in the SDF, only update these two lines.
# Everything downstream (TF, projection) adapts automatically.
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_TRANSLATION =  (0.0 - (-0.5), -1.4, 0.825)  # x: camera_x - robot_spawn_x   # 1.6 - 0.775    # x, y, z in world frame.
CAMERA_EULER_RPY = (0.0, 0.5, 1.5708)    # extra -90° yaw to align Gazebo→ROS convention  # roll, pitch, yaw in radians

CAMERA_FOV_RAD     = 1.20               # horizontal FOV — match SDF
IMAGE_WIDTH_PX     = 640
IMAGE_HEIGHT_PX    = 480

TIMER_PERIOD_S     = 5.0    # seconds between perception runs
CHANGE_THRESHOLD   = 0.05   # 5 cm — minimum move to count as a real change
MIN_CONFIDENCE     = 0.25   # YOLO-World confidence threshold

# Weight threshold — boundary between hard and soft obstacles.
# weight <= this → hard obstacle (planner can NEVER relax it)
# weight > this and < 0 → soft obstacle (planner CAN relax via ACM)
HARD_THRESHOLD     = -600


# -----------------------------------------------------------------------------
# Node
# -----------------------------------------------------------------------------

class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.last_frame_time = None
        self.last_processed_time = None

        # -- camera intrinsics ------------------------------------------------
        # cx, cy are the optical centre — the exact middle of the image
        self.cx = IMAGE_WIDTH_PX  / 2.0
        self.cy = IMAGE_HEIGHT_PX / 2.0
        self.fx = self.cx / math.tan(CAMERA_FOV_RAD / 2.0)
        self.fy = self.fx                         # square pixels

        # -- latest frames ----------------------------------------------------
        self.rgb_image   = None
        self.depth_image = None

        # -- labels and weights (LLM node will update these later) -------------
        # weights: negative = avoid, positive = goal, magnitude = priority
        # This dict is the SINGLE source of truth:
        #   weight > 0     → GOAL (robot moves toward it)
        #   weight <= -600 → HARD obstacle (never relaxed)
        #   -600 < w < 0   → SOFT obstacle (planner can relax via ACM)
        self.labels = ["bottle", "vase", "sports ball","box"]
        self.label_weights = {
            "bottle":  -1000,
            "vase":     -400,
            "sports ball":  -200,
            "box":     200,
        }

        # -- soft obstacle tracking (rebuilt every perception cycle) -----------
        # list of {"name": str, "weight": int} — sent to planner_client
        self.soft_obstacles_this_cycle: list[dict] = []

        # -- goal tracking ----------------------------------------------------
        self.goal_label: str | None  = None
        self.goal_xyz:   tuple | None = None

        # -- change detection state -------------------------------------------
        # keyed by label name → (world_x, world_y, world_z)
        self.previous_xyz: dict[str, tuple] = {}

        # -- cv_bridge --------------------------------------------------------
        self.bridge = CvBridge()

        # -- TF2 setup --------------------------------------------------------
        # Step 1: tell TF where the camera lives in the world (once, at startup)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_camera_transform()

        # Step 2: set up a listener so we can ask TF to convert points later
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -- YOLO-World ------------------------------------------------------
        self.get_logger().info('Loading YOLO-World...')
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.model.set_classes(self.labels)
        self.get_logger().info(f'YOLO-World ready. Watching: {self.labels}')

        # -- subscribers -----------------------------------------------------
        self.create_subscription(
            Image, '/rgbd_camera/image',
            self.rgb_callback, 10
        )
        self.create_subscription(
            Image, '/rgbd_camera/depth_image',
            self.depth_callback, 10
        )

        # -- publishers -------------------------------------------------------
        # ALL obstacles (hard + soft) go here. MoveIt2 treats all as hard by default.
        self.collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10
        )

        # Goal pose — planner_client subscribes to this to know where to go
        self.goal_pub = self.create_publisher(
            PoseStamped, '/argus/goal_pose', 10
        )

        # Soft obstacle names — planner_client subscribes to know what it can
        # relax via ACM if planning fails. Comma-separated, sorted by ascending
        # |weight| (relax lowest priority first).
        self.soft_obs_pub = self.create_publisher(
            String, '/argus/soft_obstacles', 10
        )

        self.get_logger().warn(
            'Run image bridge manually: '
            'ros2 run ros_gz_image image_bridge '
            '/rgbd_camera/image /rgbd_camera/depth_image'
        )

        # -- timer ------------------------------------------------------------
        self.create_timer(TIMER_PERIOD_S, self.process)
        self.get_logger().info('Perception node started.')

        # publish the table itself so it's visible in RViz
        #self.create_timer(2.0, self._publish_table)

    # -- TF — broadcast static camera transform once at startup
    # ----------------------------------------------------------------------

    """
    def _publish_table(self):
        co = CollisionObject()
        co.header = Header()
        co.header.frame_id = 'world'
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = 'table'
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.2, 0.8, 0.05]  # matches your SDF table dimensions

        pose = Pose()
        pose.position = Point(x=0.0, y=0.0, z=0.75)  # table top height
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        co.primitives = [box]
        co.primitive_poses = [pose]
        self.collision_pub.publish(co)

    """
    def _broadcast_camera_transform(self):
        """
        Tells the TF system where camera_link is in world frame.
        Must match the camera <pose> in table_scene.sdf exactly.

        After this runs once, any node can ask TF:
        "convert point P from camera_link to world frame"
        and TF knows how to answer — no hardcoded math needed anywhere.
        """
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'        # parent frame
        t.child_frame_id  = 'camera_link'  # the frame we are declaring

        # camera position in world frame (matches SDF translation)
        tx, ty, tz = CAMERA_TRANSLATION
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = tz

        # camera orientation — convert roll/pitch/yaw to quaternion
        # scipy returns [x, y, z, w] which is what ROS2 expects
        r = Rotation.from_euler('xyz', CAMERA_EULER_RPY)
        qx, qy, qz, qw = r.as_quat()
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info(
            f'Camera TF broadcast: pos={CAMERA_TRANSLATION} rpy={CAMERA_EULER_RPY}'
        )

    # -- Subscriber callbacks — store latest frame, do nothing else
    # ----------------------------------------------------------------------

    def rgb_callback(self, msg: Image):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.last_frame_time = msg.header.stamp
        self.last_processed_time = None  # force reprocess every frame

        # save every 10th frame to disk so we can inspect what the node sees
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            cv2.imwrite('/tmp/node_sees.png', self.rgb_image)

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough'
        ).astype(np.float32)

    # -- Main loop — called by timer
    # ----------------------------------------------------------------------

    def process(self):
        self.get_logger().info(f'previous_xyz state: {self.previous_xyz}')
        self.get_logger().info('Timer fired')

        if self.rgb_image is None:
            self.get_logger().info('rgb_image is None — waiting', throttle_duration_sec=2.0)
            return
        if self.depth_image is None:
            self.get_logger().info('depth_image is None — waiting', throttle_duration_sec=2.0)
            return

        if self.last_frame_time == self.last_processed_time:
            self.get_logger().info('Same frame as last time — skipping', throttle_duration_sec=2.0)
            return

        self.get_logger().info('Running YOLO-World...')
        results = self.model(self.rgb_image, verbose=True, conf=MIN_CONFIDENCE)
        n_boxes = len(results[0].boxes) if results[0].boxes is not None else 0
        self.get_logger().info(f'YOLO done. Boxes: {n_boxes}')

        if n_boxes == 0:
            self.get_logger().warn('No detections this frame')
            self.last_processed_time = self.last_frame_time
            return

        # reset soft obstacle tracking for this cycle
        self.soft_obstacles_this_cycle = []

        for box in results[0].boxes:
            label_idx  = int(box.cls[0])
            label_name = self.labels[label_idx]
            confidence = float(box.conf[0])
            self.get_logger().info(f'Detection: {label_name} conf={confidence:.2f}')

            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            cx_px    = (x_min + x_max) / 2.0
            cy_px    = (y_min + y_max) / 2.0
            box_w_px = x_max - x_min
            box_h_px = y_max - y_min

            depth_val = self._get_depth(cx_px, cy_px)
            self.get_logger().info(f'Depth at centroid: {depth_val}')

            if depth_val is None:
                self.get_logger().warn(f'Invalid depth for {label_name} — skipping')
                continue
            
            # NEW
            width  = max((box_w_px * depth_val) / self.fx, 0.05)
            height = max((box_h_px * depth_val) / self.fy, 0.05)

            size_x = width
            size_y = width
            size_z = height

            cam_x, cam_y, cam_z = self._pixel_to_camera(cx_px, cy_px, depth_val)
            result = self._camera_to_world(cam_z, -cam_x, -cam_y)
            if result is None:
                continue
            world_x, world_y, _ = result

            # Snap bottom to table.
            TABLE_Z = 0.0
            world_z = TABLE_Z + size_z / 2.0
            self.get_logger().info(
                f'{label_name} → world ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})'
            )

            new_xyz = (world_x, world_y, world_z)
            changed = self._has_changed(label_name, new_xyz)
            self.get_logger().info(f'Changed: {changed}')

            if not changed:
                continue

            self.previous_xyz[label_name] = new_xyz
            self.get_logger().info(f'Publishing CollisionObject for {label_name}')

            #first figure out if its a goal or an obstacle
            # Use label_weights directly — positive weight = goal, negative = obstacle
            weight = self.label_weights.get(label_name)

            if weight > 0:
                # means, the position of the goal changed
                self.goal_label = label_name
                self.goal_xyz   = new_xyz
                self._publish_goal(world_x, world_y, world_z, size_z)
                self.get_logger().info(
                    f'  ✓ GOAL: {label_name} at '
                    f'({world_x:.3f}, {world_y:.3f}, {world_z:.3f})')
            else:
                # means,the position of the obstacle changed
                self._publish_collision_object(
                    label_name, world_x, world_y, world_z,
                    size_x, size_y, size_z)

                #not only this, we now need to tell, if, the collision is allowed, you can get a true/false from a function
                is_soft = self._check_collision_allowance(label_name)

                #now,only if the collision is allowed, add it to soft obstacles list for planner_client
                if is_soft:
                    # SOFT obstacle — planner_client can relax this via ACM
                    self.soft_obstacles_this_cycle.append({
                        'name': label_name,
                        'weight': weight,
                    })
                    self.get_logger().info(
                        f'  ~ SOFT obstacle: {label_name} (weight={weight})')
                else:
                    # HARD obstacle — planner_client can never relax this
                    # No topic needed — it's already in the planning scene as a wall
                    self.get_logger().info(
                        f'  ✗ HARD obstacle: {label_name} (weight={weight})')

        # publish soft obstacle list so planner_client knows what to relax
        self._publish_soft_obstacles()

        self.last_processed_time = self.last_frame_time

    # -- Helpers ------------------------------------------------------------
    # ----------------------------------------------------------------------
   

    def _check_collision_allowance(self, class_name: str) -> bool:
        #basically just checks the label_weights dict against the class label, if the weight <= -600, collision NOT allowed (hard), if > -600 & < 0, collision Allowed (soft)
        weight = self.label_weights.get(class_name, -500)
        return HARD_THRESHOLD < weight < 0

    def _publish_goal(self, x: float, y: float, z: float, obj_height: float):
        """
        Publish goal pose to /argus/goal_pose.
        Hovers 10cm above the object top. End effector points down.
        planner_client.py subscribes to this and sends it to MoveIt2.
        """
        pose                     = PoseStamped()
        pose.header.frame_id     = 'world'
        pose.header.stamp        = self.get_clock().now().to_msg()
        pose.pose.position.x     = x
        pose.pose.position.y     = y
        pose.pose.position.z     = z + obj_height / 2.0 + 0.10   # 10cm above top
        # orientation: tool pointing straight down (90° rotation around Y)
        pose.pose.orientation.x  = 0.0
        pose.pose.orientation.y  = 0.707
        pose.pose.orientation.z  = 0.0
        pose.pose.orientation.w  = 0.707
        self.goal_pub.publish(pose)

    def _publish_soft_obstacles(self):
        """
        Publish soft obstacle names to /argus/soft_obstacles.
        Sorted by ascending |weight| so planner relaxes lowest priority first.
        planner_client.py subscribes to this.
        """
        sorted_soft = sorted(
            self.soft_obstacles_this_cycle,
            key=lambda obj: abs(obj['weight']))
        names = ','.join(obj['name'] for obj in sorted_soft)

        msg      = String()
        msg.data = names
        self.soft_obs_pub.publish(msg)

        if names:
            self.get_logger().info(f'  Soft obstacles (relax order): {names}')

    def _get_depth(self, cx_px: float, cy_px: float) -> float | None:
        """Return the depth value (metres) at a pixel. Returns None if invalid."""
        row = int(cy_px)
        col = int(cx_px)

        if row < 0 or row >= self.depth_image.shape[0]:
            return None
        if col < 0 or col >= self.depth_image.shape[1]:
            return None

        depth = float(self.depth_image[row, col])

        if depth <= 0.0 or np.isnan(depth) or np.isinf(depth):
            return None

        return depth

    def _pixel_to_camera(
        self, cx_px: float, cy_px: float, depth: float
    ) -> tuple[float, float, float]:
        """
        Standard pinhole camera projection.
        Converts a 2D pixel + depth value into a 3D point in camera frame.

        Camera frame origin = camera lens centre.
        Camera Z axis points INTO the scene (toward the table).
        This math is the same regardless of where the camera is mounted.
        The coordinate frame conversion happens separately in _camera_to_world.
        """
        x = (cx_px - self.cx) * depth / self.fx
        y = (cy_px - self.cy) * depth / self.fy
        z = depth
        return x, y, z

    def _camera_to_world(self, gz_x, gz_y, gz_z):
        
        from geometry_msgs.msg import PointStamped
        
        point = PointStamped()
        point.header.frame_id = 'camera_link'
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = gz_x
        point.point.y = gz_y
        point.point.z = gz_z
        
        try:
            transformed = self.tf_buffer.transform(
                point, 'world',
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}')
            return None

    def _has_changed(self, label_name: str, new_xyz: tuple) -> bool:
        """
        Returns True if this object has moved more than CHANGE_THRESHOLD
        since the last publish, or if it is being seen for the first time.

        This is the change detection gate — stops MoveIt2 from being flooded
        with identical CollisionObjects on every timer tick.
        """
        if label_name not in self.previous_xyz:
            return True  # first detection of this object

        prev = self.previous_xyz[label_name]
        distance = math.sqrt(
            (new_xyz[0] - prev[0]) ** 2 +
            (new_xyz[1] - prev[1]) ** 2 +
            (new_xyz[2] - prev[2]) ** 2
        )
        return distance > CHANGE_THRESHOLD

    def _publish_collision_object(
        self,
        label_name: str,
        x: float, y: float, z: float,
        size_x: float, size_y: float, size_z: float,
    ):
        """
        Build and publish a MoveIt2 CollisionObject.

        co.id = label_name → MoveIt2 uses this as a unique key.
        Publishing the same label again with a new pose UPDATES the object
        in the planning scene rather than duplicating it.

        frame_id = 'world' tells MoveIt2 these coordinates are already
        in world frame. MoveIt2 uses TF internally to understand what that
        means relative to the robot — I never need to convert manually.
        """
        co = CollisionObject()

        co.header          = Header()
        co.header.frame_id = 'world'
        co.header.stamp    = self.get_clock().now().to_msg()
        co.id              = label_name
        co.operation       = CollisionObject.ADD

        box            = SolidPrimitive()
        box.type       = SolidPrimitive.BOX
        box.dimensions = [size_x, size_y, size_z]

        pose                  = Pose()
        pose.position         = Point(x=x, y=y, z=z)
        pose.orientation      = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        co.primitives       = [box]
        co.primitive_poses  = [pose]

        self.collision_pub.publish(co)


# -- Entry point ----------------------------------------------------------
# --------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()