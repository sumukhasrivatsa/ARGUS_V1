#!/usr/bin/env python3
"""
planner_client.py — ARGUS v1

Listens for goal pose (/argus/goal_pose) and soft obstacle list
(/argus/soft_obstacles) from the perception node.

When a goal arrives:
  1. Plan with ALL obstacles in scene (hard + soft).
  2. If planning fails, allow collision with the lowest-priority soft
     obstacle via the Allowed Collision Matrix (ACM) and retry.
  3. Keep relaxing one obstacle at a time until a plan is found.
  4. Execute the first successful plan.
  5. Restore the ACM after execution (set those cells back to False).

The objects stay in the scene the whole time — the ACM just tells the
collision checker to ignore specific pairs.
"""

import queue           # ADDED — goal queue for planning thread
import threading       # ADDED — planning thread + threading.Event for futures

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    CollisionObject,
    WorkspaceParameters,
    PlanningScene,
    AllowedCollisionMatrix,
    AllowedCollisionEntry,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header, String

# UR5e link names — these are the links we allow collision with
# when relaxing a soft obstacle
UR5E_LINKS = [
    "base_link_inertia",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
]


class PlannerClient(Node):

    def __init__(self):
        super().__init__('planner_client')

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/argus/goal_pose',
            self.goal_callback, 10)

        self.create_subscription(
            String, '/argus/soft_obstacles',
            self.soft_obstacles_callback, 10)

        # ── MoveIt2 action client ────────────────────────────────────────────
        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self.get_logger().info('Waiting for /move_action...')
        self._move_client.wait_for_server()
        self.get_logger().info('/move_action ready.')

        # ── MoveIt2 planning scene services ─────────────────────────────────
        self._get_scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene')
        self._apply_scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')

        self.get_logger().info('Waiting for planning scene services...')
        self._get_scene_client.wait_for_service()
        self._apply_scene_client.wait_for_service()
        self.get_logger().info('Planning scene services ready.')

        # ── state ────────────────────────────────────────────────────────────
        self.latest_goal: PoseStamped | None = None
        self.soft_obstacles: list[str]        = []

        # ADDED — goal queue: goal_callback drops goals here, planning thread picks them up
        self._goal_queue: queue.Queue = queue.Queue()

        # ADDED — planning thread: runs separately from the ROS executor
        # so it can block freely without causing "Executor already spinning"
        self._planning_thread = threading.Thread(
            target=self._planning_loop, daemon=True)
        self._planning_thread.start()

        self.get_logger().info('Planner client ready. Waiting for goal...')

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks — called by the ROS executor. Must return immediately.
    # ─────────────────────────────────────────────────────────────────────────

    def soft_obstacles_callback(self, msg: String):
        if msg.data:
            self.soft_obstacles = [s for s in msg.data.split(',') if s]
        else:
            self.soft_obstacles = []

    def goal_callback(self, msg: PoseStamped):
        self.latest_goal = msg
        self._goal_queue.put(msg)   # CHANGED — just drop in queue, return immediately
        #now we have a new goal, hence we are planning and executing it

    # ─────────────────────────────────────────────────────────────────────────
    # ADDED — Planning loop runs in its own thread, can block freely
    # ─────────────────────────────────────────────────────────────────────────

    def _planning_loop(self):
        while rclpy.ok():
            try:
                goal = self._goal_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self.latest_goal = goal
            self.plan_and_execute()

    # ─────────────────────────────────────────────────────────────────────────
    # Main planning loop with ACM relaxation
    # ─────────────────────────────────────────────────────────────────────────

    def plan_and_execute(self):
        if self.latest_goal is None:
            return

        relaxed_obstacles: list[str] = []

        self.get_logger().info(
            f'Planning to ({self.latest_goal.pose.position.x:.3f}, '
            f'{self.latest_goal.pose.position.y:.3f}, '
            f'{self.latest_goal.pose.position.z:.3f})')

        # ── Attempt 1: plan with all obstacles as hard constraints ────────────
        success = self._send_plan_goal(self.latest_goal)

        if success:
            self.get_logger().info('Plan succeeded on first attempt.')
            return

        # ── Relaxation loop: allow collision with soft obstacles one by one ───
        self.get_logger().warn(
            'Initial plan failed. Relaxing soft obstacles via ACM...')

        for obstacle_name in self.soft_obstacles:
            self.get_logger().warn(
                f'  Allowing collision with: {obstacle_name}')
            self._allow_collision(obstacle_name)
            relaxed_obstacles.append(obstacle_name)

            success = self._send_plan_goal(self.latest_goal)

            if success:
                self.get_logger().info(
                    f'Plan succeeded after relaxing: {relaxed_obstacles}')
                break
        else:
            self.get_logger().error(
                'No plan found even after relaxing all soft obstacles.')

        # ── Restore ACM — disallow collisions that were relaxed ───────────────
        for obstacle_name in relaxed_obstacles:
            self._disallow_collision(obstacle_name)
            self.get_logger().info(f'ACM restored for: {obstacle_name}')

    # ─────────────────────────────────────────────────────────────────────────
    # ACM modification
    # ─────────────────────────────────────────────────────────────────────────

    def _get_current_acm(self) -> AllowedCollisionMatrix | None:
        """
        Fetch the current ACM from MoveIt2's planning scene.
        Returns the ACM object, or None if the service call fails.
        """
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX

        future = self._get_scene_client.call_async(req)
        self._wait_for_future(future)

        if future.result() is None:
            self.get_logger().error('GetPlanningScene service call failed.')
            return None

        return future.result().scene.allowed_collision_matrix

    def _apply_acm(self, acm: AllowedCollisionMatrix):
        """
        Send a modified ACM back to MoveIt2 as a diff update.
        Only the ACM changes — everything else in the scene is untouched.
        """
        scene         = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm

        req       = ApplyPlanningScene.Request()
        req.scene = scene

        future = self._apply_scene_client.call_async(req)
        self._wait_for_future(future)

        if future.result() is None or not future.result().success:
            self.get_logger().error('ApplyPlanningScene service call failed.')

    def _allow_collision(self, object_name: str):
        """
        Add entries to the ACM allowing the robot to collide with object_name.
        The object stays in the scene — the collision checker just ignores it.
        """
        acm = self._get_current_acm()
        if acm is None:
            return

        # Add object_name to the ACM if it's not already there
        if object_name not in acm.entry_names:
            acm.entry_names.append(object_name)
            # Extend all existing rows with a new False column
            for entry in acm.entry_values:
                entry.enabled.append(False)
            # Add a new row for this object (all False to start)
            new_row = AllowedCollisionEntry()
            new_row.enabled = [False] * len(acm.entry_names)
            acm.entry_values.append(new_row)

        obj_idx = acm.entry_names.index(object_name)

        # Set ACM cells to True for every pair of (object, robot_link)
        # The matrix is symmetric: [obj][link] = True AND [link][obj] = True
        for i, name in enumerate(acm.entry_names):
            if name in UR5E_LINKS:
                acm.entry_values[obj_idx].enabled[i] = True   # object row
                acm.entry_values[i].enabled[obj_idx] = True   # link row

        self._apply_acm(acm)
        self.get_logger().info(
            f'ACM: allowed collision between robot and [{object_name}]')

    def _wait_for_future(self, future):
        # CHANGED — use threading.Event instead of rclpy.spin_once
        # We are in the PLANNING THREAD (not the executor thread).
        # The executor runs in the main thread and calls future done-callbacks
        # when responses arrive. threading.Event lets us wait without touching
        # the executor at all — no "Executor already spinning" crash possible.
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        event.wait()

    def _disallow_collision(self, object_name: str):
        """
        Reverse _allow_collision — set those ACM cells back to False.
        The planner will hard-avoid the object again after this.
        """
        acm = self._get_current_acm()
        if acm is None:
            return

        if object_name not in acm.entry_names:
            return  # nothing to restore

        obj_idx = acm.entry_names.index(object_name)

        for i, name in enumerate(acm.entry_names):
            if name in UR5E_LINKS:
                acm.entry_values[obj_idx].enabled[i] = False
                acm.entry_values[i].enabled[obj_idx] = False

        self._apply_acm(acm)
        self.get_logger().info(
            f'ACM: restored hard constraint for [{object_name}]')

    # ─────────────────────────────────────────────────────────────────────────
    # MoveGroup action
    # ─────────────────────────────────────────────────────────────────────────

    def _send_plan_goal(self, goal_pose: PoseStamped) -> bool:
        """
        Send a planning + execution goal to MoveIt2.
        Returns True if planning and execution both succeeded.
        """
        goal_msg = MoveGroup.Goal()

        req                               = MotionPlanRequest()
        req.group_name                    = 'ur_manipulator'
        req.num_planning_attempts         = 10
        req.allowed_planning_time         = 5.0
        req.max_velocity_scaling_factor   = 0.1
        req.max_acceleration_scaling_factor = 0.1

        ws                 = WorkspaceParameters()
        ws.header.frame_id = 'world'
        ws.min_corner.x    = -1.5
        ws.min_corner.y    = -1.5
        ws.min_corner.z    = -0.1
        ws.max_corner.x    =  1.5
        ws.max_corner.y    =  1.5
        ws.max_corner.z    =  2.0
        req.workspace_parameters = ws

        # position constraint — end effector to goal point
        pos_c                      = PositionConstraint()
        pos_c.header.frame_id      = goal_pose.header.frame_id
        pos_c.link_name            = 'tool0'
        pos_c.weight               = 1.0

        bv                  = BoundingVolume()
        tol                 = SolidPrimitive()
        tol.type            = SolidPrimitive.BOX
        tol.dimensions      = [0.02, 0.02, 0.02]   # 2cm tolerance
        bv.primitives.append(tol)
        bv.primitive_poses.append(goal_pose.pose)
        pos_c.constraint_region = bv

        # orientation constraint — tool pointing down
        ori_c                            = OrientationConstraint()
        ori_c.header.frame_id            = goal_pose.header.frame_id
        ori_c.link_name                  = 'tool0'
        ori_c.orientation                = goal_pose.pose.orientation
        ori_c.absolute_x_axis_tolerance  = 0.5
        ori_c.absolute_y_axis_tolerance  = 0.5
        ori_c.absolute_z_axis_tolerance  = 0.5
        ori_c.weight                     = 0.5

        req.goal_constraints.append(
            Constraints(
                position_constraints=[pos_c],
                orientation_constraints=[ori_c],
            )
        )

        goal_msg.request                              = req
        goal_msg.planning_options.plan_only           = False
        goal_msg.planning_options.replan              = True
        goal_msg.planning_options.replan_attempts     = 3

        # send and wait
        send_future = self._move_client.send_goal_async(goal_msg)
        self._wait_for_future(send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by move_group.')
            return False

        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future)
        result = result_future.result()

        # MoveItErrorCodes.SUCCESS = 1
        success = (result.result.error_code.val == 1)
        if not success:
            self.get_logger().warn(
                f'Planning/execution failed. Error code: '
                f'{result.result.error_code.val}')
        return success


def main(args=None):
    rclpy.init(args=args)
    node = PlannerClient()
    try:
        rclpy.spin(node)   # main thread — executor lives here, never blocked
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()