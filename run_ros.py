#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from splat.splat_utils import GSplatLoader
from splatplan.splatplan import SplatPlan
from splatplan.spline_utils import SplinePlanner

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
#from px4_msgs.msg import VehicleOdometry  # Adjusted to use the PX4-specific odometry message
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.duration import Duration
import time

import threading
import sys
import select
import tty
import termios

from ros_utils import make_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ControlNode(Node):

    def __init__(self, mode='open-loop'):
        super().__init__('control_node')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_c = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_incoming = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Set the map name for visualization in RVIZ
        self.map_name = "camera_link"

        self.mode = mode

        # Subscribe to the estimated pose topic. In open-loop experiments, this is not used. In the closed-loop experiments,
        # this can be from the VIO or from Splat-Loc.
        # Subscribe to the odometry topic
        # self.odometry_subscriber = self.create_subscription(
        #     VehicleOdometry,
        #     '/fmu/out/vehicle_odometry',
        #     self.odometry_callback,
        #     qos_profile
        # )
        
        # Publish to the control topic
        self.control_publisher = self.create_publisher(
            Float32MultiArray,
            '/control',
            qos_profile_c
        )

        # Publishes the static voxel grid as a point cloud
        self.pcd_publisher = self.create_publisher(
            PointCloud2, "/gsplat_pcd", 10
        )
        self.timer = self.create_timer(1.0, self.pcd_callback)
        self.pcd_msg = None

        # The state of SplatPlan. This is used to trigger replanning. 
        self.state_publisher = self.create_publisher(String, "/splatplan_state", 10)

        # This publishes the goal pose
        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.goal_timer = self.create_timer(1.0, self.goal_callback)

        # if mode == 'closed-loop':
        #     # This is the timer that triggers replanning
        #     self.replan_timer = self.create_timer(1.0, self.replan)

        ### Initialize variables  ###
        self.fmu_pos = [0.0, 0.0, 0.0]
        
        self.velocity_output = [0.0, 0.0, 0.0]
        self.position_output = [0.0, 0.0, -0.5]
        self.acceleration_output = [0.0, 0.0, 0.0]

        self.goal = [4.0, 0., -0.75]

        self.des_yaw_rate = 0.0
        self.yaw = (-90.0) * 3.14/ 180.0

        self.timer = self.create_timer(1.0 / 50.0, self.publish_control)

        self.start_mission = False

        # Start the keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.key_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        ### SPLATPLAN INITIALIZATION ###
        ############# Specify scene specific data here  #############
        # Points to the config file for the GSplat
        path_to_gsplat = Path('outputs/configs/ros-depth-splatfacto/2024-10-24_153147/config.yml')

        radius = 0.02       # radius of robot
        amax = 1.
        vmax = 1.

        lower_bound = torch.tensor([-4., -2., -1.], device=device)
        upper_bound = torch.tensor([4., 2., 0.], device=device)
        resolution = 50

        #################
        # Robot configuration
        robot_config = {
            'radius': radius,
            'vmax': vmax,
            'amax': amax,
        }

        # Environment configuration (specifically voxel)
        voxel_config = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'resolution': resolution,
        }

        tnow = time.time()
        self.gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)

        print(f'There are {len(self.gsplat.means)} Gaussians in the Splat.')

        spline_planner = SplinePlanner(spline_deg=6, device=device)
        self.planner = SplatPlan(self.gsplat, robot_config, voxel_config, spline_planner, device)
        # self.traj = self.plan_path(self.position_output, self.goal)

        # Publishes the trajectory as a Pose Array
        self.trajectory_publisher = self.create_publisher(PoseArray, "/trajectory", 10)
        self.trajectory_timer = self.create_timer(1.0, self.trajectory_callback)
        self.traj = None

        print("SplatPlan Initialized...")

    def trajectory_callback(self):
        if self.traj is None or self.mode == 'closed-loop':
            traj, output = self.plan_path(self.position_output, self.goal)

            # NOTE: !!! self.traj is the np array of traj( a list)
            self.traj = np.array(traj)

        print('Start:', traj[0])
        print('Goal:', traj[-1])

        poses = []
        yaw = 0.0

        for idx, pt in enumerate(self.traj):
            msg = Pose()
            # msg.header.frame_id = self.map_name
            msg.position.x, msg.position.y, msg.position.z = pt[0], pt[1], pt[2]

            if idx < len(self.traj) - 1:

                diff = self.traj[idx + 1] - self.traj[idx]

                prev_yaw = yaw
                yaw = np.arctan2(diff[1], diff[0])

                closest_k = np.round(-(yaw - prev_yaw) / (2*np.pi))
                yaw = yaw + 2*np.pi*closest_k     

            quat = Rotation.from_euler("z", yaw).as_quat()
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ) = (quat[0], quat[1], quat[2], quat[3])

            poses.append(msg)

        msg = PoseArray()
        msg.header.frame_id = self.map_name
        msg.poses = poses

        self.trajectory_publisher.publish(msg)

    def goal_callback(self):
        msg = PoseStamped()
        msg.header.frame_id = self.map_name

        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.goal[0], self.goal[1], self.goal[2]
        yaw = 0.
        quat = Rotation.from_euler("z", yaw).as_quat()
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = (quat[0], quat[1], quat[2], quat[3])

        self.goal_publisher.publish(msg)

    def pcd_callback(self):
        if self.pcd_msg is None:
            points = self.gsplat.means.cpu().numpy()
            colors = (255 * torch.clip(self.gsplat.colors, 0., 1.).cpu().numpy()).astype(np.uint32)

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
            ]

            self.pcd_msg = make_point_cloud(points, colors, self.map_name, fields)

        self.pcd_publisher.publish(self.pcd_msg)

    def odometry_callback(self, msg):
        # Extract velocity and position in the x and y directions (assuming NED frame)
        if msg.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
            #self.current_velocity[0] = -msg.velocity[0]  # Velocity in x direction
            #self.current_velocity[1] = -msg.velocity[1]  # Velocity in y direction
            # self.current_position[0] = msg.position[0]  # Position in x direction
            # self.current_position[1] = msg.position[1]  # Position in y direction
            self.fmu_pos[0] = msg.position[0]
            self.fmu_pos[1] = msg.position[1]
            self.fmu_vel[0] = msg.velocity[0]
            self.fmu_vel[1] = msg.velocity[1]
            # print("we are getting odom")
            pass

    def key_listener(self):
        print("Press the space bar to start the mission.")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.start_mission:
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                if dr:
                    c = sys.stdin.read(1)
                    if c == ' ':
                        self.start_mission = True
                        print("Space bar pressed. Starting trajectory.")
                        break
        except Exception as e:
            print(e)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_control(self):
        # current_time = self.get_clock().now().to_msg()
        # current_time_f = current_time.sec + current_time.nanosec * 1e-9
        # print(current_time_f)
        dt = 1.0 / 30.0  # Assuming the timer runs at 30 Hz
        control_msg = Float32MultiArray()

        # pop from the first element of the trajectory 
        if (self.start_mission):

            if len(self.traj) > 0:
                # waypoint [pos, vel, accel, jerk]
                outgoing_waypoint = self.traj.pop(0)

                self.position_output = [outgoing_waypoint[0], outgoing_waypoint[1], outgoing_waypoint[2]]
                self.velocity_output = [outgoing_waypoint[3], outgoing_waypoint[4], outgoing_waypoint[5]]
                acceleration_output = [outgoing_waypoint[6], outgoing_waypoint[7], outgoing_waypoint[8]]        # We set this to 0 for now
                self.jerk = [outgoing_waypoint[9], outgoing_waypoint[10], outgoing_waypoint[11]]

            else:
                print("Trajectory complete.")

        control_msg.data = [
            self.acceleration_output[0], self.acceleration_output[1], self.acceleration_output[2],
            self.velocity_output[0], self.velocity_output[1], self.velocity_output[2],
            self.position_output[0], self.position_output[1], self.position_output[2], self.yaw
        ]

        self.control_publisher.publish(control_msg)
        self.publish_control_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9  
        # print("control message: ", control_msg.data)

    def plan_path(self, start, goal):
        start = torch.tensor(start).to(device).to(torch.float32)
        goal = torch.tensor(goal).to(device).to(torch.float32)
        output = self.planner.generate_path(start, goal)

        # OUTPUT IS A DICTIONARY
        # traj_data = {
        #     'path': path.tolist(),
        #     'polytopes': [torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes],
        #     'num_polytopes': len(polytopes),
        #     'traj': traj.tolist(),
        #     'times_astar': time_astar,
        #     'times_collision_set': times_collision_set,
        #     'times_polytope': times_polytope,
        #     'times_opt': times_opt,
        #     'feasible': feasible
        # }

        return output['traj'], output
    
    ### THIS CODE FUNCTION IS A MESS ###
    # def replan(self):
    #     print("-" * 20)
    #     print("Starting Replanning...")

    #     if self.current_position is not None and self.current_goal is not None:
    #         print("Current Position: ", self.current_position)
    #         print("Current Goal: ", self.current_goal)
    #         state_msg = String()
    #         state_msg.data = "replan"
    #         self.state_publisher.publish(state_msg)

    #         # publish goal
    #         msg = PoseStamped()
    #         print(self.current_goal)
    #         msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
    #             float(self.current_goal[0]),
    #             float(self.current_goal[1]),
    #             float(self.current_goal[2]),
    #         )
    #         msg.header.frame_id = self.map_name
    #         self.goal_publisher.publish(msg)

    #         # update the goal
    #         if (
    #             np.linalg.norm(self.current_position - self.current_goal)
    #             < self.goal_threshold
    #         ):
    #             if self.goal_idx < len(self.goal_locations):
    #                 print(f"****** Reached goal location {self.goal_idx}!")
    #                 print("Updating the goal...")
    #                 self.save_results()
    #                 self.current_goal = self.goal_locations[self.goal_idx]

    #                 # increment the goal index
    #                 self.goal_idx += 1

    #                 self.splatplan.update_goal(self.current_goal)
    #             else:
    #                 print("****** Reached all the goal locations!")

    #                 state_msg = String()
    #                 state_msg.data = "tracking"
    #                 self.state_publisher.publish(state_msg)

    #                 self.save_results()
    #                 return

    #         print("Computing Waypoints...")
    #         x0 = self.current_position

    #         # computation time
    #         start_time = time.perf_counter()

    #         try:
    #             waypoints, polytopes = self.splatplan.get_waypoint(x0)

    #             # total computation time (for this iteration)
    #             time_to_get_wps = time.perf_counter() - start_time
    #             print(f"Total waypoint gen time: {time_to_get_wps}")
    #             self.computation_time.append(time_to_get_wps)

    #             # waypoints to be cached
    #             self.cache_waypoint_queue.append(waypoints)

    #             # polytopes to be cached
    #             Ab = [
    #                 np.concatenate([A, b[:, None]], axis=-1)
    #                 for A, b in zip(polytopes["A"], polytopes["b"])
    #             ]
    #             self.cache_polytopes.append(Ab)

    #             # cache the success flag
    #             self.cache_success_flag.append(1 if waypoints is not None else 0)

    #             self.cache_gt_state.append(self.gt_state.copy())

    #             if waypoints is not None:
    #                 self.queue = waypoints.tolist()
    #                 print("Successfully computed waypoints, publishing...")

    #                 poses = []
    #                 yaw = 0.0

    #                 for idx, pt in enumerate(waypoints):
    #                     vels = pt[3:6]
    #                     msg = Pose()
    #                     # msg.header.frame_id = self.map_name
    #                     msg.position.x, msg.position.y, msg.position.z = pt[0], pt[1], pt[2]

    #                     if idx < len(waypoints) - 1:

    #                         diff = waypoints[idx + 1] - waypoints[idx]

    #                         prev_yaw = yaw
    #                         yaw = np.arctan2(diff[1], diff[0])

    #                         closest_k = np.round(-(yaw - prev_yaw) / (2*np.pi))
    #                         yaw = yaw + 2*np.pi*closest_k     

    #                     quat = Rotation.from_euler("z", yaw).as_quat()
    #                     (
    #                         msg.orientation.x,
    #                         msg.orientation.y,
    #                         msg.orientation.z,
    #                         msg.orientation.w,
    #                     ) = (quat[0], quat[1], quat[2], quat[3])

    #                     poses.append(msg)

    #                 msg = PoseArray()
    #                 msg.header.frame_id = self.map_name
    #                 msg.poses = poses

    #                 self.queue_viz_publisher.publish(msg)

    #                 # Creates JointTrajectory
    #                 joint_trajectories = []
    #                 yaw = 0.0
    #                 for idx, pt in enumerate(waypoints):
    #                     vels = pt[3:6]
    #                     msg = JointTrajectoryPoint()
    #                     # msg.header.frame_id = self.map_name
    #                     msg.velocities = [pt[3], pt[4], pt[5]]
    #                     msg.accelerations = [pt[6], pt[7], pt[8]]
    #                     msg.effort = [pt[9], pt[10], pt[11]]  # Not really effort

    #                     if idx < len(waypoints) - 1:

    #                         diff = waypoints[idx + 1] - waypoints[idx]

    #                         prev_yaw = yaw
    #                         yaw = np.arctan2(diff[1], diff[0])

    #                         closest_k = np.round(-(yaw - prev_yaw) / (2*np.pi))
    #                         yaw = yaw + 2*np.pi*closest_k                            

    #                     # yaw = np.arctan2(vels[1], vels[0])
    #                     # quat = Rotation.from_euler("z", yaw).as_quat()
    #                     # (
    #                     #     msg.orientation.x,
    #                     #     msg.orientation.y,
    #                     #     msg.orientation.z,
    #                     #     msg.orientation.w,
    #                     # ) = (quat[0], quat[1], quat[2], quat[3])

    #                     msg.positions = [pt[0], pt[1], pt[2], yaw]

    #                     joint_trajectories.append(msg)

    #                 msg = JointTrajectory()
    #                 msg.header.frame_id = self.map_name
    #                 msg.points = joint_trajectories

    #                 self.queue_publisher.publish(msg)
                
    #             state_msg = String()
    #             state_msg.data = "tracking"
    #             self.state_publisher.publish(state_msg)
                    
    #         except Exception as e:
    #             print(e)
                
    #             state_msg = String()
    #             state_msg.data = "tracking"
    #             self.state_publisher.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()
    
    rclpy.spin(control_node)
    
    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
