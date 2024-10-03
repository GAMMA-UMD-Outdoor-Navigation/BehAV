# !/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import math
import numpy as np
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError
# from matplotlib import pyplot as plt

# Message types
from std_msgs.msg import Float32, Float32MultiArray, UInt32
from geometry_msgs.msg import Twist, PointStamped,Point, PoseArray, PoseStamped, Quaternion, Pose
from nav_msgs.msg import OccupancyGrid, Odometry, GridCells
from sensor_msgs.msg import LaserScan, CompressedImage, NavSatFix
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPTokenizer
# import nevergrad as ng
import cma

import copy
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import utm

import nlopt
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import KDTree
from threading import Condition, Lock

# This import is added to allow setting autonomous mode wihtout CLI
# from ghost_manager_interfaces.srv import EnsureMode, SetParam


class GhostInit(Node):

    def __init__(self):

        super().__init__('Ghost_initializaton') 
    
        self.qos_profile_be = QoSProfile(
                                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                        history=QoSHistoryPolicy.KEEP_LAST,  
                                        depth=10  
                                        )

        self.qos_profile_kl = QoSProfile(
                                        reliability=QoSReliabilityPolicy.RELIABLE,
                                        history=QoSHistoryPolicy.KEEP_LAST,  
                                        depth=10  
                                        )
        
        # For setting to Autonomous Mode

        self.mode_client = self.create_client(EnsureMode, 'ensure_mode')
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('EnsureMode service not available, waiting again...')
        self.mode_req = EnsureMode.Request()
        self.param_client = self.create_client(SetParam, 'set_param')
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SetParam service not available, waiting again...')
        self.param_req = SetParam.Request()

        self.ensure_mode("control_mode", 170)
        self.ensure_mode("action", 2) # walk mode


    def set_param(self, param_name, value, planner=False):
        self.param_req.param.name = param_name
        self.param_req.param.val = value
        self.param_req.param.planner = planner
        future = self.param_client.call_async(self.param_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info('%s' % response.result_str)

    def ensure_mode(self, field_name, valdes):
        self.mode_req.field = field_name
        self.mode_req.valdes = valdes
        future = self.mode_client.call_async(self.mode_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info('%s' % response.result_str)

class ControlLawSettings:
    # (self, K1=1.2, K2=1, BETA=0.4, LAMBDA=2, V_MAX=0.8, V_MIN=0.0, R_THRESH=0.05):
    def __init__(self, K1=1, K2=3, BETA=1, LAMBDA=1, V_MAX=1.0, V_MIN=0.0, R_THRESH=0.05):
        self.m_K1 = K1
        self.m_K2 = K2
        self.m_BETA = BETA
        self.m_LAMBDA = LAMBDA
        self.m_V_MAX = V_MAX
        self.m_V_MIN = V_MIN
        self.m_R_THRESH = R_THRESH

class EgoPolar:
    def __init__(self, r=0.0, delta=0.0, theta=0.0):
        self.r = r
        self.delta = delta
        self.theta = theta

class ControlLaw:
    def __init__(self, settings=ControlLawSettings()):
        self.settings = settings

    @staticmethod
    def mod(x, y):
        m = x - y * math.floor(x / y)
        if y > 0:
            if m >= y:
                return 0
            if m < 0:
                if y + m == y:
                    return 0
                else:
                    return y + m
        else:
            if m <= y:
                return 0
            if m > 0:
                if y + m == y:
                    return 0
                else:
                    return y + m
        return m

    def update_k1_k2(self, k1, k2):
        self.settings.m_K1 = k1
        self.settings.m_K2 = k2

    def wrap_pos_neg_pi(self, angle):
        return self.mod(angle + math.pi, 2 * math.pi) - math.pi

    def get_ego_distance(self, current_pose, current_goal_pose):
        dx = current_goal_pose.position.x - current_pose.pose.pose.position.x
        dy = current_goal_pose.position.y - current_pose.pose.pose.position.y
        return math.sqrt(dx**2 + dy**2)

    def convert_to_egopolar(self, current_state, current_goal_pose):
        """
        Converts a goal position from Cartesian coordinates (in the odom frame) to egocentric polar coordinates (in the robot's frame).

        Inputs:
        current_state: The current position and orientation of the robot in the world frame (as a list [x, y, yaw]).
        current_goal_pose: The goal position and orientation in the world frame (as a Pose object).

        Outputs:
        coords: An instance of EgoPolar that contains the goal's position in the robot's frame.
        """
        coords = EgoPolar()
        
        # Compute the difference in x and y between the current state and the goal
        dx = float(current_goal_pose.position.x) - float(current_state[0])
        dy = float(current_goal_pose.position.y) - float(current_state[1])
        
        # Compute the heading to the goal
        obs_heading = math.atan2(dy, dx)
        current_yaw = float(current_state[2])
        
        # Convert quaternion orientation of the goal to yaw angle
        goal_yaw = euler_from_quaternion([
            float(current_goal_pose.orientation.x),
            float(current_goal_pose.orientation.y),
            float(current_goal_pose.orientation.z),
            float(current_goal_pose.orientation.w)
        ])[2]
        
        # Calculate the polar coordinates relative to the robot
        coords.r = math.sqrt(dx**2 + dy**2)
        coords.delta = self.wrap_pos_neg_pi(current_yaw - obs_heading)
        coords.theta = self.wrap_pos_neg_pi(goal_yaw - obs_heading)

        return coords
    
    def convert_from_egopolar(self, current_state, current_goal_coords):
        """
        Converts a goal position from egocentric polar coordinates (relative to the robot's frame) back to Cartesian coordinates (in the odom frame).

        Inputs:
        current_state: The current position and orientation of the robot in the world frame (as a list [x, y, yaw]).
        current_goal_coords: The goal position in egocentric polar coordinates (as an EgoPolar object).

        Outputs:
        current_goal_pose: The goal's position and orientation in Cartesian coordinates (as a Pose object).
        """
        
        current_yaw = float(current_state[2])

        current_goal_pose = Pose()
        
        # Calculate the Cartesian x, y position from polar coordinates
        current_goal_pose.position.x = float(current_state[0]) + float(current_goal_coords.r) * math.cos(current_yaw - float(current_goal_coords.delta))
        current_goal_pose.position.y = float(current_state[1]) + float(current_goal_coords.r) * math.sin(current_yaw - float(current_goal_coords.delta))
        current_goal_pose.position.z = 0.0  # Assuming the goal is in the 2D plane

        # Calculate the quaternion orientation from yaw angles
        quaternion = quaternion_from_euler(0, 0, current_yaw - float(current_goal_coords.delta) + float(current_goal_coords.theta))
        current_goal_pose.orientation.x = float(quaternion[0])
        current_goal_pose.orientation.y = float(quaternion[1])
        current_goal_pose.orientation.z = float(quaternion[2])
        current_goal_pose.orientation.w = float(quaternion[3])

        return current_goal_pose
    

    def get_kappa(self, current_ego_goal, k1, k2):
        kappa = (-1 / current_ego_goal.r) * (
            k2 * (current_ego_goal.delta - math.atan(-1 * k1 * current_ego_goal.theta)) +
            (1 + k1 / (1 + k1**2 * current_ego_goal.theta**2)) * math.sin(current_ego_goal.delta)
        )
        return kappa

    def get_linear_vel(self, kappa, current_ego_goal, vMax):
        lin_vel = min(self.settings.m_V_MAX / self.settings.m_R_THRESH * current_ego_goal.r,
                      self.settings.m_V_MAX / (1 + self.settings.m_BETA * abs(kappa)**self.settings.m_LAMBDA)) #original
        # lin_vel= self.settings.m_V_MAX / (1 + self.settings.m_BETA * abs(kappa)**self.settings.m_LAMBDA)

        if self.settings.m_V_MIN < lin_vel < 0.00:
            lin_vel = self.settings.m_V_MIN

        return lin_vel

    @staticmethod
    def calc_sigmoid(time_tau):
        sigma = 1.02040816 * (1 / (1 + math.exp(-9.2 * (time_tau - 0.5))) - 0.01)
        if sigma > 1:
            sigma = 1
        elif sigma < 0:
            sigma = 0
        return sigma
    
    def get_velocity_command(self, state, goal, vMax, k1=None, k2=None):
        if k1 is None:
            k1 = self.settings.m_K1
        if k2 is None:
            k2 = self.settings.m_K2
        if vMax is None:
            vMax = self.settings.m_V_MAX

        #get intermediate goal coord to polar coord w.r.t. robot
        goal_coords = self.convert_to_egopolar(state, goal)
        return self._get_velocity_command(goal_coords, k1, k2, vMax)


    def _get_velocity_command(self, goal_coords, k1, k2, vMax):
        cmd_vel = Twist()
        kappa = self.get_kappa(goal_coords, k1, k2)
        cmd_vel.linear.x = self.get_linear_vel(kappa, goal_coords, vMax)
        cmd_vel.angular.z = kappa * cmd_vel.linear.x

        R_SPEED_LIMIT = self.settings.m_V_MAX - 0.1  # Assuming R_SPEED_LIMIT is equal to V_MAX

        if abs(cmd_vel.angular.z) > R_SPEED_LIMIT:
            cmd_vel.angular.z = math.copysign(R_SPEED_LIMIT, cmd_vel.angular.z)
            cmd_vel.linear.x = cmd_vel.angular.z / kappa

        return cmd_vel


class BehAV_Planner(Node):

    def __init__(self):

        super().__init__('BehAV_planner') 

        self.qos_profile  = QoSProfile(
                                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                        history=QoSHistoryPolicy.KEEP_LAST,  
                                        depth=10  
                                        )

        self.qos_profile_intensity  = QoSProfile(
                                                reliability=QoSReliabilityPolicy.RELIABLE,
                                                history=QoSHistoryPolicy.KEEP_LAST,  
                                                depth=10  
                                                )
        
        #Run this service call inside ghost to set the robot to autonomous
        # ros2 service call /ensure_mode ghost_manager_interfaces/EnsureMode "{field: control_mode, valdes: 170}"

        # self.ghost_init = GhostInit()
        # self.config = Config()

        # k1 = 0 reduces the controller to pure waypointfollowing, while k1 >> 0 offers extreme scenario of pose-following where theta is reduced much faster than r
        # K1=1.2, K2=1, BETA=0.4, LAMBDA=2, V_MAX=0.8, V_MIN=0.0, R_THRESH=0.05

        self.settings = ControlLawSettings(K1=1.2, K2=1, BETA=0.4, LAMBDA=2, R_THRESH=0.05, V_MAX=0.8, V_MIN=0.0)
        self.control_law = ControlLaw(self.settings)

        self.odom_condition = Condition()
        self.odom_msg = None

        self.sub_odom = self.create_subscription(Odometry, '/odom_lidar', self.assignOdomCoords,self.qos_profile)
        # self.scan_subscriber = self.create_subscription(LaserScan,'/scan', self.scan_callback, self.qos_profile)

        # self.sub_odom = self.create_subscription(Odometry, '/odom', self.assignOdomCoords,self.qos_profile)
        # self.sub_cost_map = self.create_subscription(GridCells, '/costmap_translator/obstacles', self.config.occupancy_map_callback,self.qos_profile)

        self.subscription = self.create_subscription(Image,'/camera/color/image_raw', self.image_callback, 10)
        # Publisher for combined overlaid image
        self.behav_costmap_publisher = self.create_publisher(Image, '/behav_costmap', 10)
        self.traj_image_pub = self.create_publisher(Image, '/traj_marked_image', 10)

        choice = input("Publish to Robot Motors ? 1 or 0: ")
        
        if(int(choice) == 1):
            self.pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
            # self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            print("Publishing to cmd_vel")
        else:
            self.pub = self.create_publisher(Twist, "/dont_publish", 1)
            print("Not publishing!")

        self.max_speed = 0.8  # [m/s]

        #Robot state initialization
        self.x = None
        self.y = None
        self.goalX = None #specifi value to avoid robot moving until the goal is publishedh
        self.goalY = None
        self.goalPose =0
        self.th = None

        #Initial robot state and actions
        self.speed = Twist()
        self.goal_reach_thrshold = 0.7

        self.init_x = None
        self.init_y = None
        self.received_init_odom = False
        self.received_odom_once = False
        self.received_final_goal_odom = False

        self.sensing_range = 3.5
        self.obstacles_odom = None
        self.safe_dist_threshold = 0.5

        self.to_global_goal_from_init = 0
        self.current_to_goal_dist = self.goal_reach_thrshold + 1 #avoid initial goal reach condition satisfying
        self.current_pose = None

        self.final_goal_pose = Pose()

        print("torch.cuda.is_available()",torch.cuda.is_available())
        # Taking three float inputs from the user
        self.goal_radius = float(input("Enter the goal distance r (meters) : "))
        self.goal_theta = float(input("Enter the goal heading angle theta (degrees, left +ve) : "))
        self.goal_delta = float(input("Enter the goal pose angle (degrees) : "))

        self.velocityGain = 1.0

        #optimizer params
        # Trajectory model parameters
        self.V_MAX = 1.0
        self.V_MIN = 0.0
        self.trajectory_count = 0
        self.TIME_HORIZON = 4 #2.5
        self.DELTA_SIM_TIME = 0.5 #0.2
        self.SAFETY_ZONE = 0.225
        self.WAYPOINT_THRESH = 1.75

        #weighting factors for the objective function
        self.goal_factor = 1
        self.goal_angle_factor = 3 

        # Cost function parameters
        self.C1 = 0.05
        self.C2 = 2.5
        self.C3 = 0.05
        self.C4 = 0.05
        self.PHI_COL = 1.0
        self.SIGMA = 0.2

        self.pose_mutex = Lock()
        self.cost_map_mutex = Lock()

        # Behav image costmap params
        # Initialize a CvBridge to convert ROS images to OpenCV images
        self.received_img_once = False
        self.bridge = CvBridge()
        self.img_h, self.img_w = None, None

        # Set device for model computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        # Load the CLIPSeg model and processor
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.get_logger().info('CLIPSeg node has been started.')

        # self.prompts = ["Stop_Gesture", "Pavement", "Grass"]  # Prompts for segmentation
        self.prompts = ["vegetation", "Pavement","grass", "Stop gesture"]  # Prompts for segmentation

        self.cost_values = [0.05,0.95,0.48,0] #high means preferred regions (e.g., pavement), and low means avoiding regions
        self.behav_costmap = None
        # Flag to control output publishing
        self.publish_outputs = False

        #traj projection params
        self.Projection_Matrix = [[910.7625732421875, 0.0, 643.8300781250, 0.0],[0.0,910.8343505859375,373.2903137207031,0.0],[0.0, 0.0, 1.0, 0.0]] # realsense lidar camera L515

        # self.Projection_Matrix = [[607.175048828125, 0.0, 322.55340576171875, 0.0], [0.0, 607.222900390625, 248.86021423339844, 0.0], [0.0, 0.0, 1.0, 0.0]] # realsense lidar camera L515

        self.camera_height = 0.59 #1.01 #height of the camera w.r.t. the robot's base/ground level
        self.camera_tilt_angle = 0 # in degrees, downward is negative
        self.camera_offset_x = 0 #0.46
        self.camera_offset_y = 0 #0.065 #camera y axis offset in meters

    def wait_for_odom(self):
        # Wait for the odom message
        while not self.received_odom_once and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_img(self):
        # Wait for the odom message
        while not self.received_img_once and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
    def run(self):
        self.wait_for_odom()
        self.wait_for_img()
        self.get_logger().info("Odom message received, starting main loop.")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # Process incoming messages
            self.main_loop()

    def main_loop(self):

        loop_start_time = time.time()
        
        if self.received_odom_once and not self.received_final_goal_odom:
            self.goal_to_odom_pose()
            self.received_final_goal_odom = True

        if self.received_odom_once and self.received_final_goal_odom and self.received_init_odom and self.received_img_once:

            if self.current_to_goal_dist < self.goal_reach_thrshold:
                self.speed.linear.x = 0.0
                self.speed.angular.z = 0.0
                print("--- Goal Reached !! ---")

            else:

                new_coords, new_vMax = self.find_intermediate_goal_params()

                cmd_vel = self.control_law._get_velocity_command(new_coords, k1 = self.settings.m_K1, k2 = self.settings.m_K2, vMax= new_vMax)
                self.speed.linear.x = cmd_vel.linear.x
                self.speed.angular.z = cmd_vel.angular.z

            # print("Published velocities (v,w) : ",self.speed.linear.x,self.speed.angular.z)
            self.pub.publish(self.speed)
        else:
            print(" -- Waiting for odom to intialize -- ")

        loop_end_time = time.time()
        tot_inference_time = loop_end_time - loop_start_time
        print("--- Total inference rate per cycle ---", (1/tot_inference_time))



    def find_intermediate_goal_params(self):
        self.trajectory_count = 0

        # opt = nlopt.opt(nlopt.GN_DIRECT_NOSCAL, 4)
        # opt = nlopt.opt(nlopt.GN_CRS2_LM, 4) #GN_ESCH
        opt = nlopt.opt(nlopt.GN_ESCH, 4)
        opt.set_min_objective(self.score_trajectory)
        opt.set_xtol_rel(0.0001)
        lb = [0.1, -0.8, -0.8, self.V_MIN]
        ub = [5.0, 0.8, 0.8, self.V_MAX]
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_maxeval(300)

        # x = [0.0, 0.0, 0.0, 0.0]
        x = [2.5, 0.0, 0.0, (self.V_MAX + self.V_MIN) / 2.0]
        minf = opt.optimize(x)
        # print(f"Global Optimization - Trajectories evaluated: {self.trajectory_count}")
        # print(f"Global Optimization - Result: {x}, Min Cost: {minf}")

        opt2 = nlopt.opt(nlopt.LN_BOBYQA, 4)
        opt2.set_min_objective(self.score_trajectory)
        opt2.set_xtol_rel(0.0008)
        opt2.set_lower_bounds(lb)
        opt2.set_upper_bounds(ub)
        opt2.set_maxeval(75)

        minf2 = opt2.optimize(minf)
        # print(f"Local Optimization - Trajectories evaluated: {self.trajectory_count}")
        # print(f" Min Cost: {minf2}")

        # return {'r': x[0], 'delta': x[1], 'theta': x[2], 'vMax': x[3], 'k1': self.K_1, 'k2': self.K_2}
        # {'r': minf2[0], 'delta': minf2[1], 'theta': minf2[2], 'vMax': minf2[3], 'k1': self.K_1, 'k2': self.K_2}
        coords = EgoPolar()
        coords.r = minf2[0]
        coords.delta = minf2[1]
        coords.theta = minf2[2]

        vMax = minf2[3]

        return coords, vMax
    
    def score_trajectory(self, x, grad=None):
        time_horizon = self.TIME_HORIZON
        # Optional: If gradient is provided, set it to zero (not used in this case)
        if grad is not None:
            grad[:] = 0.0  # This sets all gradient elements to 0

        tot_cost = self.sim_trajectory(x[0], x[1], x[2], x[3], time_horizon)

        return tot_cost


    def sim_trajectory(self, r, delta, theta, vMax, time_horizon):


        with self.pose_mutex:
            # Initial state: [x, y, theta]
            state = np.array([self.x, self.y, self.th])
            num_steps = int(time_horizon / self.DELTA_SIM_TIME)
            trajectory = np.zeros((num_steps + 1, 3))  # Pre-allocate trajectory array
            trajectory[0, :] = state  # Set the initial state


        expected_progress = 0.0
        expected_action = 0.0
        expected_collision = 0.0
        expected_behav = 0.0
        survivability = 1.0

        # Convert goal to odom coordinates (remains static)
        sim_goal = EgoPolar(r=r, delta=delta, theta=theta)
        current_goal = self.control_law.convert_from_egopolar(state, sim_goal)

        # Initialize arrays for control inputs
        control_inputs = np.zeros((num_steps, 2))  # Each row will be [v, omega]

        # Precompute control inputs for all time steps
        for i in range(num_steps):
            # Compute velocity command based on the current state
            sim_cmd_vel = self.control_law.get_velocity_command(state, current_goal, vMax)
            control_inputs[i, :] = [sim_cmd_vel.linear.x, sim_cmd_vel.angular.z]

            # Update state using the motion model (dynamic update for control inputs)
            state = motion(self, state, control_inputs[i, :], self.DELTA_SIM_TIME)
            trajectory[i + 1, :] = state  # Store updated state in trajectory

        # Vectorized calculation of progress cost
        total_distance, total_heading_error = self.calculate_total_distance_and_heading_error(trajectory, self.final_goal_pose)
        expected_progress = 2 * total_distance + 1 * total_heading_error

        # Convert the odom trajectory to the robot frame
        robot_frame_trajectory = self.odom_traj_to_robot(trajectory, self.x, self.y, self.th)
        
        # Get behavioral costs (assuming this function is optimized)
        traj_marked_img, max_behav_cost = self.get_traj_behav_cost(robot_frame_trajectory)

        expected_behav += (max_behav_cost / 255) #normalizing the cost to 0-1 range

        # Sum the costs
        total_cost = expected_collision + expected_progress + expected_action + expected_behav

        return total_cost
       
    def get_yaw_from_quaternion(self, quaternion):
        return euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]

    def quaternion_from_yaw(self, yaw):
        return quaternion_from_euler(0, 0, yaw)
    
    def calculate_distance_and_angle(self, pose1, pose2, normalize):
        """
        Calculate the Euclidean distance and the delta angle between two poses in the odometry frame.

        Inputs:
        - pose1: The current pose of the robot (geometry_msgs/Odom) with attributes position.x, position.y, and orientation (quaternion).
        - pose2: The goal pose (geometry_msgs/Pose) with attributes position.x, position.y, and orientation (quaternion).

        Returns:
        - distance: The Euclidean distance between the two poses.
        - delta_angle: The angle between the robot's heading (from pose1) and the direction to the goal.
        """
        # Calculate the Euclidean distance
        dx = pose2.position.x - pose1.pose.pose.position.x
        dy = pose2.position.y - pose1.pose.pose.position.y

        if normalize:
            distance = math.sqrt(dx**2 + dy**2) / self.to_global_goal_from_init
        else:
            distance = math.sqrt(dx**2 + dy**2)


        # Calculate the robot's current yaw (heading)
        robot_yaw = euler_from_quaternion([
            pose1.pose.pose.orientation.x,
            pose1.pose.pose.orientation.y,
            pose1.pose.pose.orientation.z,
            pose1.pose.pose.orientation.w
        ])[2]

        # Calculate the angle to the goal
        goal_angle = math.atan2(dy, dx)

        # Calculate the delta angle (difference between robot's heading and the direction to the goal)
        delta_angle = self.control_law.wrap_pos_neg_pi(robot_yaw - goal_angle)
        delta_angle_abs = math.fabs(delta_angle) #/math.pi #normalized between 0-1

        return distance, delta_angle_abs


    def calculate_total_distance_and_heading_error(self, trajectory, goal_pose):
        """
        Vectorized calculation of total distances and heading errors.
        """
        trajectory = np.array(trajectory)
        
        # Extract x, y, and yaw coordinates from the trajectory
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        yaw_angles = trajectory[:, 2]
        
        # Extract goal position
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y

        # Vectorized distance calculation (Euclidean distance to the goal for all points)
        distances = np.sqrt((x_coords - goal_x) ** 2 + (y_coords - goal_y) ** 2)

        # Vectorized heading error calculation
        goal_directions = np.arctan2(goal_y - y_coords, goal_x - x_coords)
        heading_errors = np.abs(np.arctan2(np.sin(goal_directions - yaw_angles), np.cos(goal_directions - yaw_angles)))

        # Sum up all distances and heading errors
        total_distance = np.sum(distances)
        total_heading_error = np.sum(heading_errors)

        return total_distance, total_heading_error


    # Callback for Odometry
    def assignOdomCoords(self, msg):

        # print("inside_odom")
        self.current_pose = msg

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w]) #uses the library from ros2, leads to errors
        (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w]) #uses the code in config class

        self.th = theta

        if self.received_final_goal_odom:
            self.current_to_goal_dist = np.sqrt((self.goalX - self.x) ** 2 + (self.goalY - self.y) ** 2)

        if not self.received_init_odom and self.received_final_goal_odom:
            self.init_x = msg.pose.pose.position.x
            self.init_y = msg.pose.pose.position.y

            self.to_global_goal_from_init = np.sqrt((self.goalX - self.init_x) ** 2 + (self.goalY - self.init_y) ** 2)
            self.received_init_odom = True
            print("Robot's init goal dist and coords", self.to_global_goal_from_init,(self.init_x,self.init_y))

        self.received_odom_once = True

    def scan_callback(self, scan):
        print('scan callback')

        if self.received_final_goal_odom:
            # Filter out 'inf' values and distances beyond the sensing range from the ranges
            valid_ranges = [(distance, scan.angle_min + i * scan.angle_increment)
                            for i, distance in enumerate(scan.ranges)
                            if not math.isinf(distance) and distance <= self.sensing_range]
            
            # Convert valid ranges to Cartesian coordinates in the robot's frame
            obstacles_cartesian = [self.polar_to_cartesian(distance, angle) for distance, angle in valid_ranges]

            # Transform obstacle coordinates to the odom frame
            self.obstacles_odom = [self.transform_to_odom(x, y) for x, y in obstacles_cartesian]
        else:
            pass

    def polar_to_cartesian(self, distance, angle):
        # Convert polar coordinates to Cartesian coordinates in the robot's frame
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        return x, y
    
    def transform_to_odom(self, obstacle_x, obstacle_y):
        # Transform the obstacle position from the robot's frame to the odom frame
        robot_x, robot_y, robot_th = self.x, self.y, self.th
        odom_x = robot_x + (obstacle_x * math.cos(robot_th) - obstacle_y * math.sin(robot_th))
        odom_y = robot_y + (obstacle_x * math.sin(robot_th) + obstacle_y * math.cos(robot_th))
        return odom_x, odom_y


    def get_distances_to_obstacles(self, trajectory, obstacles_odom):
        distances_to_obstacles = []
        
        # Handle case where no obstacles are detected
        if not obstacles_odom:
            self.get_logger().info("No obstacles detected within the sensing range.")
            return [1.0] * len(trajectory)  # Assume all distances are safe, normalized to 1.0

        for state in trajectory:
            x, y, _ = state  # Unpack the state
            min_distance = float('inf')
            for obs_x, obs_y in obstacles_odom:
                distance = math.sqrt((obs_x - x) ** 2 + (obs_y - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
            # Normalize by the sensing range
            normalized_distance = min(min_distance / self.sensing_range, 1.0)  # Cap the value at 1.0
            distances_to_obstacles.append(normalized_distance)
        
        return distances_to_obstacles
    
    def global_to_ego(self, current_state, traj_state_x, traj_state_y):
        # Transform global coordinates to ego frame (polar coordinates)
        dx = traj_state_x - current_state[0]
        dy = traj_state_y - current_state[1]
        distance = math.sqrt(dx**2 + dy**2)
        angle = current_state[2] - math.atan2(dy, dx)  # Angle relative to robot's heading
        return angle, distance
    
    def odom_traj_to_robot(self, trajectory, robot_x, robot_y, robot_theta):
        """
        Convert an entire trajectory in the odom frame to the robot frame using vectorized operations.
        `trajectory`: List of states [[x_odom, y_odom, theta_odom], ...]
        """
        # Extract x, y, and theta components from the trajectory
        trajectory = np.array(trajectory)
        x_odom = trajectory[:, 0]
        y_odom = trajectory[:, 1]
        theta_odom = trajectory[:, 2]

        # Vectorized transformation to the robot frame
        x_rob = (x_odom - robot_x) * np.cos(robot_theta) + (y_odom - robot_y) * np.sin(robot_theta)
        y_rob = -(x_odom - robot_x) * np.sin(robot_theta) + (y_odom - robot_y) * np.cos(robot_theta)

        # Combine transformed x and y with theta
        robot_frame_trajectory = np.column_stack((x_rob, y_rob, theta_odom))

        return robot_frame_trajectory


    def image_callback(self, msg):
        try:
            # Convert the ROS image message to OpenCV format and extract dimensions
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.img_h, self.img_w, _ = cv_image.shape
            pil_image = PILImage.fromarray(cv_image)

            # Process the text prompts and image input in batches
            inputs = self.tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            image_inputs = self.processor(images=[pil_image] * len(self.prompts), return_tensors="pt")
            inputs.update(image_inputs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform model inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
            inference_time = time.time() - start_time

            # Get prediction logits and apply sigmoid
            preds = torch.sigmoid(outputs.logits)

            # Initialize cost map (set to 128 for non-segmented areas)
            combined_cost_map = np.full((self.img_h, self.img_w), 128, dtype=np.float32)

            # Batch process segmentation masks and update the cost map
            preds_resized = F.interpolate(preds.unsqueeze(1), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()

            for i, pred_resized in enumerate(preds_resized):
                mask = pred_resized > 0.1
                combined_cost_map[mask] = pred_resized[mask] * 255 * self.cost_values[i]  # Update only segmented regions

            # Clip and convert cost map to 8-bit for visualization
            combined_cost_map = np.clip(combined_cost_map, 0, 255).astype(np.uint8)
            self.behav_costmap = combined_cost_map

            if self.publish_outputs:
                # Apply color mapping for visualization and overlay on the original image
                combined_cost_map_colored = cv2.applyColorMap(combined_cost_map, cv2.COLORMAP_JET)
                overlaid_image = cv2.addWeighted(cv_image, 0.4, combined_cost_map_colored, 0.6, 0)
                ros_overlaid_image = self.bridge.cv2_to_imgmsg(overlaid_image, encoding='rgb8')
                self.behav_costmap_publisher.publish(ros_overlaid_image)

            # Log the inference time
            end_time2 = time.time()
            self.get_logger().info(f"CLIPSeg Model Inference Rate: {1/(end_time2 - start_time):.4f} seconds")

            self.received_img_once = True

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


    def get_traj_behav_cost(self, robot_frame_trajectory):
        """
        Project robot frame trajectory points onto the camera image plane, get pixel coordinates,
        and return the maximum cost from the costmap at the trajectory pixel locations.
        """

        # Create a copy of the behavior cost map for visualization purposes
        marked_img = self.behav_costmap.copy()

        # Convert robot_frame_trajectory to NumPy array for efficient vectorized operations
        robot_frame_trajectory = np.array(robot_frame_trajectory)

        # Extract x_rob and y_rob from the trajectory
        x_rob = robot_frame_trajectory[:, 0]
        y_rob = robot_frame_trajectory[:, 1]

        # Vectorized computation for camera frame coordinates
        traj_coords_xyz = np.column_stack((
            -y_rob + self.camera_offset_y,   # Y-axis adjustment
            np.full(x_rob.shape, self.camera_height),  # Constant camera height
            x_rob - self.camera_offset_x  # X-axis adjustment
        ))

        if traj_coords_xyz.shape[0] == 0:
            return marked_img, 0.0

        # Apply rotation for tilt
        alpha = np.deg2rad(self.camera_tilt_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(alpha), -np.sin(alpha)],
                                    [0, np.sin(alpha), np.cos(alpha)]])
        points_rotated = np.dot(traj_coords_xyz, rotation_matrix.T)

        # Homogeneous coordinates for projection
        points_homogeneous = np.hstack((points_rotated, np.ones((points_rotated.shape[0], 1))))

        # Project onto image plane using the projection matrix
        uvw = np.dot(self.Projection_Matrix, points_homogeneous.T)
        u_vec, v_vec, w_vec = uvw[0], uvw[1], uvw[2]

        # Compute pixel coordinates
        x_vec = (u_vec / w_vec).astype(int)
        y_vec = (v_vec / w_vec).astype(int)

        # Filter out points that are outside the image bounds
        valid_indices = (y_vec >= 0) & (y_vec < self.img_h) & (x_vec >= 0) & (x_vec < self.img_w)
        valid_x = x_vec[valid_indices]
        valid_y = y_vec[valid_indices]

        if len(valid_x) == 0:
            return marked_img, 0.0

        # Extract cost values from the costmap at the valid pixel coordinates
        costs = self.behav_costmap[valid_y, valid_x]
        max_cost = np.max(costs) if costs.size > 0 else 0.0

        if self.publish_outputs:
            # Visualize the trajectory points on the image
            points = np.vstack((valid_x, valid_y)).T
            if len(points) > 1:
                cv2.polylines(marked_img, [points], isClosed=False, color=255, thickness=8)

            # Publish the marked image
            marked_image_msg = self.bridge.cv2_to_imgmsg(marked_img, encoding="mono8")
            self.traj_image_pub.publish(marked_image_msg)

        return marked_img, max_cost



    def occupancy_map_callback(self, msg):
        self.cost_map = msg
        if len(self.cost_map.cells) > 0:
            points = [(cell.x, cell.y) for cell in self.cost_map.cells]
            self.obs_tree = KDTree(points)
            # self.b_has_cost_map = True

    def get_obstacle_distance(self):
        if not self.b_has_cost_map or not self.b_has_odom:
            return 0
        _, min_dist = self.find_nearest_neighbor((self.current_pose.position.x, self.current_pose.position.y))
        return min_dist

    def find_nearest_neighbor(self, point):
        dist, idx = self.obs_tree.query(point)
        return self.cost_map.cells[idx], dist
    
    def convert_to_pose_stamped(self, new_coords):
        # Convert the goal coordinates to a PoseStamped message
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = new_coords['r'] * math.cos(new_coords['theta'])
        pose.pose.position.y = new_coords['r'] * math.sin(new_coords['theta'])
        return pose
    

    def goal_to_odom_pose(self):
        """
        Convert the goal pose from the robot frame to the odom frame and return it as an Odometry message.
        """
        # Goal position w.r.t. robot frame (polar coordinates to Cartesian)
        goalX_rob = self.goal_radius * math.cos(math.radians(self.goal_theta))
        goalY_rob = self.goal_radius * math.sin(math.radians(self.goal_theta))

        # Convert goal position from robot frame to odom frame
        self.goalX = self.x + goalX_rob * math.cos(self.th) - goalY_rob * math.sin(self.th)
        self.goalY = self.y + goalX_rob * math.sin(self.th) + goalY_rob * math.cos(self.th)

        # Convert goal orientation from robot frame to odom frame
        goal_yaw_rob = math.radians(self.goal_delta)
        goal_yaw_odom = self.th + goal_yaw_rob

        # Create Pose message
        pose = Pose()
        pose.position.x = self.goalX
        pose.position.y = self.goalY
        pose.position.z = 0.0  # Assuming the goal is on a 2D plane

        # Convert the goal orientation to quaternion format
        quaternion = quaternion_from_euler(0, 0, goal_yaw_odom)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        self.final_goal_pose = pose

        print("Goal x,y w.r.t. robot and odom :", (goalX_rob,goalY_rob),(self.goalX,self.goalY))


def motion(planner, state, u, dt):
    """
    Vectorized motion model to update the robot's state.
    Args:
        state: Current state [x, y, theta].
        u: Control input [v, omega] (linear and angular velocity).
        dt: Time step duration.
    Returns:
        Updated state [x, y, theta].
    """
    x, y, theta = state
    v, omega = u

    # Update theta (orientation)
    theta_new = theta + omega * dt

    # Update x and y positions based on the new theta and velocities
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt

    return np.array([x_new, y_new, theta_new])


def euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).

    The quaternion is expected to be a tuple (x, y, z, w).

    Returns:
    A tuple of three elements representing the Euler angles (roll, pitch, yaw) in radians.
    """
    x, y, z, w = quaternion

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw  # in radians

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) into a quaternion.
    
    roll is rotation around x-axis in radians (counterclockwise)
    pitch is rotation around y-axis in radians (counterclockwise)
    yaw is rotation around z-axis in radians (counterclockwise)
    
    Returns:
    A tuple of four elements representing the quaternion (x, y, z, w).
    """
    
    # Compute the quaternion components
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)




if __name__ == '__main__':
    
    rclpy.init()

    node = BehAV_Planner()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()