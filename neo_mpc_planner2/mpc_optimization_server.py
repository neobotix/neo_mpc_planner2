#!/usr/bin/env python3

# MIT License

# Copyright (c) 2022 neobotix gmbh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rclpy
from rclpy.node import Node
from neo_srvs2.srv import Optimizer
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose, Polygon
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
from scipy.optimize import minimize
import math
from functools import partial
import time
from neo_nav2_py_costmap2D.line_iterator import LineIterator
from neo_nav2_py_costmap2D.costmap import Costmap2d
from geometry_msgs.msg import PolygonStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from geometry_msgs.msg import Point32
import copy
from angles import shortest_angular_distance

class MpcOptimizationServer(Node):
	def __init__(self):
		super().__init__('mpc_optimization_server')

		# [Keep all existing parameter declarations]
		self.declare_parameter('acc_x_limit', value = 0.5)
		self.declare_parameter('acc_y_limit', value = 0.5)
		self.declare_parameter('acc_theta_limit', value = 0.5)

		self.declare_parameter('min_vel_x', value = -0.5)
		self.declare_parameter('min_vel_y', value = -0.5)
		self.declare_parameter('min_vel_trans', value = 0.5)
		self.declare_parameter('min_vel_theta', value = -0.5)

		self.declare_parameter('max_vel_x', value = 0.5)
		self.declare_parameter('max_vel_y', value = 0.5)
		self.declare_parameter('max_vel_trans', value = 0.5)
		self.declare_parameter('max_vel_theta', value = 0.5)

		self.declare_parameter('w_trans', value = 0.5)
		self.declare_parameter('w_orient', value = 0.5)
		self.declare_parameter('w_control', value = 0.5)
		self.declare_parameter('w_terminal', value = 0.5)
		self.declare_parameter('w_costmap', value = 0.5)
		self.declare_parameter('w_footprint', value = 2000)

		self.declare_parameter('waiting_time', value = 3.0)
		self.declare_parameter('low_pass_gain', value = 0.5)
		self.declare_parameter('opt_tolerance', value = 1e-5)
		self.declare_parameter('prediction_horizon', value = 0.5)
		self.declare_parameter('control_steps', value = 3)
		self.declare_parameter('sharp_turn_threshold', value = 0.52)
		self.declare_parameter('tight_lookahead_dist_threshold', value = 0.5)
		self.declare_parameter('control_time_scale', value = 0.4)  # Scale factor for collision check lookahead
		self.declare_parameter('costmap_max_age', value = 0.3)
		self.declare_parameter('use_footprint_constraints', value = True)
		self.declare_parameter('footprint_circle_offset_x', value = 0.25)
		self.declare_parameter('footprint_circle_offset_y', value = 0.25)
		self.declare_parameter('footprint_circle_radius', value = 0.36)
		self.declare_parameter('footprint_safety_margin', value = 0.05)
		self.declare_parameter('footprint_constraint_substeps', value = 5)
		self.declare_parameter('footprint_constraint_tolerance', value = 5e-3)
		self.declare_parameter('solver_finite_difference_step', value = 1e-3)

		# Get Parameters
		self.acc_x_limit = self.get_parameter('acc_x_limit').value
		self.acc_y_limit = self.get_parameter('acc_y_limit').value
		self.acc_theta_limit = self.get_parameter('acc_theta_limit').value

		self.min_vel_x = self.get_parameter('min_vel_x').value
		self.min_vel_y = self.get_parameter('min_vel_y').value
		self.min_vel_trans = self.get_parameter('min_vel_trans').value
		self.min_vel_theta = self.get_parameter('min_vel_theta').value

		self.max_vel_x = self.get_parameter('max_vel_x').value
		self.max_vel_y = self.get_parameter('max_vel_y').value
		self.max_vel_trans = self.get_parameter('max_vel_trans').value
		self.max_vel_theta = self.get_parameter('max_vel_theta').value

		self.w_trans = self.get_parameter('w_trans').value
		self.w_orient = self.get_parameter('w_orient').value
		self.w_control= self.get_parameter('w_control').value
		self.w_terminal = self.get_parameter('w_terminal').value
		self.w_costmap_scale = self.get_parameter('w_costmap').value
		self.w_footprint_scale = self.get_parameter('w_footprint').value

		self.low_pass_gain = self.get_parameter('low_pass_gain').value
		self.opt_tolerance = self.get_parameter('opt_tolerance').value
		self.prediction_horizon= self.get_parameter('prediction_horizon').value
		self.no_ctrl_steps = self.get_parameter('control_steps').value
		self.waiting_time = self.get_parameter('waiting_time').value
		self.sharp_turn_threshold = self.get_parameter('sharp_turn_threshold').value
		self.tight_lookahead_dist_threshold = self.get_parameter('tight_lookahead_dist_threshold').value
		self.control_time_scale = self.get_parameter('control_time_scale').value
		self.costmap_max_age = self.get_parameter('costmap_max_age').value
		self.use_footprint_constraints = self.get_parameter('use_footprint_constraints').value
		self.footprint_circle_radius = self.get_parameter('footprint_circle_radius').value
		self.footprint_safety_margin = self.get_parameter('footprint_safety_margin').value
		self.footprint_constraint_substeps = self.get_parameter('footprint_constraint_substeps').value
		self.footprint_constraint_tolerance = self.get_parameter('footprint_constraint_tolerance').value
		self.solver_finite_difference_step = self.get_parameter('solver_finite_difference_step').value

		circle_offset_x = self.get_parameter('footprint_circle_offset_x').value
		circle_offset_y = self.get_parameter('footprint_circle_offset_y').value
		self.footprint_circle_centers = (
			(circle_offset_x, circle_offset_y),
			(circle_offset_x, -circle_offset_y),
			(-circle_offset_x, circle_offset_y),
			(-circle_offset_x, -circle_offset_y))

		if (self.costmap_max_age <= 0.0 or self.footprint_circle_radius <= 0.0 or
				self.footprint_safety_margin < 0.0 or
				self.footprint_constraint_substeps < 1 or
				self.footprint_constraint_tolerance < 0.0 or
				self.solver_finite_difference_step <= 0.0):
			raise ValueError('Invalid costmap or footprint constraint parameters')

		self.srv = self.create_service(Optimizer, 'optimizer', self.optimizer)
		self.add_on_set_parameters_callback(self.cb_params)
		self.PubRaysPath = self.create_publisher(Path, 'local_plan', 10)
		self.Pubfootprint = self.create_publisher(PolygonStamped, 'predicted_footprint', 10)
		
		# NEW: Publisher for footprint array visualization
		self.PubFootprintArray = self.create_publisher(Path, 'predicted_footprints_debug', 10)
		
		self.current_pose = Pose()
		self.carrot_pose = PoseStamped()
		self.goal_pose = PoseStamped()
		self.current_velocity = TwistStamped()
		self.local_plan = Path()

		self.cost_total = 0.0
		self.costmap_cost = 0.0
		self.turn_yaw_ = 0.0
		self.turn_yaw_tight_ = 0.0
		self.effective_turn_angle_ = 0.0
		self.carrot_pose_tight = PoseStamped()
		self.carrot_pose_terminal = PoseStamped()
		self.last_control = [0,0,0]
		self.costmap_ros = Costmap2d(self)

		self.update_x = 0.0
		self.update_y = 0.0
		self.size_x_ = 0
		self.turn_yaw_ = 0.0

		# NEW: Storage for debug footprints
		self.debug_footprints = []
		self.debug_costs = []
		self.debug_poses = []

		self.bnds  = list()
		self.cons = []
		b_x_vel = (self.min_vel_x, self.max_vel_x)
		b_y_vel = (self.min_vel_y, self.max_vel_y)
		b_rot = (self.min_vel_theta, self.max_vel_theta)
		for i in range(self.no_ctrl_steps):
			self.bnds.append(b_x_vel)
			self.bnds.append(b_y_vel)
			self.bnds.append(b_rot)
			
			# Velocity magnitude constraint
			self.cons.append({'type': 'ineq', 'fun': partial(self.f_constraint, index = i)})
			
			# Acceleration constraints for smooth motion
			# self.cons.append({'type': 'ineq', 'fun': partial(self.acc_x_constraint, index = i)})
			# self.cons.append({'type': 'ineq', 'fun': partial(self.acc_y_constraint, index = i)})
			# self.cons.append({'type': 'ineq', 'fun': partial(self.acc_theta_constraint, index = i)})

		if self.use_footprint_constraints:
			# One vector-valued constraint keeps all four footprint circles outside
			# lethal/unknown space at every prediction step and intermediate substep.
			self.cons.append({'type': 'ineq', 'fun': self.footprint_clearance_constraint})
			
		self.initial_guess = np.zeros(self.no_ctrl_steps * 3)
		self.dt  = self.prediction_horizon /self.no_ctrl_steps
		self.last_time = 0.0
		self.update_opt_param = False
		self.subscription_footprint = self.create_subscription(
			PolygonStamped,
			'/local_costmap/published_footprint',
			self.footprint_callback,
			10)
		self.subscription_footprint
		self.old_goal = PoseStamped()
		self.no_acceleration_limit = False
		self.collision = False
		self.collision_footprint = False
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.control_interval = 0.0
		self.solver_failure_count = 0
		self.last_solver_failure_log_time = 0.0
		self.last_costmap_failure_log_time = 0.0

	def footprint_callback(self, msg):
		self.footprint = msg

	def f_constraint(self, initial, index):
		return  self.max_vel_trans - (np.sqrt((initial[0 + index * 3]) * (initial[0 + index * 3]) +(initial[1 + index * 3]) * (initial[1 + index * 3])))   

	def footprint_clearance_constraint(self, cmd_vel):
		"""Hard ESDF clearance constraints for the four-circle footprint."""
		constraint_count = (
			self.no_ctrl_steps * self.footprint_constraint_substeps *
			len(self.footprint_circle_centers))
		if self.costmap_ros.collision_esdf.distance_field is None:
			return np.full(constraint_count, -1.0)

		_, _, initial_world_yaw = self.euler_from_quaternion(
			self.current_pose.pose.orientation.x,
			self.current_pose.pose.orientation.y,
			self.current_pose.pose.orientation.z,
			self.current_pose.pose.orientation.w)
		initial_world_x = self.current_pose.pose.position.x
		initial_world_y = self.current_pose.pose.position.y
		cos_initial_yaw = np.cos(initial_world_yaw)
		sin_initial_yaw = np.sin(initial_world_yaw)
		substep_dt = self.dt / self.footprint_constraint_substeps

		local_x = 0.0
		local_y = 0.0
		local_yaw = 0.0
		constraints = []

		for i in range(self.no_ctrl_steps):
			vx = cmd_vel[3 * i]
			vy = cmd_vel[1 + 3 * i]
			omega = cmd_vel[2 + 3 * i]

			for _ in range(self.footprint_constraint_substeps):
				local_yaw += omega * substep_dt
				local_x += (
					vx * np.cos(local_yaw) - vy * np.sin(local_yaw)) * substep_dt
				local_y += (
					vx * np.sin(local_yaw) + vy * np.cos(local_yaw)) * substep_dt

				world_x = (
					initial_world_x + cos_initial_yaw * local_x -
					sin_initial_yaw * local_y)
				world_y = (
					initial_world_y + sin_initial_yaw * local_x +
					cos_initial_yaw * local_y)
				world_yaw = initial_world_yaw + local_yaw
				cos_world_yaw = np.cos(world_yaw)
				sin_world_yaw = np.sin(world_yaw)

				for offset_x, offset_y in self.footprint_circle_centers:
					circle_x = (
						world_x + cos_world_yaw * offset_x -
						sin_world_yaw * offset_y)
					circle_y = (
						world_y + sin_world_yaw * offset_x +
						cos_world_yaw * offset_y)
					clearance = self.costmap_ros.getCollisionDistanceBilinear(
						circle_x, circle_y)
					constraints.append(
						clearance - self.footprint_circle_radius -
						self.footprint_safety_margin)

		return np.asarray(constraints)

	# Acceleration constraint functions
	def acc_x_constraint(self, cmd_vel, index):
		"""Ensure vx doesn't exceed acceleration limits"""
		if index == 0:
			# First step: compare to last control
			delta_v = cmd_vel[0] - self.last_control[0]
		else:
			# Subsequent steps: compare to previous step
			delta_v = cmd_vel[0 + index * 3] - cmd_vel[0 + (index-1) * 3]
		
		max_delta = self.acc_x_limit * self.dt
		# Return positive value when constraint is satisfied
		return max_delta - abs(delta_v)
	
	def acc_y_constraint(self, cmd_vel, index):
		"""Ensure vy doesn't exceed acceleration limits"""
		if index == 0:
			delta_v = cmd_vel[1] - self.last_control[1]
		else:
			delta_v = cmd_vel[1 + index * 3] - cmd_vel[1 + (index-1) * 3]
		
		max_delta = self.acc_y_limit * self.dt
		return max_delta - abs(delta_v)
	
	def acc_theta_constraint(self, cmd_vel, index):
		"""Ensure angular velocity doesn't exceed acceleration limits"""
		if index == 0:
			delta_v = cmd_vel[2] - self.last_control[2]
		else:
			delta_v = cmd_vel[2 + index * 3] - cmd_vel[2 + (index-1) * 3]
		
		max_delta = self.acc_theta_limit * self.dt
		return max_delta - abs(delta_v)   

	def euler_from_quaternion(self, x, y, z, w):
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)

		return roll_x, pitch_y, yaw_z

	def quaternion_from_euler(self, roll, pitch, yaw):
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		cp = math.cos(pitch * 0.5)
		sp = math.sin(pitch * 0.5)
		cr = math.cos(roll * 0.5)
		sr = math.sin(roll * 0.5)

		q = [0] * 4
		q[0] = cy * cp * cr + sy * sp * sr
		q[1] = cy * cp * sr - sy * sp * cr
		q[2] = sy * cp * sr + cy * sp * cr
		q[3] = sy * cp * cr - cy * sp * sr

		return q

	def initial_guess_update(self, guess):
		shifted_guess = np.empty_like(guess)
		shifted_guess[:-3] = guess[3:]
		shifted_guess[-3:] = guess[-3:]
		return shifted_guess


	# objective function: Distance calculation should be all done in the local frame of the robot
	def objective(self, cmd_vel):
		self.cost_total = 0
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

		# NEW: Clear debug storage at start of each optimization
		self.debug_footprints = []
		self.debug_costs = []
		self.debug_poses = []

		_, _, initial_world_yaw = self.euler_from_quaternion(self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w)
		_, _, terminal_yaw = self.euler_from_quaternion(self.carrot_pose_terminal.pose.orientation.x, self.carrot_pose_terminal.pose.orientation.y, self.carrot_pose_terminal.pose.orientation.z, self.carrot_pose_terminal.pose.orientation.w)

		initial_world_x = self.current_pose.pose.position.x
		initial_world_y = self.current_pose.pose.position.y
		cos_initial_yaw = np.cos(initial_world_yaw)
		sin_initial_yaw = np.sin(initial_world_yaw)

		for i in range((self.no_ctrl_steps)):
			self.costmap_cost = 0

			# Update the position for the predicted velocity
			self.z += cmd_vel[2+3*i] *  self.dt
			self.x += (cmd_vel[0+3*i]*np.cos(self.z)* self.dt - cmd_vel[1+3*i]*np.sin(self.z)* self.dt) 
			self.y += (cmd_vel[0+3*i]*np.sin(self.z)* self.dt + cmd_vel[1+3*i]*np.cos(self.z)* self.dt)
			
			pos_x = initial_world_x + cos_initial_yaw * self.x - sin_initial_yaw * self.y
			pos_y = initial_world_y + sin_initial_yaw * self.x + cos_initial_yaw * self.y
			
			# Check distance to nearest obstacle for adaptive behavior
			# Use center point for this general context switch
			mx_curr, my_curr = self.costmap_ros.getWorldToMap(pos_x, pos_y)
			dist_to_obs = self.costmap_ros.esdf.get_distance_bilinear(mx_curr, my_curr)
			
			# Context-aware lookahead selection
			# Use tight lookahead ONLY when:
			# 1. Close to obstacles (< threshold) AND
			# 2. Making a sharp turn (> sharp_turn_threshold)
			use_tight_lookahead = (dist_to_obs < self.tight_lookahead_dist_threshold and 
			                       self.effective_turn_angle_ > self.sharp_turn_threshold)
			
			if use_tight_lookahead and not self.update_opt_param:
				# Use tight lookahead for precise control near obstacles during turns
				step_target_yaw = self.turn_yaw_tight_
				curr_pos = np.array((self.carrot_pose_tight.pose.position.x, self.carrot_pose_tight.pose.position.y))
			else:
				# Use normal lookahead for smoother motion
				step_target_yaw = self.turn_yaw_
				curr_pos = np.array((self.carrot_pose.pose.position.x, self.carrot_pose.pose.position.y))
			
			# i) Evaluating cost for error in displacement and orientation
			step_dist_error =  np.linalg.norm(curr_pos - np.array((self.x, self.y)))

			# Adaptive Orientation Control
			if dist_to_obs < self.tight_lookahead_dist_threshold and not self.update_opt_param:
				# Narrow space: Bidirectional Alignment
				# Both self.turn_yaw_ and self.z are in local frame (relative to base_link)
				# self.turn_yaw_ is the angle to goal (0 = straight ahead)
				
				# Calculate both alignment options
				forward_target = step_target_yaw
				# For backward, add/subtract pi to flip 180 degrees
				if step_target_yaw >= 0:
					backward_target = step_target_yaw - np.pi
				else:
					backward_target = step_target_yaw + np.pi
				
				# Calculate angular distance from current orientation (self.z) to each target
				# Use ROS angles library for proper angle wrapping
				diff_to_forward = abs(shortest_angular_distance(self.z, forward_target))
				diff_to_backward = abs(shortest_angular_distance(self.z, backward_target))
				
				# Choose the alignment that requires less rotation
				if diff_to_backward < diff_to_forward:
					# Backward alignment is closer
					step_target_yaw = backward_target
				else:
					# Forward alignment is closer
					step_target_yaw = forward_target
				
			# Calculate angular error (handling wrapping)
			step_orient_error = shortest_angular_distance(self.z, step_target_yaw)
			# step_orient_error = np.arctan2(np.sin(diff), np.cos(diff))

			self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient * step_orient_error**2)) / self.no_ctrl_steps            
			self.cost_total += self.w_control * (np.linalg.norm(np.array((self.current_velocity.linear.x , self.current_velocity.linear.y, \
			self.current_velocity.angular.z )) - np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]))))  / self.no_ctrl_steps

		# iii) terminal cost
		# Use carrot_pose_terminal (far lookahead) for terminal cost to avoid redundancy
		# carrot_pose is already being tracked in step-by-step costs
		terminal_goal_pos = np.array((self.carrot_pose_terminal.pose.position.x, self.carrot_pose_terminal.pose.position.y))
		step_dist_error =  np.linalg.norm(terminal_goal_pos - np.array((self.x, self.y)))
		step_orient_error = shortest_angular_distance(self.z, terminal_yaw)
		
		self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient * step_orient_error**2)) * self.w_terminal
		
		return self.cost_total

	# NEW: Function to publish debug footprints
	def publishDebugFootprints(self, cmd_vel):
		"""
		Publishes the stored footprints as a Path for visualization in RViz.
		Each footprint corner becomes a pose in the path.
		"""
		debug_footprints = []
		# add the next predicted footprint to the debug footprints
		# use cmd_vel to calculate the updates to the footprint
		update_footprint = copy.deepcopy(self.footprint)
		_, _, odom_yaw = self.euler_from_quaternion(self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w)
		next_odom_yaw = odom_yaw + cmd_vel[2] * self.control_time_scale * self.dt
		for j in range(len(update_footprint.polygon.points)):
			update_footprint.polygon.points[j].x += (cmd_vel[0]*np.cos(next_odom_yaw)* self.control_time_scale * self.dt - cmd_vel[1]*np.sin(next_odom_yaw)* self.control_time_scale * self.dt) 
			update_footprint.polygon.points[j].y += (cmd_vel[0]*np.sin(next_odom_yaw)* self.control_time_scale * self.dt + cmd_vel[1]*np.cos(next_odom_yaw)* self.control_time_scale * self.dt)
		debug_footprints.append(copy.deepcopy(update_footprint))		
		
		debug_path = Path()
		debug_path.header.stamp = self.get_clock().now().to_msg()
		debug_path.header.frame_id = "map"
		
		# Add each corner of the footprint as a pose
		for j, point in enumerate(update_footprint.polygon.points):
			pose = PoseStamped()
			pose.header.stamp = self.get_clock().now().to_msg()
			pose.header.frame_id = "map"
			pose.pose.position.x = point.x
			pose.pose.position.y = point.y
			pose.pose.position.z = 0.1  # Stack vertically for visualization
			
			# Use orientation to encode step number (for color coding in RViz)
			q = self.quaternion_from_euler(0, 0, self.z)
			pose.pose.orientation.w = q[0]
			pose.pose.orientation.x = q[1]
			pose.pose.orientation.y = q[2]
			pose.pose.orientation.z = q[3]
			debug_path.poses.append(pose)
		
		self.PubFootprintArray.publish(debug_path)

	def publishLocalPlan(self, x):
		self.local_plan.poses.clear()
		try:
			now = rclpy.time.Time()
			trans = self.tf_buffer.lookup_transform(
				"map",
				"base_link",
				now)
		except TransformException as ex:
			self.get_logger().info(
				f'Could not transform map to base_link: {ex}')
			return

		pos_x = trans.transform.translation.x
		pos_y = trans.transform.translation.y
		_, _, yaw = self.euler_from_quaternion(trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w)

		pose = PoseStamped()
		pose.pose.position.x = pos_x
		pose.pose.position.y = pos_y
		self.local_plan.poses.append(pose)

		for i in range((self.no_ctrl_steps)):
			pose = PoseStamped()
			yaw += x[2+3*i] * self.dt
			pos_x += x[3*i]*np.cos(yaw) * self.dt - x[1+3*i]*np.sin(yaw) * self.dt
			pos_y += x[3*i]*np.sin(yaw) * self.dt + x[1+3*i]*np.cos(yaw) * self.dt   
			pose.pose.position.x = pos_x
			pose.pose.position.y = pos_y
			pose.header.stamp = self.get_clock().now().to_msg()
			q = self.quaternion_from_euler(0, 0, yaw)
			pose.pose.orientation.w = q[0]
			pose.pose.orientation.x = q[1]
			pose.pose.orientation.y = q[2]
			pose.pose.orientation.z = q[3]
			self.local_plan.poses.append(pose)

		self.local_plan.header.stamp = self.get_clock().now().to_msg()
		self.local_plan.header.frame_id = "map"
		self.PubRaysPath.publish(self.local_plan)

	def collision_check(self, x):
		# Collision check with footprint
		pos_x = self.current_pose.pose.position.x
		pos_y = self.current_pose.pose.position.y
		_, _, odom_yaw = self.euler_from_quaternion(self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w)

		# Calculate collision check lookahead time
		# Scale dt to check at an intermediate point between control_interval and dt
		# With scale=0.4 and dt=0.267s, this checks at ~0.11s (about 3 control cycles ahead)
		collision_time = self.dt * self.control_time_scale
		next_odom_yaw = odom_yaw + x[2] * collision_time
		next_pos_x = pos_x + x[0]*np.cos(next_odom_yaw) * collision_time - x[1]*np.sin(next_odom_yaw) * collision_time
		next_pos_y = pos_y + x[0]*np.sin(next_odom_yaw) * collision_time + x[1]*np.cos(next_odom_yaw) * collision_time

		# Check point collision at next predicted position
		mx1, my1 = self.costmap_ros.getWorldToMap(next_pos_x, next_pos_y)
		col = self.costmap_ros.getCost(mx1, my1)
		if (col >= 0.99):
			self.collision = True
			print("Collision ahead, stopping the robot")
		else:
			self.collision = False

		# Check footprint collision at next predicted pose
		# Footprint is already in map frame, offset by predicted displacement
		update_footprint = copy.deepcopy(self.footprint)
		# Calculate displacement in world frame
		delta_x = x[0]*np.cos(odom_yaw) * collision_time - x[1]*np.sin(odom_yaw) * collision_time
		delta_y = x[0]*np.sin(odom_yaw) * collision_time + x[1]*np.cos(odom_yaw) * collision_time
		for j in range(len(update_footprint.polygon.points)):
			update_footprint.polygon.points[j].x += delta_x
			update_footprint.polygon.points[j].y += delta_y
		
		if (self.costmap_ros.getFootprintCost(update_footprint.polygon) == 1.0):
			self.collision_footprint = True
			print("Footprint in collision, stopping the robot")
		else:
			self.collision_footprint = False

	def optimizer(self, request, response):
		self.current_pose = request.current_pose
		self.carrot_pose = request.carrot_pose
		self.carrot_pose_tight = request.carrot_pose_tight
		self.carrot_pose_terminal = request.carrot_pose_terminal
		self.current_velocity = request.current_vel
		self.goal_pose = request.goal_pose
		self.update_opt_param = request.switch_opt
		self.control_interval = request.control_interval
		self.turn_yaw_ = request.turn_yaw
		self.turn_yaw_tight_ = request.turn_yaw_tight
		self.effective_turn_angle_ = request.effective_turn_angle

		# on new goal reset all the flags and initializers
		if (self.old_goal != self.goal_pose):
			self.initial_guess = np.zeros(self.no_ctrl_steps*3)
			self.last_control = [0,0,0]
			self.waiting_time = 0.0

		costmap_is_current = self.costmap_ros.isCurrent(self.costmap_max_age)
		costmap_frame_matches = (
			bool(self.current_pose.header.frame_id) and
			self.current_pose.header.frame_id == self.costmap_ros.frame_id)
		if not costmap_is_current or not costmap_frame_matches:
			now = time.monotonic()
			if now - self.last_costmap_failure_log_time >= 1.0:
				self.get_logger().error(
					'Optimizer stopped for invalid costmap input: '
					f'age={self.costmap_ros.getAge():.3f}s, '
					f'max_age={self.costmap_max_age:.3f}s, '
					f'pose_frame="{self.current_pose.header.frame_id}", '
					f'costmap_frame="{self.costmap_ros.frame_id}"')
				self.last_costmap_failure_log_time = now

			response.output_vel.twist.linear.x = 0.0
			response.output_vel.twist.linear.y = 0.0
			response.output_vel.twist.angular.z = 0.0
			self.last_control = [0.0, 0.0, 0.0]
			self.initial_guess = np.zeros(self.no_ctrl_steps * 3)
			self.old_goal = self.goal_pose
			return response

		x = minimize(self.objective, self.initial_guess,
				method='SLSQP', bounds=self.bnds, constraints=self.cons,
				options={
					'ftol': self.opt_tolerance,
					'eps': self.solver_finite_difference_step,
					'disp': False})

		# Never execute a failed or invalid optimizer result. SciPy may still
		# populate x when SLSQP terminates unsuccessfully, but that candidate is
		# not guaranteed to satisfy the configured bounds and constraints.
		expected_solution_size = self.no_ctrl_steps * 3
		solution = np.asarray(getattr(x, 'x', []), dtype=float)
		minimum_footprint_constraint = float('nan')
		footprint_constraint_valid = not self.use_footprint_constraints
		if (self.use_footprint_constraints and
				solution.shape == (expected_solution_size,) and
				np.all(np.isfinite(solution))):
			footprint_constraints = self.footprint_clearance_constraint(solution)
			if (footprint_constraints.size > 0 and
					np.all(np.isfinite(footprint_constraints))):
				minimum_footprint_constraint = float(np.min(footprint_constraints))
				footprint_constraint_valid = (
					minimum_footprint_constraint >=
					-self.footprint_constraint_tolerance)

		# Costmap freshness is checked immediately before minimize(). Rechecking
		# it here would reject otherwise valid results merely because the
		# single-threaded service callback prevents costmap callbacks while SLSQP
		# is running.
		solver_result_valid = (
			x.success and
			solution.shape == (expected_solution_size,) and
			np.all(np.isfinite(solution)) and
			np.isfinite(getattr(x, 'fun', np.nan)) and
			footprint_constraint_valid)

		if not solver_result_valid:
			self.solver_failure_count += 1
			now = time.monotonic()
			if now - self.last_solver_failure_log_time >= 1.0:
				self.get_logger().error(
					'SLSQP result rejected: '
					f'status={getattr(x, "status", "unknown")}, '
					f'message={getattr(x, "message", "unknown")}, '
					f'min_footprint_constraint={minimum_footprint_constraint:.4f}, '
					f'consecutive_failures={self.solver_failure_count}')
				self.last_solver_failure_log_time = now

			response.output_vel.twist.linear.x = 0.0
			response.output_vel.twist.linear.y = 0.0
			response.output_vel.twist.angular.z = 0.0
			self.last_control = [0.0, 0.0, 0.0]
			self.initial_guess = np.zeros(expected_solution_size)
			self.old_goal = self.goal_pose
			return response

		self.solver_failure_count = 0
		
		# Predicted footprint publishing is disabled while its frame handling is
		# validated independently from the controller output.
		# self.publishDebugFootprints(solution)

		# Check collision
		# self.collision_check(solution)
		
		self.publishLocalPlan(solution)
		for i in range(0,3):
			solution[i] = solution[i] * self.low_pass_gain + self.last_control[i] * (1 - self.low_pass_gain)

		current_time = time.time()
		delta_t = current_time - self.last_time
		self.last_time = current_time

		if (self.collision == True or self.collision_footprint == True):
			response.output_vel.twist.linear.x = 0.0
			response.output_vel.twist.linear.y = 0.0
			response.output_vel.twist.angular.z = 0.0
			self.waiting_time += delta_t
			if (self.waiting_time >= 3.0):
				self.collision = False
				self.waiting_time = 0.0
		else:
			temp_x = np.fmin(solution[0], self.last_control[0] + self.acc_x_limit * self.control_interval)
			temp_y = np.fmin(solution[1], self.last_control[1] + self.acc_y_limit * self.control_interval)
			temp_z = np.fmin(solution[2], self.last_control[2] + self.acc_theta_limit * self.control_interval)

			response.output_vel.twist.linear.x = np.fmax(temp_x, self.last_control[0] - self.acc_x_limit * self.control_interval)
			response.output_vel.twist.linear.y = np.fmax(temp_y, self.last_control[1] - self.acc_y_limit * self.control_interval) 
			response.output_vel.twist.angular.z = np.fmax(temp_z, self.last_control[2] - self.acc_theta_limit * self.control_interval)

		self.last_control[0] = response.output_vel.twist.linear.x 
		self.last_control[1] = response.output_vel.twist.linear.y 
		self.last_control[2] = response.output_vel.twist.angular.z

		self.initial_guess = self.initial_guess_update(solution)

		self.old_goal = self.goal_pose
		return response

	def cb_params(self, data):
		for parameter in data:
			if parameter.type_ == Parameter.Type.DOUBLE:
				if parameter.name == "min_vel_x":
					self.min_vel_x = parameter.value
				elif parameter.name == "min_vel_y":
					self.min_vel_y = parameter.value
				elif parameter.name == "min_vel_trans":
					self.min_vel_trans = parameter.value
				elif parameter.name == "min_vel_theta":
					self.min_vel_theta = parameter.value
				elif parameter.name == "max_vel_x":
					self.max_vel_x = parameter.value
				elif parameter.name == "max_vel_y":
					self.max_vel_y = parameter.value
				elif parameter.name == "max_vel_trans":
					self.max_vel_trans = parameter.value
				elif parameter.name == "max_vel_theta":
					self.max_vel_theta = parameter.value
				elif parameter.name == "w_trans":
					self.w_trans = parameter.value
				elif parameter.name == "w_orient":
					self.w_orient = parameter.value
				elif parameter.name == "w_control":
					self.w_control = parameter.value
				elif parameter.name == "w_terminal":
					self.w_terminal = parameter.value
				elif parameter.name == "w_costmap":
					self.w_costmap = parameter.value
				elif parameter.name == "w_footprint":
					self.w_footprint = parameter.value
				else:
					print("The selected parameter cannot be dynamically changed")

		return SetParametersResult(successful = True)

def main(args=None):
	rclpy.init(args = args)
	MpcOptimization = MpcOptimizationServer()
	rclpy.spin(MpcOptimization)

if __name__ == '__main__':
	main()
