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

class MpcOptimizationServer(Node):
	def __init__(self):
		super().__init__('mpc_optimization_server')

		# declare parameters
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
		# self.declare_parameter('control_horizon', value = 0.5)
		self.declare_parameter('control_steps', value = 3)

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

		self.srv = self.create_service(Optimizer, 'optimizer', self.optimizer)
		self.add_on_set_parameters_callback(self.cb_params)
		self.PubRaysPath = self.create_publisher(Path, 'local_plan', 10)
		self.Pubfootprint = self.create_publisher(PolygonStamped, 'predicted_footprint', 10)
		self.current_pose = Pose()
		self.carrot_pose = PoseStamped()
		self.goal_pose = PoseStamped()
		self.current_velocity = TwistStamped()
		self.local_plan = Path()

		self.cost_total = 0.0
		self.costmap_cost = 0.0
		self.last_control = [0,0,0]
		self.costmap_ros = Costmap2d(self)

		self.update_x = 0.0
		self.update_y = 0.0
		self.update_yaw = 0.0
		self.size_x_ = 0

		self.bnds  = list()
		self.cons = []
		b_x_vel = (self.min_vel_x, self.max_vel_x)
		b_y_vel = (self.min_vel_y, self.max_vel_y)
		b_rot = (self.min_vel_theta, self.max_vel_theta)
		for i in range(self.no_ctrl_steps):
			self.bnds.append(b_x_vel)
			self.bnds.append(b_y_vel)
			self.bnds.append(b_rot)
			self.cons.append({'type': 'ineq', 'fun': partial(self.f_constraint, index = i)})
			
		self.initial_guess = np.zeros(self.no_ctrl_steps * 3)
		self.dt  = self.prediction_horizon /self.no_ctrl_steps #time_interval_between_control_pts used in integration
		self.last_time = 0.0
		self.update_opt_param = False
		self.subscription_footprint = self.create_subscription(
            PolygonStamped,
            '/local_costmap/published_footprint',
            self.footprint_callback,
            10)
		self.subscription_footprint  # prevent unused variable warning.
		self.old_goal = PoseStamped()
		self.no_acceleration_limit = False
		self.collision = False
		self.collision_footprint = False
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.control_interval = 0.0;

	def footprint_callback(self, msg):
		self.footprint = msg.polygon

	def f_constraint(self, initial, index):
		return  self.max_vel_trans - (np.sqrt((initial[0 + index * 3]) * (initial[0 + index * 3]) +(initial[1 + index * 3]) * (initial[1 + index * 3])))   

	def euler_from_quaternion(self, x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
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

		return roll_x, pitch_y, yaw_z # in radians

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

	def initial_guess_update(self, init_guess,guess):
		for i in range(0, self.no_ctrl_steps-1):
			init_guess[0+3*i:3+3*i] = guess[3+3*i:6+3*i]
		init_guess[0+3*(self.no_ctrl_steps-1):3+3*(self.no_ctrl_steps-1)] = guess[0:3]
		return init_guess

	def objective(self, cmd_vel):

		self.cost_total= 0
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

		_, _, target_yaw = self.euler_from_quaternion(self.carrot_pose.pose.orientation.x, self.carrot_pose.pose.orientation.y, self.carrot_pose.pose.orientation.z, self.carrot_pose.pose.orientation.w)
		_, _, final_yaw = self.euler_from_quaternion(self.goal_pose.orientation.x, self.goal_pose.orientation.y, self.goal_pose.orientation.z, self.goal_pose.orientation.w)
		_, _, odom_yaw = self.euler_from_quaternion(self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.goal_pose.orientation.w)

		count = 1.0
		tot_x = self.current_velocity.linear.x 
		tot_y = self.current_velocity.linear.y 
		tot_z = self.current_velocity.angular.z
		curr_pos = np.array((self.carrot_pose.pose.position.x,self.carrot_pose.pose.position.y))
		pos_x = self.current_pose.pose.position.x
		pos_y = self.current_pose.pose.position.y
		c_c = 0.0
		
		for i in range((self.no_ctrl_steps)):
			self.costmap_cost = 0
			update_footprint = Polygon()
			update_footprint.points = self.footprint.points

			# Update the position for the predicted velocity
			self.z += cmd_vel[2+3*i] *  self.dt
			self.x += (cmd_vel[0+3*i]*np.cos(self.z)* self.dt - cmd_vel[1+3*i]*np.sin(self.z)* self.dt) 
			self.y += (cmd_vel[0+3*i]*np.sin(self.z)* self.dt + cmd_vel[1+3*i]*np.cos(self.z)* self.dt)
			
			odom_yaw += cmd_vel[2+3*i] * self.dt 
			pos_x += cmd_vel[0+3*i] *np.cos(odom_yaw) *  self.dt  - cmd_vel[1+3*i] * np.sin(odom_yaw) *  self.dt
			pos_y += cmd_vel[0+3*i] *np.sin(odom_yaw) *  self.dt  + cmd_vel[1+3*i] * np.cos(odom_yaw) *  self.dt
			
			for j in range(0, len(self.footprint.points)):
				a = self.footprint.points[j].x
				b = self.footprint.points[j].y
				update_footprint.points[j].x = self.x + self.footprint.points[j].x * np.cos(self.z)  - self.footprint.points[j].y * np.sin(self.z) 
				update_footprint.points[j].y = self.y + self.footprint.points[j].x * np.sin(self.z)  + self.footprint.points[j].y * np.cos(self.z) 
				self.footprint.points[j].x = a
				self.footprint.points[j].y = b

			mx1, my1 = self.costmap_ros.getWorldToMap(pos_x, pos_y)
			self.costmap_cost += self.costmap_ros.getCost(mx1, my1) ** 2

			# i) Evaluvating cost for error in displacement and orientation
			step_dist_error =  np.linalg.norm(curr_pos - np.array((self.x, self.y)))
			step_orient_error = target_yaw - self.z
			self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient * step_orient_error**2)) / self.no_ctrl_steps            
			self.cost_total += self.w_control * (np.linalg.norm(np.array((self.current_velocity.linear.x , self.current_velocity.linear.y, \
			self.current_velocity.angular.z )) - np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]))))  / self.no_ctrl_steps          
			
			# ii) Evaluvating obstacle cost
			if(self.costmap_ros.getCost(mx1, my1) == 1.0):
				self.cost_total +=  self.costmap_cost * 1000 / self.no_ctrl_steps
			else:
				self.cost_total +=  self.w_costmap_scale * self.costmap_cost / self.no_ctrl_steps
			
			if(self.costmap_ros.getFootprintCost(update_footprint) == 1.0):
				self.cost_total += (self.costmap_ros.getFootprintCost(update_footprint)**2) * self.w_footprint_scale / self.no_ctrl_steps

		# iii) terminal self.cost
		step_dist_error =  np.linalg.norm(curr_pos - np.array((self.goal_pose.position.x,self.goal_pose.position.y)))
		step_orient_error = final_yaw - self.z
		self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient* step_orient_error**2)) * self.w_terminal 
		return self.cost_total

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
		footprint = self.footprint
		pos_x = self.current_pose.pose.position.x
		pos_y = self.current_pose.pose.position.y
		_, _, odom_yaw = self.euler_from_quaternion(self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w)

		pose = PoseStamped()
		pose.pose.position.x = pos_x
		pose.pose.position.y = pos_y

		for i in range((self.no_ctrl_steps)):
			pose = PoseStamped()
			odom_yaw += x[2+3*i] * self.dt
			pos_x += x[3*i]*np.cos(odom_yaw) * self.dt - x[1+3*i]*np.sin(odom_yaw) * self.dt
			pos_y += x[3*i]*np.sin(odom_yaw) * self.dt + x[1+3*i]*np.cos(odom_yaw) * self.dt   
			pose.pose.position.x = pos_x
			pose.pose.position.y = pos_y
			pose.header.stamp = self.get_clock().now().to_msg()
			q = self.quaternion_from_euler(0, 0, odom_yaw)
			mx1, my1 = self.costmap_ros.getWorldToMap(pos_x, pos_y)
			col = self.costmap_ros.getCost(mx1, my1)
			pose.pose.orientation.w = q[0]
			pose.pose.orientation.x = q[1]
			pose.pose.orientation.y = q[2]
			pose.pose.orientation.z = q[3]
			if (col >= 0.99):
				self.collision = True
				print("Collision ahead, stopping the robot")
				break

		if (self.costmap_ros.getFootprintCost(footprint) == 1.0):
			self.collision_footprint = True
			print("Footprint in collision, stopping the robot")
		else:
			self.collision_footprint = False

	def optimizer(self, request, response):
		self.current_pose = request.current_pose
		self.carrot_pose = request.carrot_pose
		self.current_velocity = request.current_vel
		self.goal_pose = request.goal_pose
		self.update_opt_param = request.switch_opt
		self.control_interval = request.control_interval

		# on new goal reset all the flags and initializers
		if (self.old_goal != self.goal_pose):
			self.initial_guess = np.zeros(self.no_ctrl_steps*3)
			self.last_control = [0,0,0]
			self.waiting_time = 0.0

		x = minimize(self.objective, self.initial_guess,
				method='SLSQP',bounds= self.bnds, constraints = self.cons, options={'ftol':self.opt_tolerance,'disp':False})		
		self.publishLocalPlan(x.x)
		for i in range(0,3):
			x.x[i] = x.x[i] * self.low_pass_gain + self.last_control[i] * (1 - self.low_pass_gain)

		current_time = time.time()
		delta_t = current_time - self.last_time
		self.last_time = current_time
		self.collision_check(x.x)

		if (self.collision == True or self.collision_footprint == True):
			response.output_vel.twist.linear.x = 0.0
			response.output_vel.twist.linear.y = 0.0
			response.output_vel.twist.angular.z = 0.0
			self.waiting_time += delta_t
			# After waiting for 3 seconds for the obstacle to clear, the robot proceeds further in the next service call.
			if (self.waiting_time >= 3.0):
				self.collision = False
				self.waiting_time = 0.0
		else:
			# avoiding sudden jerks and inertia
			temp_x = np.fmin(x.x[0], self.last_control[0] + self.acc_x_limit * self.control_interval)
			temp_y = np.fmin(x.x[1], self.last_control[1] + self.acc_y_limit * self.control_interval) 
			temp_z = np.fmin(x.x[2], self.last_control[2] + self.acc_theta_limit * self.control_interval)

			response.output_vel.twist.linear.x = np.fmax(temp_x, self.last_control[0] - self.acc_x_limit * self.control_interval)
			response.output_vel.twist.linear.y = np.fmax(temp_y, self.last_control[1] - self.acc_y_limit * self.control_interval) 
			response.output_vel.twist.angular.z = np.fmax(temp_z, self.last_control[2] - self.acc_theta_limit * self.control_interval)

		self.last_control[0] = response.output_vel.twist.linear.x 
		self.last_control[1] = response.output_vel.twist.linear.y 
		self.last_control[2] = response.output_vel.twist.angular.z

		if (x.success):
			self.initial_guess = self.initial_guess_update(self.initial_guess, x.x)
		else:
			self.initial_guess = x.x

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
				else:
					print("The selected parameter cannot be dynamically changed")

def main(args=None):
	rclpy.init(args = args)
	MpcOptimization = MpcOptimizationServer()
	rclpy.spin(MpcOptimization)

if __name__ == '__main__':
	main()
