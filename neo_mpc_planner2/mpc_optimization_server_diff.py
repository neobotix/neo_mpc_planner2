#!/usr/bin/env python3

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

class MpcOptimizationServerDiff(Node):
	def __init__(self):
		super().__init__('mpc_optimization_server')
		self.srv = self.create_service(Optimizer, 'optimizer_diff', self.optimizer)
		self.PubRaysPath = self.create_publisher(Path, 'local_plan', 10)
		self.Pubfootprint = self.create_publisher(PolygonStamped, 'predicted_footprint', 10)
		self.current_pose = Pose()
		self.carrot_pose = PoseStamped()
		self.goal_pose = PoseStamped()
		self.current_velocity = TwistStamped()
		self.local_plan = Path()
		self.no_ctrl_steps = 3
		self.low_pass_gain = 0.5
		self.prediction_horizon = 0.8

		self.cost_total = 0.0
		self.costmap_cost = 0.0
		self.last_control = [0,0,0]
		self.c = Costmap2d(self)
		self.d = Costmap2d(self)

		self.count = 0

		self.update_x = 0.0
		self.update_yaw = 0.0

		self.w_trans = 0.0
		self.w_orient = 0.0
		self.w_control = 0.0
		self.w_terminal = 0.0
		self.size_x_ = 0
		self.waiting_time = 0.0

		self.bnds  = list()
		b_x_vel = (-0.7, 0.7)
		b_rot = (-0.5, 0.5)
		for i in range(self.no_ctrl_steps):
			self.bnds.append(b_x_vel)
			self.bnds.append(b_rot)
			
		self.initial_guess = np.zeros(self.no_ctrl_steps * 2)
		self.dt  = self.prediction_horizon /self.no_ctrl_steps #time_interval_between_control_pts used in integration
		self.last_time = 0.0
		self.update_opt_param = False

		self.old_goal = PoseStamped()
		self.collision = False
		self.collision_footprint = False
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

	def footprint_callback(self, msg):
		self.footprint = msg.polygon

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
		tot_z = self.current_velocity.angular.z
		curr_pos = np.array((self.carrot_pose.pose.position.x,self.carrot_pose.pose.position.y))
		pos_x = self.current_pose.pose.position.x
		pos_y = self.current_pose.pose.position.y
		c_c = 0.0
		
		for i in range((self.no_ctrl_steps)):
			self.costmap_cost = 0
			footprintss = Polygon()
			footprintss.points = self.footprint.points

			# Update the position for the predicted velocity
			self.z += cmd_vel[1+3*i] *  self.dt
			self.x += cmd_vel[0+3*i] * np.cos(self.z) * self.dt
			self.y += cmd_vel[0+3*i] * np.sin(self.z) * self.dt
			
			odom_yaw += cmd_vel[1+3*i] * self.dt 
			pos_x += cmd_vel[0+3*i] * np.cos(odom_yaw) * self.dt
			pos_y += cmd_vel[0+3*i] * np.sin(odom_yaw) * self.dt
			
			for j in range(0, len(self.footprint.points)):
				a = self.footprint.points[j].x
				b = self.footprint.points[j].y
				footprintss.points[j].x = self.x + self.footprint.points[j].x * np.cos(self.z)  - self.footprint.points[j].y * np.sin(self.z) 
				footprintss.points[j].y = self.y + self.footprint.points[j].x * np.sin(self.z)  + self.footprint.points[j].y * np.cos(self.z) 
				self.footprint.points[j].x = a
				self.footprint.points[j].y = b

			mx1, my1 = self.c.getWorldToMap(pos_x, pos_y)
			self.costmap_cost += self.c.getCost(mx1, my1) ** 2

			# i) Evaluvating cost for error in displacement and orientation
			step_dist_error =  np.linalg.norm(curr_pos - np.array((self.x, self.y)))
			step_orient_error = target_yaw - self.z
			self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient * step_orient_error**2)) / self.no_ctrl_steps            
			self.cost_total += self.w_control * (np.linalg.norm(np.array((self.current_velocity.linear.x , self.current_velocity.linear.y, \
			self.current_velocity.angular.z )) - np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]))))  / self.no_ctrl_steps          
			# ii) Evaluvating obstacle cost
			if(self.c.getCost(mx1, my1) == 1.0):
				self.cost_total +=  self.costmap_cost*1000 / self.no_ctrl_steps
			else:
				self.cost_total +=  self.w_costmap_scale * self.costmap_cost / self.no_ctrl_steps
			
			if(self.d.getFootprintCost(footprintss) == 1.0):
				self.cost_total += (self.d.getFootprintCost(footprintss)**2) * 2000 / self.no_ctrl_steps

		# iii) terminal self.cost
		step_dist_error =  np.linalg.norm(curr_pos - np.array((self.goal_pose.position.x,self.goal_pose.position.y)))
		step_orient_error = final_yaw - self.z
		self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient* step_orient_error**2))*self.w_terminal 
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

		for i in range((self.no_ctrl_steps) - 2):
			x = np.append(x, np.array([x[3 * i], x[1 + 3*i], x[2 + 3*i]]))

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

		for i in range((self.no_ctrl_steps) - 2):
			x = np.append(x, np.array([x[3 * i], x[1 + 3*i], x[2 + 3*i]]))

		for i in range((self.no_ctrl_steps)):
			pose = PoseStamped()
			odom_yaw += x[2+3*i] * self.dt
			pos_x += x[3*i]*np.cos(odom_yaw) * self.dt - x[1+3*i]*np.sin(odom_yaw) * self.dt
			pos_y += x[3*i]*np.sin(odom_yaw) * self.dt + x[1+3*i]*np.cos(odom_yaw) * self.dt   
			pose.pose.position.x = pos_x
			pose.pose.position.y = pos_y
			pose.header.stamp = self.get_clock().now().to_msg()
			q = self.quaternion_from_euler(0, 0, odom_yaw)
			mx1, my1 = self.c.getWorldToMap(pos_x, pos_y)
			col = self.c.getCost(mx1, my1)
			pose.pose.orientation.w = q[0]
			pose.pose.orientation.x = q[1]
			pose.pose.orientation.y = q[2]
			pose.pose.orientation.z = q[3]
			if (col >= 0.99):
				self.collision = True
				print("Collision ahead, stopping the robot")
				break

		if (self.c.getFootprintCost(footprint) == 1.0):
			self.collision_footprint = True
			print("Footprint in collision, stopping the robot")
		else:
			self.collision_footprint = False

	def optimizer(self, request, response):

		self.w_trans = 0.82
		self.w_orient = 0.50
		self.w_control= 0.06
		self.w_terminal = 0.08
		self.w_costmap_scale = 0.05

		self.current_pose = request.current_pose
		self.carrot_pose = request.carrot_pose
		self.current_velocity = request.current_vel
		self.goal_pose = request.goal_pose
		self.update_opt_param = request.switch_opt

		# on new goal reset all the flags and initializers
		if (self.old_goal != self.goal_pose):
			self.initial_guess = np.zeros(self.no_ctrl_steps*3)
			self.last_control = [0,0,0]
			self.count = 0
			self.waiting_time = 0.0

		x = minimize(self.objective, self.initial_guess,
				method='SLSQP',bounds= self.bnds, constraints = self.cons, options={'ftol':1e-3,'disp':False})		
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
			temp_x = np.sign(x.x[0]) * np.fmin(abs(x.x[0]), abs(self.last_control[0]) + 0.10 * self.dt)
			temp_y = np.sign(x.x[1]) * np.fmin(abs(x.x[1]), abs(self.last_control[1])+ 0.10 * self.dt) 
			temp_z = np.sign(x.x[2]) * np.fmin(abs(x.x[2]), abs(self.last_control[2]) + 3.0 * self.dt)

			response.output_vel.twist.linear.x = np.sign(temp_x) * np.fmax(abs(temp_x), abs(self.last_control[0]) - 2.5 * self.dt)
			response.output_vel.twist.linear.y = np.sign(temp_y) * np.fmax(abs(temp_y), abs(self.last_control[1])- 2.5 * self.dt) 
			response.output_vel.twist.angular.z = np.sign(temp_z) * np.fmax(abs(temp_z), abs(self.last_control[2]) - 3.0 * self.dt)

		self.last_control[0] = response.output_vel.twist.linear.x 
		self.last_control[1] = response.output_vel.twist.linear.y 
		self.last_control[2] = response.output_vel.twist.angular.z

		if (x.success):
			self.initial_guess = self.initial_guess_update(self.initial_guess, x.x)
		else:
			self.initial_guess = x.x

		self.old_goal = self.goal_pose
		return response

def main(args=None):
	rclpy.init(args = args)
	MpcOptimization = MpcOptimizationServer()
	rclpy.spin(MpcOptimization)

if __name__ == '__main__':
	main()
