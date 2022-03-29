#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from neo_srvs2.srv import Optimizer
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
from scipy.optimize import minimize
import math
from functools import partial
import time

class MpcOptimizationServer(Node):
	def __init__(self):
		super().__init__('mpc_optimization_server')
		self.srv = self.create_service(Optimizer, 'optimizer', self.optimizer)
		self.subscription = self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 10)
		self.PubRaysPath = self.create_publisher(Path, 'local_plan', 10)
		self.current_pose = Pose()
		self.carrot_pose = PoseStamped()
		self.goal_pose = PoseStamped()
		self.current_velocity = TwistStamped()
		self.costmap = OccupancyGrid()
		self.no_ctrl_steps = 5

		self.cost_trans = 0.0
		self.cost_orient = 0.0
		self.cost_control = 0.0
		self.cost_terminal = 0.0
		self.cost_total = 0.0
		self.costmap_cost = 0.0
		self.last_control = [0,0,0]

		self.update_x = 0.0
		self.update_y = 0.0
		self.update_yaw = 0.0

		self.w_trans = 0.0
		self.w_orient = 0.0
		self.w_control = 0.0
		self.w_terminal = 0.0
		self.size_x_ = 0

		self.map_resolution = 0.0

		self.bnds  = list()
		self.cons = []
		b_x_vel = (-0.7, 0.7)
		b_y_vel = (-0.7, 0.7)
		b_rot = (-0.7, 0.7)
		for i in range(self.no_ctrl_steps):
			self.bnds.append(b_x_vel)
			self.bnds.append(b_y_vel)
			self.bnds.append(b_rot)
			self.cons.append({'type': 'ineq', 'fun': partial(self.f_constraint, index = i)})

		self.initial_guess = np.zeros(self.no_ctrl_steps*3)
		self.local_plan = Path()
		self.prediction_horizon = 5.0
		self.dt  = self.prediction_horizon/self.no_ctrl_steps #time_interval_between_control_pts used in integration

	# Handle costmap information
	def costmap_callback(self, msg):
		self.costmap = msg
		self.map_resolution = msg.info.resolution
		self.grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
		self.size_x_ = msg.info.width

	def get_cost(self, pose):
		# pose is an array of (x,y)
		mx = round((pose[0] - self.costmap.info.origin.position.x)/ self.costmap.info.resolution)
		my = round((pose[1] - self.costmap.info.origin.position.y)/ self.costmap.info.resolution)
		if(abs(mx) > self.costmap.info.height - 1 or abs(my) > self.costmap.info.width - 1):
			return 1.0

		return self.grid[int(mx)][int(my)] / 100.

	def f_constraint(self, initial, index):
		return  0.7 - (np.sqrt((initial[0+index*3])*(initial[0+index*3]) +(initial[1+index*3])*(initial[1+index*3])))   

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

	def initial_guess_update(self, init_guess,guess):
		for i in range(0, self.no_ctrl_steps-1):
			init_guess[0+3*i:3+3*i] = guess[3+3*i:6+3*i]
		init_guess[0+3*(self.no_ctrl_steps-1):3+3*(self.no_ctrl_steps-1)] = guess[0:3]
		return init_guess

	def objective(self, initial_guess):

		self.cost = 0
		self.cost_d = 0
		self.cost_o = 0
		self.cost_r = 0
		self.cost_t = 0
		self.cost_t_d = 0
		self.cost_t_o = 0
		self.cost_d1 = 0
		self.cost_r1 = 0
		self.costmap_cost = 0
		_, _, target_yaw = self.euler_from_quaternion(self.carrot_pose.pose.orientation.x, self.carrot_pose.pose.orientation.y, self.carrot_pose.pose.orientation.z, self.carrot_pose.pose.orientation.w)

		wp12 = [self.carrot_pose.pose.position.x, self.carrot_pose.pose.position.y, target_yaw]

		_, _, final_yaw = self.euler_from_quaternion(self.goal_pose.orientation.x, self.goal_pose.orientation.y, self.goal_pose.orientation.z, self.goal_pose.orientation.w)
		
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0
		cmd_vel = initial_guess
		tot_x = self.current_velocity.linear.x
		tot_y = self.current_velocity.linear.y
		tot_z = self.current_velocity.angular.z
		for i in range((self.no_ctrl_steps)):

			# Predict the velocity

			tot_x = self.dt*(cmd_vel[0+3*i]-tot_x*1.) + tot_x
			tot_y = self.dt*(cmd_vel[1+3*i]-tot_y*1.) + tot_y
			tot_z = self.dt*(cmd_vel[2+3*i]-tot_z*1.) + tot_z

			# Update the position for the predicted velocity
			self.x += tot_x*np.cos(self.z) - tot_y*np.sin(self.z)
			self.y += tot_x*np.sin(self.z) + tot_y*np.cos(self.z)   
			self.z += tot_z * self.dt

			# Step 2: Validate the various self.cost (Using the same idea from Markus Nuernberger's thesis)
			# for i in range((self.no_ctrl_steps)):
			# i) self.cost for error in displacement and orientation
			curr_pos = np.array((wp12[0],wp12[1]))
			pred_pos = np.array((self.x,self.y))

			self.costmap_cost = self.get_cost(pred_pos)
			step_dist_error =  np.linalg.norm(curr_pos - pred_pos)
			step_orient_error = wp12[2] - self.z
			self.cost_d1 = ((self.w_d * step_dist_error**2) + (self.w_o * step_orient_error**2)) / self.no_ctrl_steps            
			curr_pos1 = [self.x,self.y]
			self.cost += self.cost_d1
			curr_vel = np.array((self.current_velocity.linear.x, self.current_velocity.linear.y, \
			self.current_velocity.angular.z ))
			pred_vel = np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]))
			self.cost_r1 = self.w_c * (np.linalg.norm(curr_vel - pred_vel))  / self.no_ctrl_steps          
			self.cost += self.cost_r1
			self.cost +=  self.w_costmap_scale * self.costmap_cost ** 2 / self.no_ctrl_steps
		
		# iii) terminal self.cost

		final_goal = [self.goal_pose.position.x, self.goal_pose.position.y]
		step_dist_error =  np.linalg.norm(curr_pos - final_goal)
		step_orient_error = final_yaw - self.z
		step_dist_error *= 1.0
		self.cost_t = ((self.w_d * step_dist_error**2) + (self.w_o * step_orient_error**2))*self.w_t
		self.cost_t_d = self.w_d * step_dist_error**2
		self.cost_t_o = self.w_o * step_orient_error**2
		self.cost += self.cost_t    
		return self.cost

	def publishLocalPlan(self, x):
		self.local_plan.poses.clear()
		pos_x = self.current_pose.pose.position.x
		pos_y = self.current_pose.pose.position.y

		for i in range((self.no_ctrl_steps)):
			pose = PoseStamped()
			pos_x += x[3*i]*np.cos(0.0) - x[1+3*i]*np.sin(0.0)
			pos_y += x[3*i]*np.sin(0.0) + x[1+3*i]*np.cos(0.0)   
			pose.pose.position.x = pos_x
			pose.pose.position.y = pos_y
			pose.header.stamp = self.get_clock().now().to_msg()
			self.local_plan.poses.append(pose)
	
		self.local_plan.header.stamp = self.get_clock().now().to_msg()
		self.local_plan.header.frame_id = "map"

	def optimizer(self, request, response):

		self.w_d = 0.25
		self.w_o = 0.25
		self.w_c= 0.05
		self.w_t = 0.15
		self.w_costmap_scale = 0.05

		self.current_pose = request.current_pose
		self.carrot_pose = request.carrot_pose
		self.current_velocity = request.current_vel
		self.goal_pose = request.goal_pose

		ig = self.initial_guess
		x = minimize(self.objective, ig,
				method='SLSQP',bounds= self.bnds, constraints = self.cons, options={'ftol':1e-5,'disp':False})		
		self.publishLocalPlan(x.x)
		self.PubRaysPath.publish(self.local_plan)

		for i in range(0,3):
			x.x[i] = x.x[i] * 0.5 + self.last_control[i]*(1 - 0.5)
			self.last_control[i] = x.x[i]

		response.output_vel.twist.linear.x = x.x[0]
		response.output_vel.twist.linear.y = x.x[1]
		response.output_vel.twist.angular.z = x.x[2]
		if (x.success):
			self.initial_guess = self.initial_guess_update(self.initial_guess, x.x)
		else:
			self.initial_guess = x.x
		return response

def main(args=None):
	rclpy.init(args = args)
	MpcOptimization = MpcOptimizationServer()
	rclpy.spin(MpcOptimization)

if __name__ == '__main__':
	main()