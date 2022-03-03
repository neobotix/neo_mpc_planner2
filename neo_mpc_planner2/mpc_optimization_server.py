#!/usr/bin/env python3
from __future__ import division

import rclpy
from rclpy.node import Node
from neo_srvs2.srv import Optimizer
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
import numpy as np
from scipy.optimize import minimize
import math
from functools import partial
import time
 

class MpcOptimizationServer(Node):

	def __init__(self):
		super().__init__('mpc_optimization_server')
		self.srv = self.create_service(Optimizer, 'optimizer', self.optimizer)
		self.current_pose = Pose()
		self.carrot_pose = PoseStamped()
		self.goal_pose = PoseStamped()
		self.current_velocity = TwistStamped()

		self.no_ctrl_steps = 3

		self.cost_trans = 0.0
		self.cost_orient = 0.0
		self.cost_control = 0.0
		self.cost_terminal = 0.0
		self.cost_total = 0.0

		self.update_x = 0.0
		self.update_y = 0.0
		self.update_yaw = 0.0

		self.w_trans = 0.0
		self.w_orient = 0.0
		self.w_control = 0.0
		self.w_terminal = 0.0

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
		self.prediction_horizon = 3.0
		self.dt  = self.prediction_horizon/self.no_ctrl_steps #time_interval_between_control_pts used in integration

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

	def objective(self, gue):

		self.cost_trans = 0.0
		self.cost_orient = 0.0
		self.cost_control = 0.0
		self.cost_terminal = 0.0
		self.cost_total = 0.0

		self.update_x = self.current_pose.pose.position.x;
		self.update_y = self.current_pose.pose.position.y;
		
		self.update_roll, _, self.update_yaw = self.euler_from_quaternion(self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w)

		_, _, target_yaw = self.euler_from_quaternion(self.carrot_pose.pose.orientation.x, self.carrot_pose.pose.orientation.y, self.carrot_pose.pose.orientation.z, self.carrot_pose.pose.orientation.w)

		_, _, final_yaw = self.euler_from_quaternion(self.goal_pose.orientation.x, self.goal_pose.orientation.y, self.goal_pose.orientation.z, self.goal_pose.orientation.w)

		tot_x = self.current_velocity.linear.x
		tot_y = self.current_velocity.linear.y
		tot_z = self.current_velocity.angular.z

		cmd_vel = gue

		for i in range(0, self.no_ctrl_steps):
			# Predict the velocity
			tot_x = self.dt*(cmd_vel[0+3*i]-tot_x*1.) + tot_x
			tot_y = self.dt*(cmd_vel[0+3*i]-tot_y*1.) + tot_y
			tot_z = self.dt*(cmd_vel[0+3*i]-tot_z*1.) + tot_z

			# Update the position for the predicted velocity
			self.update_x += tot_x*np.cos(self.update_yaw) - tot_y*np.sin(self.update_yaw)
			self.update_y += tot_x*np.sin(self.update_yaw) + tot_y*np.cos(self.update_yaw)   
			self.update_yaw += tot_z * self.dt

			tar_pos = np.array((self.carrot_pose.pose.position.x, self.carrot_pose.pose.position.y))
			pred_pos = np.array((self.update_x,self.update_y))

			# Predicted velocity
			curr_vel = np.array((self.current_velocity.linear.x, self.current_velocity.linear.y, self.current_velocity.angular.z))
			pred_vel = np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]))

			# Cost update
			self.cost_trans = self.w_trans * (np.linalg.norm(tar_pos - pred_pos)**2) / self.no_ctrl_steps
			self.cost_total += self.cost_trans
			self.cost_orient = self.w_orient * (self.update_yaw - 0.0)**2 / self.no_ctrl_steps
			self.cost_total += self.cost_orient
			self.cost_control = self.w_control * (np.linalg.norm(curr_vel - pred_vel)**2)  / self.no_ctrl_steps      
			self.cost_total += self.cost_control

		fin_goal = [self.goal_pose.position.x, self.goal_pose.position.y]
		dist_error = np.linalg.norm(tar_pos - fin_goal)
		orient_error = final_yaw - 0.0
		self.cost_terminal = ((self.w_trans * dist_error**2) + (self.w_orient * orient_error**2))*self.w_terminal
		self.cost_total += self.cost_terminal

		return self.cost_total

	def optimizer(self, request, response):

		self.w_trans = 0.55
		self.w_orient = 0.55
		self.w_control = 0.05
		self.w_terminal = 0.15

		self.current_pose = request.current_pose
		self.carrot_pose = request.carrot_pose
		self.current_velocity = request.current_vel
		self.goal_pose = request.goal_pose

		print("x_bef", self.initial_guess)
		x = minimize(self.objective, self.initial_guess,
				method='SLSQP',bounds= self.bnds, constraints = self.cons, options={'ftol':1e-5,'disp':False})
		
		print("x", x.x)
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