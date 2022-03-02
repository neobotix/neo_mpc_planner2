#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from neo_srvs2.srv import Optimizer
from geometry_msgs.msg import TwistStamped, PoseStamped
import numpy as np
from scipy.optimize import minimize

class MpcOptimizationServer(Node):

	def __init__(self):
		super().__init__('mpc_optimization_server')
		self.srv = self.create_service(Optimizer, 'optimizer', self.optimizer)

		self.no_ctrl_steps = 3

		self.cost_trans = 0.0
		self.cost_orient = 0.0
		self.cost_control = 0.0
		self.cost_terminal = 0.0
		self.cost_total = 0.0

		self.update_x = 0.0
		self.update_y = 0.0
		self.update_yaw = 0.0

		self.w_trans = 0.55
		self.w_orient = 0.55
		self.w_control = 0.05
		self.w_terminal = 0.15

		self.bnds  = list()
		b_x_vel = (-0.7, 0.7)
		b_y_vel = (-0.7, 0.7)
		b_rot = (-0.7, 0.7)
		for i in range(self.no_ctrl_steps):
			self.bnds.append(b_x_vel)
			self.bnds.append(b_y_vel)
			self.bnds.append(b_rot)

		self.initial_guess = np.zeros(self.no_ctrl_steps*3)
		self.prediction_horizon = 3.0
		self.dt  = self.prediction_horizon/self.no_ctrl_steps #time_interval_between_control_pts used in integration


	def initial_guess_update(self, initial_guess,guess,no_ctrl_steps):
		for i in range(0, no_ctrl_steps-1):
			initial_guess[0+3*i:3+3*i] = guess[3+3*i:6+3*i]
		initial_guess[0+3*(no_ctrl_steps-1):3+3*(no_ctrl_steps-1)] = guess[0:3]
		return initial_guess

	def objective(self, initial_guess, target_pose, current_velocity, final_goal):
		tot_x = 0
		tot_y = 0
		tot_z = 0
		for i in range((self.no_ctrl_steps)):
			# Predict the velocity
			tot_x = self.dt*(initial_guess[0+3*i]-tot_x*1.) + tot_x
			tot_y = self.dt*(initial_guess[1+3*i]-tot_y*1.) + tot_y
			tot_z = self.dt*(initial_guess[2+3*i]-tot_z*1.) + tot_z

			# Update the position for the predicted velocity
			self.update_x += tot_x*np.cos(self.update_yaw) - tot_y*np.sin(self.update_yaw)
			self.update_y += tot_x*np.sin(self.update_yaw) + tot_y*np.cos(self.update_yaw)   
			self.update_yaw += 0.0

			tar_pos = np.array((target_pose.pose.position.x, target_pose.pose.position.y))
			pred_pos = np.array((self.update_x,self.update_y))

			# Predicted velocity
			curr_vel = np.array((current_velocity.linear.x, current_velocity.linear.y))
			pred_vel = np.array((initial_guess[0], initial_guess[1]))

			# Cost update
			self.cost_trans = self.w_trans * (np.linalg.norm(tar_pos - pred_pos)**2) / self.no_ctrl_steps
			self.cost_total += self.cost_trans
			self.cost_orient += self.w_orient * (target_pose.pose.orientation.z - self.update_yaw)**2 / self.no_ctrl_steps
			self.cost_total += self.cost_orient
			self.cost_control += self.w_control * (np.linalg.norm(curr_vel - pred_vel))  / self.no_ctrl_steps      
			self.cost_total += self.cost_orient

		fin_goal = [final_goal.position.x, final_goal.position.y]
		dist_error = np.linalg.norm(tar_pos - fin_goal)
		orient_error = final_goal.orientation.z - self.update_yaw
		self.cost_terminal = (self.w_trans * dist_error**2)*self.w_terminal
		self.cost_total += self.cost_terminal
		return self.cost_total

	def optimizer(self, request, response):
		self.cost_total = 0.0
		x = minimize(self.objective, self.initial_guess, args=(request.carrot_pose, request.current_vel, request.goal_pose),
				method='SLSQP',bounds= self.bnds, options={'ftol':1e-5,'disp':False})
		
		response.output_vel.twist.linear.x = x.x[0]
		response.output_vel.twist.linear.y = x.x[1]
		response.output_vel.twist.angular.z = x.x[2]
		print(x)
		self.initial_guess = self.initial_guess_update(self.initial_guess, x.x, self.no_ctrl_steps)
		
		return response

def main(args=None):
	rclpy.init(args = args)
	MpcOptimization = MpcOptimizationServer()
	rclpy.spin(MpcOptimization)

if __name__ == '__main__':
	main()