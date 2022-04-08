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
from neo_nav2_py_costmap2D.line_iterator import LineIterator
from neo_nav2_py_costmap2D.costmap import Costmap2d

class MpcOptimizationServer(Node):
	def __init__(self):
		super().__init__('mpc_optimization_server')
		self.srv = self.create_service(Optimizer, 'optimizer', self.optimizer)
		# self.subscription = self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 10)
		self.PubRaysPath = self.create_publisher(Path, 'local_plan', 10)
		self.current_pose = Pose()
		self.carrot_pose = PoseStamped()
		self.goal_pose = PoseStamped()
		self.current_velocity = TwistStamped()
		self.local_plan = Path()
		self.no_ctrl_steps = 3
		self.low_pass_gain = 0.5
		self.prediction_horizon = 3.0

		self.cost_total = 0.0
		self.costmap_cost = 0.0
		self.last_control = [0,0,0]
		self.c = Costmap2d(self)

		self.count = 0

		self.update_x = 0.0
		self.update_y = 0.0
		self.update_yaw = 0.0

		self.w_trans = 0.0
		self.w_orient = 0.0
		self.w_control = 0.0
		self.w_terminal = 0.0
		self.size_x_ = 0

		self.bnds  = list()
		self.bnds1  = list()
		self.cons = []
		self.cons1 = []
		b_x_vel = (-0.07, 0.07)
		b_y_vel = (-0.07, 0.07)
		b_rot = (-0.7, 0.7)
		for i in range(self.no_ctrl_steps):
			self.bnds.append(b_x_vel)
			self.bnds.append(b_y_vel)
			self.bnds.append(b_rot)
			self.cons.append({'type': 'ineq', 'fun': partial(self.f_constraint, index = i)})

		b_x_vel = (-0.7, 0.7)
		b_y_vel = (-0.7, 0.7)
		b_rot = (-0.7, 0.7)

		for i in range(self.no_ctrl_steps):
			self.bnds1.append(b_x_vel)
			self.bnds1.append(b_y_vel)
			self.bnds1.append(b_rot)
			self.cons1.append({'type': 'ineq', 'fun': partial(self.f_constraint1, index = i)})
			
		self.initial_guess = np.zeros(self.no_ctrl_steps * 3)
		self.dt  = self.prediction_horizon /self.no_ctrl_steps #time_interval_between_control_pts used in integration
		self.last_time = 0.0
		self.update_opt_param = False

	def f_constraint(self, initial, index):
		return  0.07 - (np.sqrt((initial[0 + index * 3]) * (initial[0 + index * 3]) +(initial[1 + index * 3]) * (initial[1 + index * 3])))   

	def f_constraint1(self, initial, index):
		return  0.7 - (np.sqrt((initial[0 + index * 3]) * (initial[0 + index * 3]) +(initial[1 + index * 3]) * (initial[1 + index * 3]))) 

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

	def objective1(self, cmd_vel):
		self.cost_total= 0
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

		_, _, target_yaw = self.euler_from_quaternion(self.carrot_pose.pose.orientation.x, self.carrot_pose.pose.orientation.y, self.carrot_pose.pose.orientation.z, self.carrot_pose.pose.orientation.w)
		_, _, final_yaw = self.euler_from_quaternion(self.goal_pose.orientation.x, self.goal_pose.orientation.y, self.goal_pose.orientation.z, self.goal_pose.orientation.w)
		
		count = 1.0
		tot_x = self.current_velocity.linear.x
		tot_y = self.current_velocity.linear.y
		tot_z = self.current_velocity.angular.z
		curr_pos = np.array((self.carrot_pose.pose.position.x,self.carrot_pose.pose.position.y))
		last_pos = curr_pos

		for i in range((self.no_ctrl_steps)):
			self.costmap_cost = 0
			# Predict the velocity

			tot_x = self.dt*(cmd_vel[0+3*i] - tot_x) + tot_x
			tot_y = self.dt*(cmd_vel[1+3*i] - tot_y) + tot_y
			tot_z = self.dt*(cmd_vel[2+3*i] - tot_z) + tot_z

			# Update the position for the predicted velocity
			self.x += tot_x*np.cos(self.z) - tot_y*np.sin(self.z)
			self.y += tot_x*np.sin(self.z) + tot_y*np.cos(self.z)   
			self.z += tot_z * self.dt

			# Step 2: Validate the various self.cost_total(Using the same idea from Markus Nuernberger's thesis)
			# for i in range((self.no_ctrl_steps)):
			# i) self.cost_totalfor error in displacement and orientation
			
			mx0, my0 = self.c.getWorldToMap(last_pos[0], last_pos[1])
			mx1, my1 = self.c.getWorldToMap(self.x, self.y)
			line = LineIterator(mx0, my0, mx1, my1)
			
			c_c = 0.0
			while(line.isValid() and count<=10):
				line.advance()
				lin_pos = np.array((line.getX(), line.getY()))
				if(line.isValid()):
					c_c += self.c.getCost(line.getX(), line.getY())
					count = count + 1
					self.costmap_cost += c_c

			#  Just take the average cost
			self.costmap_cost = self.costmap_cost / count

			last_pos = np.array((self.x, self.y))

			step_dist_error =  np.linalg.norm(curr_pos - np.array((self.x, self.y)))
			step_orient_error = target_yaw - self.z
			self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient * step_orient_error**2)) / self.no_ctrl_steps            
			self.cost_total += self.w_control * (np.linalg.norm(np.array((self.current_velocity.linear.x, self.current_velocity.linear.y, \
			self.current_velocity.angular.z)) - np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]))))  / self.no_ctrl_steps          
			self.cost_total +=  self.w_costmap_scale * self.costmap_cost ** 2 / self.no_ctrl_steps
		
		# iii) terminal self.cost

		step_dist_error =  np.linalg.norm(curr_pos - np.array((self.goal_pose.position.x,self.goal_pose.position.y)))
		step_orient_error = final_yaw - self.z
		self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient* step_orient_error**2))*self.w_terminal  
		return self.cost_total

	def objective(self, cmd_vel):

		self.cost_total= 0
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

		_, _, target_yaw = self.euler_from_quaternion(self.carrot_pose.pose.orientation.x, self.carrot_pose.pose.orientation.y, self.carrot_pose.pose.orientation.z, self.carrot_pose.pose.orientation.w)
		_, _, final_yaw = self.euler_from_quaternion(self.goal_pose.orientation.x, self.goal_pose.orientation.y, self.goal_pose.orientation.z, self.goal_pose.orientation.w)
		
		count = 1.0
		tot_x = self.current_velocity.linear.x / 10.0
		tot_y = self.current_velocity.linear.y / 10.0
		tot_z = self.current_velocity.angular.z / 10.0
		curr_pos = np.array((self.carrot_pose.pose.position.x,self.carrot_pose.pose.position.y))
		last_pos = curr_pos

		for i in range((self.no_ctrl_steps)):
			self.costmap_cost = 0
			# Predict the velocity

			tot_x = self.dt*(cmd_vel[0+3*i] - tot_x) + tot_x
			tot_y = self.dt*(cmd_vel[1+3*i] - tot_y) + tot_y
			tot_z = self.dt*(cmd_vel[2+3*i] / 10.0 - tot_z) + tot_z

			# Update the position for the predicted velocity
			self.x += tot_x*np.cos(self.z) - tot_y*np.sin(self.z)
			self.y += tot_x*np.sin(self.z) + tot_y*np.cos(self.z)   
			self.z += tot_z * self.dt

			# Step 2: Validate the various self.cost_total(Using the same idea from Markus Nuernberger's thesis)
			# for i in range((self.no_ctrl_steps)):
			# i) self.cost_totalfor error in displacement and orientation
			
			mx0, my0 = self.c.getWorldToMap(last_pos[0], last_pos[1])
			mx1, my1 = self.c.getWorldToMap(self.x, self.y)
			line = LineIterator(mx0, my0, mx1, my1)
			
			c_c = 0.0
			while(line.isValid() and count<=10):
				line.advance()
				lin_pos = np.array((line.getX(), line.getY()))
				if(line.isValid()):
					c_c += self.c.getCost(line.getX(), line.getY())
					count = count + 1
					self.costmap_cost += c_c

			#  Just take the average cost
			self.costmap_cost = self.costmap_cost / count

			last_pos = np.array((self.x, self.y))

			step_dist_error =  np.linalg.norm(curr_pos - np.array((self.x, self.y)))
			step_orient_error = target_yaw - self.z
			self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient * step_orient_error**2)) / self.no_ctrl_steps            
			self.cost_total += self.w_control * (np.linalg.norm(np.array((self.current_velocity.linear.x / 10.0, self.current_velocity.linear.y/10.0, \
			self.current_velocity.angular.z/10.0 )) - np.array((cmd_vel[0+3*i], cmd_vel[1+3*i], cmd_vel[2+3*i]/10.0))))  / self.no_ctrl_steps          
			self.cost_total +=  self.w_costmap_scale * self.costmap_cost ** 2 / self.no_ctrl_steps
		
		# iii) terminal self.cost
		step_dist_error =  np.linalg.norm(curr_pos - np.array((self.goal_pose.position.x,self.goal_pose.position.y)))
		step_orient_error = final_yaw - self.z
		self.cost_total += ((self.w_trans * step_dist_error**2) + (self.w_orient* step_orient_error**2))*self.w_terminal  
		return self.cost_total

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

		self.w_trans = 0.25
		self.w_orient = 0.35
		self.w_control= 0.05
		self.w_terminal = 0.01
		self.w_costmap_scale = 0.05

		self.current_pose = request.current_pose
		self.carrot_pose = request.carrot_pose
		self.current_velocity = request.current_vel
		self.goal_pose = request.goal_pose

		# on new goal reset all the flags and initializers
		if (self.update_opt_param == True and (self.old_goal != self.goal_pose)):
			self.update_opt_param = False
			self.initial_guess = np.zeros(self.no_ctrl_steps*3)
			self.last_control = [0,0,0]
			self.count = 0

		if ((np.linalg.norm(np.array((self.current_pose.pose.position.x, self.current_pose.pose.position.y)) - np.array((self.goal_pose.position.x, self.goal_pose.position.y)))) <= 1.0):
			self.update_opt_param = True

		if (self.update_opt_param == False): 
			x = minimize(self.objective, self.initial_guess,
					method='SLSQP',bounds= self.bnds, constraints = self.cons, options={'ftol':1e-5,'disp':False})		
			self.publishLocalPlan(x.x)
			self.PubRaysPath.publish(self.local_plan)

			for i in range(0,3):
				x.x[i] = x.x[i] * self.low_pass_gain + self.last_control[i] * (1 - self.low_pass_gain)

			current_time = time.time()
			delta_t = current_time - self.last_time
			self.last_time = current_time

			# avoiding sudden jerks 
			response.output_vel.twist.linear.x = np.sign(x.x[0]) * np.fmin(abs(x.x[0]), abs(self.last_control[0]) + 0.025 * delta_t) * 10.0
			response.output_vel.twist.linear.y = np.sign(x.x[1]) * np.fmin(abs(x.x[1]), abs(self.last_control[1])+ 0.025 * delta_t) * 10.0
			response.output_vel.twist.angular.z = np.sign(x.x[2]) * np.fmin(abs(x.x[2]), abs(self.last_control[2]) + 0.25 * delta_t)

			self.last_control[0] = response.output_vel.twist.linear.x / 10.0
			self.last_control[1] = response.output_vel.twist.linear.y / 10.0
			self.last_control[2] = response.output_vel.twist.angular.z

			if (x.success):
				self.initial_guess = self.initial_guess_update(self.initial_guess, x.x)
			else:
				self.initial_guess = x.x

		# gracisouly slowing down and reaching the goal
		else:
			if (self.count == 0):
				self.last_control[0] = self.last_control[0] * 10.0
				self.last_control[1] = self.last_control[1] * 10.0
				self.initial_guess[0] = self.initial_guess[0] * 10.0
				self.initial_guess[1] = self.initial_guess[1] * 10.0
				self.count = 1

			x = minimize(self.objective1, self.initial_guess,
					method='SLSQP',bounds= self.bnds1,constraints = self.cons1, options={'ftol':1e-5,'disp':False})		

			for i in range(0,3):
				x.x[i] = x.x[i] * self.low_pass_gain + self.last_control[i] * (1 - self.low_pass_gain)

			current_time = time.time()
			delta_t = current_time - self.last_time
			self.last_time = current_time

			response.output_vel.twist.linear.x = np.sign(x.x[0]) * np.fmax(abs(x.x[0]), abs(self.last_control[0]) - 0.25 * delta_t)
			response.output_vel.twist.linear.y = np.sign(x.x[1]) * np.fmax(abs(x.x[1]), abs(self.last_control[1]) - 0.25 * delta_t) 
			response.output_vel.twist.angular.z = np.sign(x.x[2]) * np.fmax(abs(x.x[2]), abs(self.last_control[2]) - 0.25 * delta_t)

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