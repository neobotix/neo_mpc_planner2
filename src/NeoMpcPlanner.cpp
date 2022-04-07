/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Neobotix GmbH
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Neobotix nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "../include/NeoMpcPlanner.h"

#include <tf2/utils.h>
#include "nav2_util/node_utils.hpp"
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include "nav2_util/line_iterator.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <algorithm>
#include <tf2_eigen/tf2_eigen.hpp>
#include <chrono>

using std::hypot;
using std::min;
using std::max;
using std::abs;
using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;
using namespace nav2_costmap_2d;  // NOLINT
using rcl_interfaces::msg::ParameterType;
using namespace std::chrono_literals;


namespace neo_mpc_planner {

nav_msgs::msg::Path NeoMpcPlanner::transformGlobalPlan(
  const geometry_msgs::msg::PoseStamped & pose)
{
  if (global_plan_.poses.empty()) {
    throw nav2_core::PlannerException("Received plan with zero length");
  }

  // let's get the pose of the robot in the frame of the plan
  geometry_msgs::msg::PoseStamped robot_pose;
  if (!transformPose(global_plan_.header.frame_id, pose, robot_pose)) {
    throw nav2_core::PlannerException("Unable to transform robot pose into global plan's frame");
  }

  // We'll discard points on the plan that are outside the local costmap
  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  const double max_costmap_dim = std::max(costmap->getSizeInCellsX(), costmap->getSizeInCellsY());
  const double max_transform_dist = max_costmap_dim * costmap->getResolution() / 2.0;

  // First find the closest pose on the path to the robot
  auto transformation_begin =
    nav2_util::geometry_utils::min_by(
    global_plan_.poses.begin(), global_plan_.poses.end(),
    [&robot_pose](const geometry_msgs::msg::PoseStamped & ps) {
      return euclidean_distance(robot_pose, ps);
    });

  // Find points definitely outside of the costmap so we won't transform them.
  auto transformation_end = std::find_if(
    transformation_begin, end(global_plan_.poses),
    [&](const auto & global_plan_pose) {
      return euclidean_distance(robot_pose, global_plan_pose) > max_transform_dist;
    });

  // Lambda to transform a PoseStamped from global frame to local
  auto transformGlobalPoseToLocal = [&](const auto & global_plan_pose) {
      geometry_msgs::msg::PoseStamped stamped_pose, transformed_pose;
      stamped_pose.header.frame_id = global_plan_.header.frame_id;
      stamped_pose.header.stamp = robot_pose.header.stamp;
      stamped_pose.pose = global_plan_pose.pose;
      transformPose(costmap_ros_->getBaseFrameID(), stamped_pose, transformed_pose);
      return transformed_pose;
    };

  // Transform the near part of the global plan into the robot's frame of reference.
  nav_msgs::msg::Path transformed_plan;
  std::transform(
    transformation_begin, transformation_end,
    std::back_inserter(transformed_plan.poses),
    transformGlobalPoseToLocal);
  transformed_plan.header.frame_id = costmap_ros_->getBaseFrameID();
  transformed_plan.header.stamp = robot_pose.header.stamp;

  global_plan_.poses.erase(begin(global_plan_.poses), transformation_begin);
  global_path_pub_->publish(transformed_plan);

  if (transformed_plan.poses.empty()) {
    throw nav2_core::PlannerException("Resulting plan has 0 poses in it.");
  }

  return transformed_plan;
}

bool NeoMpcPlanner::transformPose(
  const std::string frame,
  const geometry_msgs::msg::PoseStamped & in_pose,
  geometry_msgs::msg::PoseStamped & out_pose) const
{
  if (in_pose.header.frame_id == frame) {
    out_pose = in_pose;
    return true;
  }

  try {
    tf_->transform(in_pose, out_pose, frame, transform_tolerance_);
    out_pose.header.frame_id = frame;
    return true;
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(logger_, "Exception in transformPose: %s", ex.what());
  }
  return false;
}

double NeoMpcPlanner::getLookAheadDistance(const geometry_msgs::msg::Twist & speed)
{
  // If using velocity-scaled look ahead distances, find and clamp the dist
  // Else, use the static look ahead distance
  double lookahead_dist = 0.4;

  // if ( abs(speed.linear.x) > 0.0 && abs(speed.linear.y) > 0.0) {
  // 	lookahead_dist = 0.4;
  // }
 
  return lookahead_dist;
}

geometry_msgs::msg::PoseStamped NeoMpcPlanner::getLookAheadPoint(
  const double & lookahead_dist,
  const nav_msgs::msg::Path & transformed_plan)
{
  // Find the first pose which is at a distance greater than the lookahead distance
  auto goal_pose_it = std::find_if(
    transformed_plan.poses.begin(), transformed_plan.poses.end(), [&](const auto & ps) {
      return hypot(ps.pose.position.x, ps.pose.position.y) >= lookahead_dist;
    });

  // If the no pose is not far enough, take the last pose
  if (goal_pose_it == transformed_plan.poses.end()) {
    goal_pose_it = std::prev(transformed_plan.poses.end());
    is_last_point = true;
  }

  return *goal_pose_it;
}

std::unique_ptr<geometry_msgs::msg::PointStamped> NeoMpcPlanner::createCarrotMsg(
  const geometry_msgs::msg::PoseStamped & carrot_pose)
{
  auto carrot_msg = std::make_unique<geometry_msgs::msg::PointStamped>();
  carrot_msg->header = carrot_pose.header;
  carrot_msg->point.x = carrot_pose.pose.position.x;
  carrot_msg->point.y = carrot_pose.pose.position.y;
  carrot_msg->point.z = 0.01;  // publish right over map to stand out
  return carrot_msg;
}

geometry_msgs::msg::TwistStamped NeoMpcPlanner::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & position,
  const geometry_msgs::msg::Twist & speed,
  nav2_core::GoalChecker * goal_checker)
{
	auto transformed_plan = transformGlobalPlan(position);

  // Find look ahead distance and point on path and publish
  double lookahead_dist = getLookAheadDistance(speed);

	// For now just for testing
  auto carrot_pose = getLookAheadPoint(lookahead_dist, transformed_plan);

  carrot_pub_->publish(createCarrotMsg(carrot_pose));

  auto request = std::make_shared<neo_srvs2::srv::Optimizer::Request>();
  request->current_vel = speed;
  request->carrot_pose = carrot_pose;
  request->goal_pose = goal_pose;
  request->current_pose = position;

  auto result = client->async_send_request(request);

	auto out = result.get();
	geometry_msgs::msg::TwistStamped cmd_vel_final;
		cmd_vel_final = out->output_vel; 
	
	const rclcpp::Time time_now = rclcpp::Clock().now();
	const double dt = (time_now - m_last_time).seconds();
	m_last_time = time_now;
	cmd_vel_final = out->output_vel; 

	// if (euclidean_distance(position.pose, goal_pose) > 0.4 ) {
	// 	cmd_vel_final = out->output_vel; 
	// }

	// else {
	// 	if (fabs(out->output_vel.twist.linear.x) > 0.0) {
	// 		out->output_vel.twist.linear.x = (out->output_vel.twist.linear.x > 0.0 ? 1.0 : -1.0) * fmin(0.7, fabs(m_last_cmd_vel.twist.linear.x) - 0.25 * dt);	
	// 	}
	// 	if (fabs(out->output_vel.twist.linear.y) > 0.0) {
	// 		out->output_vel.twist.linear.y = (out->output_vel.twist.linear.y > 0.0 ? 1.0 : -1.0) * fmin(0.7, fabs(m_last_cmd_vel.twist.linear.y) - 0.25 * dt);
	// 	}
		
	// 	// out->output_vel.twist.angular.z = (out->output_vel.twist.angular.z > 0.0 ? 1.0 : -1.0) * fmin(fabs(out->output_vel.twist.angular.z), fabs(m_last_cmd_vel.twist.angular.z) - 0.25 * 0.03);
	// 	cmd_vel_final = out->output_vel; 
	// }
	m_last_cmd_vel = cmd_vel_final;
  return cmd_vel_final;
}

void NeoMpcPlanner::cleanup()
{
}

void NeoMpcPlanner::activate()
{
	  global_path_pub_->on_activate();
	  carrot_pub_->on_activate();
}

void NeoMpcPlanner::deactivate()
{
}

void NeoMpcPlanner::setPlan(const nav_msgs::msg::Path & plan)
{
	is_last_point = false;
  global_plan_ = plan;
  goal_pose = plan.poses[plan.poses.size() - 1].pose;
}

void NeoMpcPlanner::setSpeedLimit(
  const double & speed_limit,
  const bool & percentage)
{
  
}

void NeoMpcPlanner::configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,  std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
	auto node = parent.lock();
  node_ = parent;
  if (!node) {
    throw nav2_core::PlannerException("Unable to lock node!");
  }

  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  client = node->create_client<neo_srvs2::srv::Optimizer>("optimizer");
  global_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);

  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
    }
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
  }
  
  carrot_pub_ = node->create_publisher<geometry_msgs::msg::PointStamped>("/lookahead_point", 1);
}

} // neo_mpc_planner

PLUGINLIB_EXPORT_CLASS(neo_mpc_planner::NeoMpcPlanner, nav2_core::Controller)
