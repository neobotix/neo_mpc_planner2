/*********************************************************************
MIT License

Copyright (c) 2022 neobotix gmbh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 *********************************************************************/

#include "../include/NeoMpcPlanner.h"

#include <tf2/utils.h>
#include "nav2_util/node_utils.hpp"
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include "nav2_util/line_iterator.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
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
using std::placeholders::_1;

namespace neo_mpc_planner {

double createYawFromQuat(const geometry_msgs::msg::Quaternion & orientation)
{
  tf2::Quaternion q(orientation.x, orientation.y, orientation.z, orientation.w);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return yaw;
}


nav_msgs::msg::Path NeoMpcPlanner::transformGlobalPlan(
  const geometry_msgs::msg::PoseStamped & pose)
{
  if (global_plan_.poses.empty()) {
    throw nav2_core::ControllerException("Received plan with zero length");
  }

  // let's get the pose of the robot in the frame of the plan
  geometry_msgs::msg::PoseStamped robot_pose;
  if (!transformPose(global_plan_.header.frame_id, pose, robot_pose)) {
    throw nav2_core::ControllerException("Unable to transform robot pose into global plan's frame");
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

  geometry_msgs::msg::PoseStamped final_pose;
  final_pose.header = global_plan_.header;
  final_pose.pose = global_plan_.poses[global_plan_.poses.size() - 1].pose;
  if (euclidean_distance(robot_pose, final_pose) <= lookahead_dist_close_to_goal_) {
    closer_to_goal = true;
  }
  else {
    closer_to_goal = false;
  }
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
      if (!transformPose(costmap_ros_->getBaseFrameID(), stamped_pose, transformed_pose)) {
        throw nav2_core::ControllerTFError(
                "Unable to transform global plan pose into robot base frame");
      }
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
    throw nav2_core::ControllerException("Resulting plan has 0 poses in it.");
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

bool NeoMpcPlanner::isPoseInCollision(
  const double x, const double y, const double yaw,
  const nav2_costmap_2d::Footprint & footprint)
{
  unsigned int mx = 0;
  unsigned int my = 0;
  if (!costmap_->worldToMap(x, y, mx, my)) {
    return true;
  }

  double center_cost = costmap_->getCost(mx, my);
  if (center_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
    return true;
  }

  double footprint_cost = collision_checker_->footprintCostAtPose(
    x, y, yaw, footprint);
  if (!std::isfinite(footprint_cost) || footprint_cost < 0.0 ||
    footprint_cost >= nav2_costmap_2d::LETHAL_OBSTACLE)
  {
    return true;
  }

  return false;
}

bool NeoMpcPlanner::isCollisionImminent(
  const geometry_msgs::msg::PoseStamped & current_pose,
  const geometry_msgs::msg::TwistStamped & command,
  double & collision_time)
{
  collision_time = 0.0;
  const auto footprint = costmap_ros_->getRobotFootprint();
  if (footprint.empty()) {
    RCLCPP_ERROR_THROTTLE(
      logger_, *clock_, 1000,
      "Predicted footprint collision check has no configured footprint");
    return true;
  }

  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> costmap_lock(
    *costmap_->getMutex());

  double pose_x = current_pose.pose.position.x;
  double pose_y = current_pose.pose.position.y;
  double pose_yaw = tf2::getYaw(current_pose.pose.orientation);
  if (isPoseInCollision(pose_x, pose_y, pose_yaw, footprint)) {
    return true;
  }

  const double vx = command.twist.linear.x;
  const double vy = command.twist.linear.y;
  const double omega = command.twist.angular.z;

  double footprint_radius = 0.0;
  for (const auto & point : footprint) {
    footprint_radius = max(footprint_radius, hypot(point.x, point.y));
  }

  double boundary_speed = hypot(vx, vy) + fabs(omega) * footprint_radius;
  if (boundary_speed <= 1e-6 || collision_prediction_time_ <= 0.0) {
    return false;
  }

  double controller_period =
    control_frequency > 0.0 ? 1.0 / control_frequency : collision_prediction_time_;
  double spatial_period = costmap_->getResolution() / boundary_speed;
  double sample_period = std::clamp(
    min(controller_period, spatial_period), 1e-3, collision_prediction_time_);

  double elapsed = 0.0;
  while (elapsed < collision_prediction_time_) {
    double dt = min(sample_period, collision_prediction_time_ - elapsed);
    double midpoint_yaw = pose_yaw + 0.5 * omega * dt;
    pose_x += (vx * cos(midpoint_yaw) - vy * sin(midpoint_yaw)) * dt;
    pose_y += (vx * sin(midpoint_yaw) + vy * cos(midpoint_yaw)) * dt;
    pose_yaw = angles::normalize_angle(pose_yaw + omega * dt);
    elapsed += dt;

    if (isPoseInCollision(pose_x, pose_y, pose_yaw, footprint)) {
      collision_time = elapsed;
      return true;
    }
  }

  return false;
}

double NeoMpcPlanner::getLookAheadDistance(const geometry_msgs::msg::Twist & speed)
{
  // If using velocity-scaled look ahead distances, find and clamp the dist
  // Else, use the static look ahead distance
  double lookahead_dist = lookahead_dist_min_;

  if (!slow_down_ || closer_to_goal)
  {
    lookahead_dist = lookahead_dist_max_;
    if (closer_to_goal) {
      lookahead_dist = lookahead_dist_close_to_goal_;
    }
  }
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
  std::lock_guard<std::mutex> lock_reinit(mutex_);
  auto transformed_plan = transformGlobalPlan(position);

  // Find look ahead distance and point on path and publish
  double lookahead_dist = getLookAheadDistance(speed);

  // For now just for testing
  auto carrot_pose = getLookAheadPoint(lookahead_dist, transformed_plan);

  // carrot_pose2 for determining the turn
  auto carrot_pose2 = getLookAheadPoint(lookahead_dist2, transformed_plan);

  // check if the pose orientation and robot orientation greater than 180
  // change the lookahead accordingly
  double footprint_cost = collision_checker_->footprintCostAtPose(
    position.pose.position.x, position.pose.position.y, tf2::getYaw(position.pose.orientation), costmap_ros_->getRobotFootprint());

  if (fabs(createYawFromQuat(carrot_pose.pose.orientation)) < 1.0) {
    slow_down_ = false;
    // Check if the robot can speed up once again, so that there is no oscillations between the lookaheads
    auto check_pose_up = getLookAheadPoint(lookahead_dist, transformed_plan);
    if (fabs(createYawFromQuat(check_pose_up.pose.orientation)) > 1.0 and footprint_cost > 200) {
      slow_down_ = true;
    }
  } else if (fabs(createYawFromQuat(carrot_pose.pose.orientation)) >= 1.0 and footprint_cost > 200) {
    slow_down_ = true;
  } else {
    slow_down_ = false;
  }

  double target_yaw = std::atan2(carrot_pose.pose.position.y, carrot_pose.pose.position.x); // The desired orientation
  double target_yaw_l1 = std::atan2(carrot_pose2.pose.position.y, carrot_pose2.pose.position.x); // Orientation for farther lookahead

  if (closer_to_goal) {
    auto goal_orientation = transformed_plan.poses.back().pose.orientation;
    target_yaw = createYawFromQuat(goal_orientation);
  }

  // Calculate effective turn angle for Python to use
  // Use actual velocity to determine if robot is driving backward
  bool driving_backward = (speed.linear.x < -0.05);  // Threshold to avoid noise
  
  double effective_turn_angle;
  if (driving_backward) {
    // If driving backward, check turn angle relative to backward direction (±180°)
    double backward_angle = std::abs(target_yaw) - M_PI;
    effective_turn_angle = std::abs(backward_angle);
  } else {
    // If driving forward, use target_yaw directly
    effective_turn_angle = std::abs(target_yaw);
  }
  
  // Always calculate tight lookahead point for Python to choose from
  auto carrot_pose_tight = getLookAheadPoint(tight_lookahead_dist, transformed_plan);
  double target_yaw_tight = std::atan2(carrot_pose_tight.pose.position.y, carrot_pose_tight.pose.position.x);

  if (closer_to_goal) {
    auto goal_orientation = transformed_plan.poses.back().pose.orientation;
    target_yaw_tight = createYawFromQuat(goal_orientation);
  }

  if (!std::isfinite(footprint_cost) || footprint_cost < 0.0 ||
    footprint_cost >= nav2_costmap_2d::LETHAL_OBSTACLE)
  {
    throw nav2_core::ControllerException("MPC detected collision!");
  }

  carrot_pub_->publish(createCarrotMsg(carrot_pose));
  carrot_pub2_->publish(createCarrotMsg(carrot_pose2));

  auto request = std::make_shared<neo_srvs2::srv::Optimizer::Request>();
  request->current_vel = speed;
  request->carrot_pose = carrot_pose;
  request->carrot_pose_tight = carrot_pose_tight;
  request->carrot_pose_terminal = carrot_pose2;  // Use far lookahead for terminal cost
  request->turn_yaw = target_yaw;
  request->turn_yaw_tight = target_yaw_tight;
  request->effective_turn_angle = effective_turn_angle;
  request->goal_pose = goal_pose;
  request->current_pose = position;
  request->switch_opt = closer_to_goal;
  request->control_interval = 1.0 / control_frequency;

  // Use nav2::ServiceClient API
  auto result = client->async_send_request(request);
  auto out = result.get();
  geometry_msgs::msg::TwistStamped cmd_vel_final;
  cmd_vel_final = out->output_vel;

  if (use_predicted_footprint_collision_check_) {
    double collision_time = 0.0;
    if (isCollisionImminent(position, cmd_vel_final, collision_time)) {
      RCLCPP_WARN_THROTTLE(
        logger_, *clock_, 1000,
        "Predicted footprint collision in %.3f s; commanding zero velocity",
        collision_time);
      cmd_vel_final.twist.linear.x = 0.0;
      cmd_vel_final.twist.linear.y = 0.0;
      cmd_vel_final.twist.angular.z = 0.0;
    }
  }

  return cmd_vel_final;
}

void NeoMpcPlanner::cleanup()
{
}

void NeoMpcPlanner::activate()
{
  global_path_pub_->on_activate();
  carrot_pub_->on_activate();
  carrot_pub2_->on_activate();
  auto node = node_.lock();
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(&NeoMpcPlanner::dynamicParametersCallback, this, std::placeholders::_1));
}

void NeoMpcPlanner::deactivate()
{
}

void NeoMpcPlanner::setPlan(const nav_msgs::msg::Path & plan)
{
  global_plan_ = plan;
  if (goal_pose != plan.poses[plan.poses.size() - 1].pose) {
    slow_down_ = true;
  }
  goal_pose = plan.poses[plan.poses.size() - 1].pose;
}

void NeoMpcPlanner::setSpeedLimit(
  const double & speed_limit,
  const bool & percentage)
{

}

void NeoMpcPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  
  if (!node) {
    throw nav2_core::ControllerException("Unable to lock node!");
  }

  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  client = node->create_client<neo_srvs2::srv::Optimizer>("optimizer");
  global_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);

  declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_dist_min", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_dist_max", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_dist_close_to_goal", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".tight_lookahead_dist", rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_predicted_footprint_collision_check",
    rclcpp::ParameterValue(true));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".collision_prediction_time", rclcpp::ParameterValue(0.3));

  node->get_parameter(plugin_name_ + ".lookahead_dist_min", lookahead_dist_min_);
  node->get_parameter(plugin_name_ + ".lookahead_dist_max", lookahead_dist_max_);
  node->get_parameter(
    plugin_name_ + ".lookahead_dist_close_to_goal",
    lookahead_dist_close_to_goal_);
  node->get_parameter(plugin_name_ + ".tight_lookahead_dist", tight_lookahead_dist);
  node->get_parameter(
    plugin_name_ + ".use_predicted_footprint_collision_check",
    use_predicted_footprint_collision_check_);
  node->get_parameter(
    plugin_name_ + ".collision_prediction_time", collision_prediction_time_);
  node->get_parameter("controller_frequency", control_frequency);

  if (collision_prediction_time_ < 0.0) {
    throw nav2_core::ControllerException(
            "collision_prediction_time must be greater than or equal to zero");
  }
  
  // Validation: lookahead_dist_close_to_goal should be less than tight_lookahead_dist
  // if (lookahead_dist_close_to_goal_ >= tight_lookahead_dist) {
  //   RCLCPP_WARN(
  //     logger_,
  //     "lookahead_dist_close_to_goal (%.2f) should be less than or equal to tight_lookahead_dist (%.2f)",
  //     lookahead_dist_close_to_goal_, tight_lookahead_dist);
  //   lookahead_dist_close_to_goal_ = tight_lookahead_dist; // Set to default value
  // }

  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
    }
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
  }
  carrot_pub_ = node->create_publisher<geometry_msgs::msg::PointStamped>("/lookahead_point", 1);
  carrot_pub2_ = node->create_publisher<geometry_msgs::msg::PointStamped>("/lookahead_point2", 1);
  collision_checker_ = std::make_unique<nav2_costmap_2d::
      FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>(costmap_);
}

rcl_interfaces::msg::SetParametersResult
NeoMpcPlanner::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  std::lock_guard<std::mutex> lock_reinit(mutex_);
  rcl_interfaces::msg::SetParametersResult result;

  for (auto parameter : parameters) {
    const auto & type = parameter.get_type();
    const auto & name = parameter.get_name();

    // If we are trying to change the parameter of a plugin we can just skip it at this point
    // as they handle parameter changes themselves and don't need to lock the mutex
    if (name.find('.') != std::string::npos) {
      continue;
    }

    if (!mutex_.try_lock()) {
      RCLCPP_WARN(
        logger_,
        "Unable to dynamically change Parameters while the controller is currently running");
      result.successful = false;
      result.reason =
        "Unable to dynamically change Parameters while the controller is currently running";
      return result;
    }

    if (type == ParameterType::PARAMETER_DOUBLE) {
      if (name == plugin_name_ + "lookahead_dist_min") {
        lookahead_dist_min_ = parameter.as_double();
      } else if (name == plugin_name_ + "lookahead_dist_max") {
        lookahead_dist_max_ = parameter.as_double();
      } else if (name == plugin_name_ + "lookahead_dist_close_to_goal") {
        lookahead_dist_close_to_goal_ = parameter.as_double();
      } else if (name == plugin_name_ + "tight_lookahead_dist") {
        tight_lookahead_dist = parameter.as_double();
      } 
    }
    mutex_.unlock();
  }

  result.successful = true;
  return result;
}

} // neo_mpc_planner

PLUGINLIB_EXPORT_CLASS(neo_mpc_planner::NeoMpcPlanner, nav2_core::Controller)
