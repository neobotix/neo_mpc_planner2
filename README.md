# neo_mpc_planner2
<img src="https://user-images.githubusercontent.com/20242192/171641578-05abb9d7-1fa3-4756-a0df-1778a192fb5e.gif" width="700" /> 

The Model Predictive Control (MPC) is an optimization-based feedback control technique, which was initially developed for the field of process industries. With the recent advancements in cost-effective computation, MPC was extended to the field of robotics for real-time control. This means that the optimization happens on the fly. MPC otherwise also called as the Receding Horizon Control (RHC) is utilized in robots for the purpose of path following and / or trajectory tracking.

neo_mpc_planner2 is exclusively developed for quasi-omnidirectional robots such as Neobotix MPO-700. MPO-700 is used for transporting various objects in an structured environment. Therefore this planner aims in catering all the omni-directional robots used under a known environment. The controller was tested on an Intel i5 process and was found that it can easily work at an average rate of 50 Hz. This would help to instatneously react to the suddenly occuring obstacles. Currently, the planner is designed in a way to stop the robot if there is an unknown obstacle in the predicted path. 

The entire planner is built on the Navigation2 stack for ROS-2. Currently, we support omni-directional robots and later will be extended to differential robots. 

## Working

1. With smac_planner (Simulation):

<img src="https://user-images.githubusercontent.com/20242192/171638805-ea465629-5c95-4c5c-8ac0-7cb93e8d2fdd.gif" width="700" /> 

## Dependencies 

Python 3.8.10
Scipy 1.6.3
neo_nav2_py_costmap2D (https://github.com/neobotix/neo_nav2_py_costmap2D)

## Sample Parameters

```
controller_server:
  ros__parameters:
    # controller server parameters (see Controller Server for more info)
    controller_plugins: ["FollowPath"]
    controller_frequency: 30.0
    controller_plugin_types: ["neo_mpc_planner::NeoMpcPlanner"]
    goal_checker_plugins: ["general_goal_checker"]
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 100.0
    general_goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.03
      yaw_goal_tolerance: 0.03
      stateful: True
    FollowPath:
      plugin: "neo_mpc_planner::NeoMpcPlanner"
      lookahead_dist_min: 0.4
      lookahead_dist_max: 0.4
      lookahead_dist_close_to_goal: 0.4
      control_steps: 3

mpc_optimization_server:
  ros__parameters:
    acc_x_limit: 2.5
    acc_y_limit: 2.5
    acc_theta_limit: 3.0
    min_vel_x: -0.7
    min_vel_y: -0.7
    min_vel_trans: -0.7
    min_vel_theta: -0.7
    max_vel_x: 0.7
    max_vel_y: 0.7
    max_vel_trans: 0.7
    max_vel_theta: 0.7
    w_trans: 0.82
    w_orient: 0.50
    w_control: 0.05
    w_terminal: 0.05
    w_footprint: 0
    w_costmap: 0.05
    waiting_time: 3.0
    low_pass_gain: 0.5
    opt_tolerance: 1e-3
    prediction_horizon: 0.8
    control_steps: 3

```

Note that the mpc_optimization server is a seperate node, since the optimization depends on the Scipy library. 

In the near future we plan to migrate the optimization process to C++. 

Special mention: Some of the code from nav2_regulated_pure_pursuit controller has been reused, since the neo_mpc_planner2 works on the principle of pure-pursuit as well. Thanks to the Nav2 team for that. 
