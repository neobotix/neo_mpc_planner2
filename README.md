# neo_mpc_planner2

The Model Predictive Control (MPC) is an optimization-based feedback control technique, which was initially developed for the field of process industries. With the recent advancements in cost-effective computation, MPC was extended to the field of robotics for real-time control. This means that the optimization happens on the fly. MPC otherwise also called as the Receding Horizon Control (RHC) is utilized in robots for the purpose of path following and / or trajectory tracking.

neo_mpc_planner2 is exclusively developed for quasi-omnidirectional robots such as Neobotix MPO-700. MPO-700 is used for transporting various objects in an structured environment. Therefore this planner aims in catering all the omni-directional robots used under a known environment. The controller was tested on an Intel i5 process and was found that it can easily work at an average rate of 50 Hz. This would help to instatneously react to the suddenly occuring obstacles. Currently, the planner is designed in a way to stop the robot if there is an unknown obstacle in the predicted path. 

The entire planner is built on the Navigation2 stack for ROS-2. Currently, we support omni-directional robots and later will be extended to differential robots. 

## Dependencies 

Python3.8.10
Scipy 
