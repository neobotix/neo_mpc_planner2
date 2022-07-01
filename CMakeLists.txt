cmake_minimum_required(VERSION 3.5)
project(neo_mpc_planner2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(neo_srvs2 REQUIRED)
find_package(rclpy REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
)

set(dependencies
  rclcpp
  rclpy
  geometry_msgs
  nav2_costmap_2d
  pluginlib
  nav_msgs
  nav2_util
  nav2_core
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  neo_srvs2
)

ament_python_install_package(${PROJECT_NAME})

set(library_name neo_mpc_planner2)

add_library(${library_name} SHARED
        src/NeoMpcPlanner.cpp)

ament_target_dependencies(${library_name}
  ${dependencies}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(FILES 
  neo_mpc_planner_plugin.xml
  DESTINATION share
  )

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(PROGRAMS neo_mpc_planner2/mpc_optimization_server.py DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})

pluginlib_export_plugin_description_file(nav2_core neo_mpc_planner_plugin.xml)

ament_package()
