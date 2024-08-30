cmake_minimum_required(VERSION 3.0.2)
project(neural_tuner)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

## Add custom message files
add_message_files(
  FILES
  Observation.msg
)


## Generate messages with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs

)

## Declare a catkin package with message_runtime
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs 
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Add your source files, libraries, executables, etc., here if applicable
