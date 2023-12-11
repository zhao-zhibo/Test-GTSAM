cmake_minimum_required(VERSION 2.8.3)
project(test_gtsam)

find_package(Boost REQUIRED COMPONENTS thread)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
)

find_package( GTSAMCMakeTools )

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(Ceres REQUIRED)


find_package(Boost REQUIRED COMPONENTS timer)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(Boost REQUIRED COMPONENTS chrono)

find_package(PythonLibs REQUIRED)

# Python3,numpy
find_package(Python3 COMPONENTS Development NumPy)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)


generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM EIGEN3

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  message_generation
  visualization_msgs
)

# include directories
include_directories(
	include
	${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

# link directories
link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

###########
## Build ##
###########


add_executable(${PROJECT_NAME}_testGtsam src/testGtsam.cpp)
add_dependencies(${PROJECT_NAME}_testGtsam ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_testGtsam ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam )

add_executable(${PROJECT_NAME}_gaussNewton src/gaussNewton.cpp)
add_dependencies(${PROJECT_NAME}_gaussNewton ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_gaussNewton ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam )

add_executable(${PROJECT_NAME}_curveFitting src/curveFitting.cpp)
add_dependencies(${PROJECT_NAME}_curveFitting ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_curveFitting ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam )


# 如果你想画图，那么就采用下面的三个文件的编译方式
# add_executable(${PROJECT_NAME}_testGtsam src/testGtsam.cpp)
# add_dependencies(${PROJECT_NAME}_testGtsam ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(${PROJECT_NAME}_testGtsam ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam Python3::Python Python3::NumPy)
# target_include_directories(${PROJECT_NAME}_testGtsam PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# add_executable(${PROJECT_NAME}_gaussNewton src/gaussNewton.cpp)
# add_dependencies(${PROJECT_NAME}_gaussNewton ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(${PROJECT_NAME}_gaussNewton ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam Python3::Python Python3::NumPy)
# target_include_directories(${PROJECT_NAME}_gaussNewton PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# add_executable(${PROJECT_NAME}_curveFitting src/curveFitting.cpp)
# add_dependencies(${PROJECT_NAME}_curveFitting ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(${PROJECT_NAME}_curveFitting ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam Python3::Python Python3::NumPy)
# target_include_directories(${PROJECT_NAME}_curveFitting PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

##----------------------------分隔符 下面的代码是lio-sam的代码，用不上暂时注释掉---------------------------------------------
# # Feature Association
# add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
# add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# # Mapping Optimization
# add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
# add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)

# # IMU Preintegration
# add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
# target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)
