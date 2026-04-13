cmake_minimum_required(VERSION 3.16)
project(lidar_camera_fusion)

add_compile_options(-std=c++17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# ROS 2 build system
# ---------------------------------------------------------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# ---------------------------------------------------------------------------
# Third-party (non-ROS) dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# nlohmann/json – prefer the cmake package, fall back to a bare header search,
# then fall back to the copy bundled with direct_visual_lidar_calibration.
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS
      /usr/include
      /usr/local/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../direct_visual_lidar_calibration/thirdparty/json/include
  )
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
      "nlohmann/json.hpp not found. Install nlohmann-json3-dev or ensure "
      "direct_visual_lidar_calibration is present in the same workspace.")
  endif()
endif()

# cv_bridge include-path guard (humble/galactic/foxy use cv_bridge.h,
# newer distros use cv_bridge.hpp)
set(OLD_DISTRO "humble;galactic;foxy")
if($ENV{ROS_DISTRO} IN_LIST OLD_DISTRO)
  add_definitions(-DCV_BRIDGE_INCLUDE_H)
else()
  add_definitions(-DCV_BRIDGE_INCLUDE_HPP)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
ament_auto_add_executable(lidar_camera_fusion_node
  src/lidar_camera_fusion_node.cpp
)

target_include_directories(lidar_camera_fusion_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(lidar_camera_fusion_node
  ${OpenCV_LIBRARIES}
)

if(nlohmann_json_FOUND)
  target_link_libraries(lidar_camera_fusion_node nlohmann_json::nlohmann_json)
else()
  target_include_directories(lidar_camera_fusion_node PRIVATE
    ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# ---------------------------------------------------------------------------
install(TARGETS lidar_camera_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()
