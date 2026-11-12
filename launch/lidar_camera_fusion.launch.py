"""Start lidar_camera_fusion_node. Every argument is the node parameter of the same name.

  ros2 launch lidar_camera_fusion lidar_camera_fusion.launch.py calib_path:=/path/calib.json image_topic:=/image
Second camera (optional): calib_path_back:=/path/calib_back.json image_topic_back:=/image_back
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name, default (= node default; None = required), type
ARGS = [
    ('image_topic', '/image', str),
    ('cloud_topic', '/ouster/points', str),
    ('calib_path', None, str),
    ('overlap_image_topic', '/overlap_image', str),
    ('colored_cloud_topic', '/colored_cloud', str),
    ('image_topic_back', '/camera_back', str),
    ('calib_path_back', '', str),
    ('overlap_image_topic_back', '/overlap_image_back', str),
    ('colored_cloud_topic_back', '/colored_cloud_back', str),
    ('queue_size', '3', int),
    ('image_qos', 'reliable', str),
    ('point_radius', '2', int),
    ('max_distance', '8.0', float),
]


def generate_launch_description():
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=typ) for name, _, typ in ARGS}
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default, _ in ARGS] + [  # None = required
            Node(package='lidar_camera_fusion', executable='lidar_camera_fusion_node',
                 name='lidar_camera_fusion', output='screen', parameters=[params]),
        ])
