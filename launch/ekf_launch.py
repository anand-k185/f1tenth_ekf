import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('f1tenth_ekf'), 'config', 'ekf.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('ekf_config', default_value=default_config),
        Node(
            package='f1tenth_ekf',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[LaunchConfiguration('ekf_config')],
        ),
    ])
