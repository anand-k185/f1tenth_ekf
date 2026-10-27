import os
from glob import glob

from setuptools import setup

package_name = 'f1tenth_ekf'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anand Krishnan',
    maintainer_email='anandkrishnan1835@gmail.com',
    description='EKF fusing VESC wheel odometry with the VESC IMU gyro for F1TENTH.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = f1tenth_ekf.ekf_node:main',
        ],
    },
)
