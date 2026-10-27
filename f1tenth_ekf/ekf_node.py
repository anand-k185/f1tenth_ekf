"""
ROS 2 node: fuses VESC wheel odometry with the VESC IMU gyro and republishes odometry.

Subscribes
    odom_topic  (nav_msgs/Odometry)   from vesc_to_odom_node, ~50 Hz
                twist.linear.x  = wheel speed from ERPM (used)
                twist.angular.z = v*tan(servo_cmd)/L (used only as IMU fallback)
    imu_topic   (sensors_msgs/Imu)    from vesc_driver_node /sensors/imu/raw, ~50 Hz
                angular_velocity is in deg/s (see vesc_packet.cpp), no frame_id,
                zero covariance -> those fields are ignored and noise comes from params

Publishes
    output_topic (nav_msgs/Odometry)  fused odom, one message per input odom message
                                      so the particle filter keeps its update cadence
    TF odom_frame -> base_frame       if publish_tf (turn vesc_to_odom's TF off then)
"""

import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from f1tenth_ekf.ekf_core import IBG, ITH, IV, IW, IX, IY, PlanarEKF

# Parameters forwarded to PlanarEKF as floats (Foxy does not enforce param types).
CORE_PARAMS = [
    'accel_noise', 'yaw_accel_noise', 'gyro_bias_noise',
    'odom_speed_std', 'odom_speed_std_rel', 'odom_yaw_rate_std', 'gyro_std',
    'zupt_yaw_rate_std', 'init_gyro_bias_std', 'max_step', 'max_gap', 'gate_chi2',
]


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_node')
        dp = self.declare_parameter
        self.odom_topic = dp('odom_topic', '/odom').value
        self.imu_topic = dp('imu_topic', '/sensors/imu/raw').value
        self.output_topic = dp('output_topic', '/ekf/odom').value
        self.odom_frame = dp('odom_frame', 'odom').value
        self.base_frame = dp('base_frame', 'base_link').value
        self.publish_tf = bool(dp('publish_tf', False).value)

        # VESC reports gyro in deg/s. Axis/sign depend on how the VESC is mounted.
        self.gyro_axis = dp('gyro_axis', 'z').value
        self.gyro_scale = float(dp('gyro_scale', math.pi / 180.0).value)
        self.gyro_sign = float(dp('gyro_sign', 1.0).value)

        # 'fallback': servo-derived yaw rate only while the IMU is stale (recommended)
        # 'always'  : fuse it too (it is biased by understeer/servo lag: hurts heading)
        # 'never'   : never use it
        self.odom_yaw_rate_mode = dp('odom_yaw_rate_mode', 'fallback').value
        self.imu_timeout = float(dp('imu_timeout', 0.25).value)

        # Zero-rate updates while stopped: learn the gyro bias.
        self.stationary_speed = float(dp('stationary_speed', 0.01).value)
        self.zupt_max_gyro = float(dp('zupt_max_gyro', 0.3).value)

        core = {k: float(dp(k, PlanarEKF().p[k]).value) for k in CORE_PARAMS}
        self.ekf = PlanarEKF(core)

        if self.gyro_axis not in ('x', 'y', 'z'):
            raise ValueError("gyro_axis must be 'x', 'y' or 'z'")
        if self.odom_yaw_rate_mode not in ('fallback', 'always', 'never'):
            raise ValueError("odom_yaw_rate_mode must be fallback, always or never")

        self.last_imu_time = None     # filter-time of the last gyro sample
        self.last_gyro = 0.0
        self.stationary = True
        self.sign_votes = [0, 0]      # [agree, disagree] gyro vs servo yaw rate
        self.sign_warned = False

        self.pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.tf_pub = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_timer(10.0, self.log_stats)

        self.get_logger().info(
            f'EKF: {self.odom_topic} + {self.imu_topic} -> {self.output_topic} '
            f'(gyro {self.gyro_sign:+.0f}*{self.gyro_axis}, '
            f'servo yaw rate: {self.odom_yaw_rate_mode}, tf: {self.publish_tf})')

    # ----------------------------------------------------------------- utils

    def stamp_to_sec(self, stamp):
        t = Time.from_msg(stamp).nanoseconds
        if t == 0:
            t = self.get_clock().now().nanoseconds
        return t * 1e-9

    def imu_fresh(self, t):
        return self.last_imu_time is not None and (t - self.last_imu_time) < self.imu_timeout

    # ------------------------------------------------------------- callbacks

    def odom_cb(self, msg):
        t = self.stamp_to_sec(msg.header.stamp)
        v = msg.twist.twist.linear.x
        w_odom = msg.twist.twist.angular.z

        if not self.ekf.initialized:
            self.ekf.initialize(t, v)
            self.get_logger().info('EKF initialized from first odom message')

        self.ekf.advance_to(t)
        self.ekf.update_odom_speed(v)

        unbiased_gyro = self.last_gyro - self.ekf.x[IBG]
        self.stationary = (abs(v) < self.stationary_speed
                           and (not self.imu_fresh(t) or abs(unbiased_gyro) < self.zupt_max_gyro))

        if self.stationary:
            self.ekf.update_zero_yaw_rate()
        elif self.odom_yaw_rate_mode == 'always' or (
                self.odom_yaw_rate_mode == 'fallback' and not self.imu_fresh(t)):
            self.ekf.update_odom_yaw_rate(w_odom)
            if self.last_imu_time is not None and self.odom_yaw_rate_mode == 'fallback':
                self.get_logger().warn(
                    f'No IMU on {self.imu_topic} for {t - self.last_imu_time:.2f} s, '
                    'using servo-command yaw rate', throttle_duration_sec=2.0)

        self.check_gyro_sign(v, w_odom, unbiased_gyro, t)
        self.publish(msg.header.stamp)

    def imu_cb(self, msg):
        if not self.ekf.initialized:
            return  # wait for odom so both share a time origin
        t = self.stamp_to_sec(msg.header.stamp)
        gz = self.gyro_sign * self.gyro_scale * getattr(msg.angular_velocity, self.gyro_axis)
        self.ekf.advance_to(t)
        if self.stationary:
            # Keep the zero-rate constraint at gyro time too, otherwise the yaw-rate
            # uncertainty grows between odom and gyro samples and the bias is
            # mis-learned.
            self.ekf.update_zero_yaw_rate()
        self.ekf.update_gyro(gz)
        self.last_gyro = gz
        self.last_imu_time = t

    # ---------------------------------------------------------- diagnostics

    def check_gyro_sign(self, v, w_odom, gyro, t):
        """Warn once if the gyro looks mounted upside down relative to steering."""
        if self.sign_warned or not self.imu_fresh(t):
            return
        if abs(v) < 0.5 or abs(w_odom) < 0.5:
            return
        self.sign_votes[0 if gyro * w_odom > 0 else 1] += 1
        agree, disagree = self.sign_votes
        if agree + disagree >= 100:
            if disagree > 0.8 * (agree + disagree):
                self.get_logger().error(
                    'Gyro yaw rate has the opposite sign to the steering yaw rate in '
                    f'{disagree}/{agree + disagree} turning samples. Flip gyro_sign '
                    '(or check gyro_axis / the VESC mounting).')
            self.sign_warned = True

    def log_stats(self):
        if not self.ekf.initialized:
            self.get_logger().warn(f'Waiting for odometry on {self.odom_topic}',
                                   throttle_duration_sec=10.0)
            return
        if self.last_imu_time is None:
            self.get_logger().warn(f'No IMU received on {self.imu_topic}; running on '
                                   'servo yaw rate only')
        s = self.ekf.stats
        self.get_logger().info(
            f'gyro bias {self.ekf.x[IBG]:+.4f} rad/s, '
            f'yaw std {math.degrees(math.sqrt(self.ekf.P[ITH, ITH])):.1f} deg, '
            f'stale msgs {s["stale"]}, clamped gaps {s["clamped_gaps"]}, '
            f'gated {s["rejected"]}')

    # ---------------------------------------------------------------- output

    def publish(self, stamp):
        x, P = self.ekf.x, self.ekf.P
        yaw = self.ekf.yaw_wrapped
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(x[IX])
        odom.pose.pose.position.y = float(x[IY])
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(x[IV])
        odom.twist.twist.angular.z = float(x[IW])

        # 6x6 row-major, order (x, y, z, roll, pitch, yaw). Planar: small z/roll/pitch.
        pc = [0.0] * 36
        for i, si in ((0, IX), (1, IY), (5, ITH)):
            for j, sj in ((0, IX), (1, IY), (5, ITH)):
                pc[6 * i + j] = float(P[si, sj])
        pc[14] = pc[21] = pc[28] = 1e-6
        odom.pose.covariance = pc
        tc = [0.0] * 36
        tc[0] = float(P[IV, IV])
        tc[35] = float(P[IW, IW])
        tc[5] = tc[30] = float(P[IV, IW])
        tc[7] = tc[14] = tc[21] = tc[28] = 1e-6
        odom.twist.covariance = tc
        self.pub.publish(odom)

        if self.tf_pub is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = float(x[IX])
            tf.transform.translation.y = float(x[IY])
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_pub.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
