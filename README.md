# f1tenth_ekf

Planar EKF that sits between the VESC stack and the particle filter. It fuses
wheel speed from `/odom` with the VESC onboard IMU gyro from `/sensors/imu/raw`
and republishes fused odometry on `/ekf/odom`, plus `odom -> base_link` TF.

```
vesc_driver_node ── /sensors/core ──> vesc_to_odom_node ── /odom ──┐
        └────────── /sensors/imu/raw ─────────────────────────────┤
                                                                  v
                                                             ekf_node ── /ekf/odom ──> particle_filter
```

## Why

`vesc_to_odom` computes yaw rate as `v * tan(servo_command) / wheelbase`. It uses the
*commanded* servo, so it ignores servo lag, understeer and wheel slip, and heading drifts
fast. The VESC already streams its gyro, so the EKF uses that for yaw rate and keeps wheel
speed for distance.

## Model

State `[x, y, yaw, v, yaw_rate, gyro_bias]`, constant velocity / turn rate process model.

| Measurement | Source | Used when |
|---|---|---|
| `v` | `/odom` twist.linear.x (ERPM) | always |
| `yaw_rate + bias` | IMU gyro (deg/s → rad/s) | always |
| `yaw_rate = 0` | zero-rate update | wheels stopped and gyro quiet: this is how the gyro bias is learned |
| `yaw_rate` | `/odom` twist.angular.z (servo model) | only if the IMU is silent for `imu_timeout` |

Two design decisions came out of simulation testing (`test/test_ekf_core.py`):

* **The servo yaw rate is not fused while the gyro is alive.** It is systematically biased,
  so even with a large noise value it pulls heading off by tens of degrees over a minute.
  It is only a fallback (`odom_yaw_rate_mode: fallback`).
* **The gyro bias is only learned while stopped.** Otherwise the bias state absorbs
  understeer error during corners. Stopping for a couple of seconds at startup is enough;
  each later stop re-trims it.

In simulation (servo lag, ~20% understeer, 0.03 rad/s gyro bias, 60 s run) heading error
stays within ~3–6° versus several full turns of drift for raw `/odom`. Final position error
drops from ~6 m to ~0.4 m.

## Build and run

```bash
cd ~/f1tenth_ws && colcon build --packages-select f1tenth_ekf f1tenth_stack
source install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py   # starts ekf_node too
```

Then set the particle filter's `odometry_topic: '/ekf/odom'` in `particle_filter/config/localize.yaml`.

Standalone: `ros2 launch f1tenth_ekf ekf_launch.py`. Tests (no ROS needed): `python3 -m pytest test/`.

## First run on the car

1. **Leave the car still for ~3 s after launch.** The bias is learned here. The node logs
   `gyro bias ...` every 10 s; it should settle near a constant, typically a few hundredths of a rad/s.
2. **Check the gyro sign.** Rotate the car counter-clockwise by hand (seen from above), briskly, about a
   quarter turn per second (slower rotation with the wheels stopped is treated as standing still):
   `ros2 topic echo /ekf/odom --field twist.twist.angular.z` must go **positive**. If not,
   set `gyro_sign: -1.0`. The node also logs an error after ~100 turning samples if the
   gyro disagrees with steering. If the VESC is mounted on its side, change `gyro_axis`.
3. **Check the IMU is publishing:** `ros2 topic hz /sensors/imu/raw` should be ~50 Hz. If it is
   silent, check in VESC Tool that your VESC has an onboard IMU and that it reports data. The
   EKF then runs on the servo model only and logs a warning every 10 s.
4. **Circle test:** drive a few full circles and come back to the start mark. `/ekf/odom`
   yaw should return within a few degrees; `/odom` usually will not.
5. **Speed calibration still matters.** The EKF trusts wheel speed for distance. If straight-line
   distance is off, fix `speed_to_erpm_gain` in `vesc.yaml`. The EKF cannot correct it.

## Tuning

* Heading still drifts while driving → lower `gyro_std` or check bias settled.
* Yaw rate noisy → lower `yaw_accel_noise`.
* Wheel spin on hard launches → raise `odom_speed_std_rel`, or set `gate_chi2: 9.0`.
* Running without the EKF: set `publish_tf: true` in `vesc.yaml`, remove `ekf_node` from
  bringup, and point the particle filter back to `/odom`.
