"""Simulation tests for the planar EKF (no ROS needed): pytest test/test_ekf_core.py"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from f1tenth_ekf.ekf_core import IBG, ITH, IW, PlanarEKF  # noqa: E402

WHEELBASE = 0.25
RATE = 50.0


def simulate(seed=0, duration=60.0, bias=0.03, stationary=3.0):
    """
    Ground truth plus the two sensor streams, mimicking the real stack:
      - wheel speed: true speed + noise, zeroed below 0.05 m/s (vesc_to_odom does this)
      - odom yaw rate: v * tan(commanded steering) / L  (vesc_to_odom uses the servo
        command, so it ignores servo lag and understeer)
      - gyro: true yaw rate + constant bias + white noise
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / RATE
    n = int(duration * RATE)
    x = y = th = 0.0
    delta = 0.0
    out = []
    for k in range(n):
        t = k * dt
        if t < stationary:
            v_cmd, d_cmd = 0.0, 0.0
        else:
            tt = t - stationary
            v_cmd = 2.5 + 1.5 * math.sin(0.3 * tt)
            d_cmd = 0.30 * math.sin(0.5 * tt) + 0.1 * math.sin(1.7 * tt)
        delta += (d_cmd - delta) * min(1.0, dt / 0.08)    # servo lag
        understeer = 1.0 / (1.0 + 0.04 * v_cmd ** 2)       # slip at speed
        w_true = v_cmd * math.tan(delta) / WHEELBASE * understeer
        x += v_cmd * math.cos(th + 0.5 * w_true * dt) * dt
        y += v_cmd * math.sin(th + 0.5 * w_true * dt) * dt
        th += w_true * dt

        v_meas = v_cmd + rng.normal(0, 0.04) if v_cmd > 0 else 0.0
        if abs(v_meas) < 0.05:
            v_meas = 0.0
        w_odom = v_meas * math.tan(d_cmd) / WHEELBASE
        gz = w_true + bias + rng.normal(0, 0.02)
        out.append(dict(t=t, x=x, y=y, th=th, w=w_true, v_meas=v_meas,
                        w_odom=w_odom, gz=gz))
    return out


def run_ekf(data, imu_dropout=None, sample=None):
    """Mirrors the node's default policy: servo yaw rate only while the IMU is stale."""
    ekf = PlanarEKF()
    ekf.initialize(data[0]['t'])
    for d in data:
        imu_ok = imu_dropout is None or not (imu_dropout[0] <= d['t'] < imu_dropout[1])
        stationary = d['v_meas'] == 0.0
        ekf.advance_to(d['t'])
        ekf.update_odom_speed(d['v_meas'])
        if stationary:
            ekf.update_zero_yaw_rate()
        elif not imu_ok:
            ekf.update_odom_yaw_rate(d['w_odom'])
        if imu_ok:
            # gyro arrives slightly after odom, like two separate topics
            ekf.advance_to(d['t'] + 0.004)
            if stationary:
                ekf.update_zero_yaw_rate()
            ekf.update_gyro(d['gz'])
        if sample is not None:
            sample(d, ekf.x)
    return ekf


def heading_err(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def errors(data, estimator):
    """(max unwrapped heading error, final position error) along the whole run."""
    errs = []
    final = {}

    def sample(d, x):
        errs.append(abs(x[ITH] - d['th']))   # unwrapped: no saturation at pi
        final['pos'] = math.hypot(x[0] - d['x'], x[1] - d['y'])

    estimator(data, sample)
    return max(errs), final['pos']


def raw_estimator(data, sample):
    x = y = th = 0.0
    prev = None
    for d in data:
        if prev is not None:
            dt = d['t'] - prev['t']
            x += d['v_meas'] * math.cos(th) * dt
            y += d['v_meas'] * math.sin(th) * dt
            th += d['w_odom'] * dt
        prev = d
        sample(d, (x, y, th))


def ekf_estimator(**kw):
    return lambda data, sample: run_ekf(data, sample=sample, **kw)


def test_ekf_beats_wheel_odometry():
    for seed in range(3):
        data = simulate(seed=seed)
        th_raw, pos_raw = errors(data, raw_estimator)
        th_ekf, pos_ekf = errors(data, ekf_estimator())
        print(f'\nseed {seed}: raw max heading err {math.degrees(th_raw):5.1f} deg, '
              f'final pos err {pos_raw:.2f} m | EKF {math.degrees(th_ekf):4.1f} deg, '
              f'{pos_ekf:.2f} m')
        assert th_ekf < math.radians(8.0)
        assert pos_ekf < 0.25 * pos_raw


def test_imu_dropout_falls_back_to_servo_yaw_rate():
    data = simulate(seed=2)
    raw, ekf = [], []
    raw_estimator(data, lambda d, x: raw.append(x[2] - d['th']))
    run_ekf(data, imu_dropout=(20.0, 30.0), sample=lambda d, x: ekf.append(x[ITH] - d['th']))
    k20, k30, kend = int(20 * RATE), int(30 * RATE), len(data) - 1
    raw_drift = raw[k30] - raw[k20]
    ekf_drift = ekf[k30] - ekf[k20]
    after = ekf[kend] - ekf[k30 + 25]
    print(f'\nIMU out 20-30 s: heading drift EKF {math.degrees(ekf_drift):.1f} deg, '
          f'raw odom {math.degrees(raw_drift):.1f} deg; after IMU returns '
          f'{math.degrees(after):.1f} deg over 30 s')
    # Without the gyro the EKF can only be as good as the servo model ...
    assert abs(ekf_drift - raw_drift) < 0.2 * abs(raw_drift)
    # ... and once the gyro is back, drift returns to gyro-grade.
    assert abs(after) < math.radians(8.0)


def test_gyro_bias_learned_while_stationary():
    data = simulate(duration=3.0, stationary=3.0, bias=0.04)
    ekf = run_ekf(data)
    assert abs(ekf.x[IBG] - 0.04) < 0.005
    assert abs(ekf.x[IW]) < 0.01


def test_yaw_rate_tracks_gyro():
    data = simulate(seed=3)
    ekf = PlanarEKF()
    ekf.initialize(0.0)
    errs = []
    for d in data:
        ekf.advance_to(d['t'])
        ekf.update_odom_speed(d['v_meas'])
        if d['v_meas'] == 0.0:
            ekf.update_zero_yaw_rate()
        ekf.update_gyro(d['gz'])
        if d['t'] > 10:
            errs.append(abs(ekf.x[IW] - d['w']))
    assert np.mean(errs) < 0.03


def test_stale_and_gap_handling():
    ekf = PlanarEKF()
    ekf.initialize(10.0, v=1.0)
    assert ekf.advance_to(10.1)
    assert not ekf.advance_to(10.05)       # older message is not rewound
    ekf.advance_to(20.0)                   # 10 s gap is clamped
    assert ekf.stats['clamped_gaps'] == 1
    assert np.all(np.isfinite(ekf.P))
    assert np.allclose(ekf.P, ekf.P.T)


def test_bias_not_corrupted_by_understeer():
    data = simulate(seed=1, duration=90.0, bias=0.03)
    ekf = run_ekf(data)
    assert abs(ekf.x[IBG] - 0.03) < 0.006


def test_covariance_stays_positive_definite():
    data = simulate(seed=7, duration=30.0)
    ekf = run_ekf(data)
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0)


if __name__ == '__main__':
    test_ekf_beats_wheel_odometry()
