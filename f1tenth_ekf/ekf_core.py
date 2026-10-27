"""
Planar EKF for an Ackermann car fusing wheel odometry and a yaw-rate gyro.

Pure numpy, no ROS dependency, so it can be unit tested off the car.

State (6):
    x, y      position in the odom frame [m]
    th        heading in the odom frame [rad], kept unwrapped internally
    v         forward speed in base_link [m/s]
    w         yaw rate [rad/s]
    bg        gyro z bias [rad/s]

Process model (constant velocity / constant turn rate, midpoint integration):
    x'  = x + v cos(th + w dt/2) dt
    y'  = y + v sin(th + w dt/2) dt
    th' = th + w dt
    v, w are driven by white acceleration / angular acceleration noise,
    bg is a slow random walk.

Measurements (all linear in the state):
    wheel speed      z = v
    odom yaw rate    z = w          (from servo command, so large noise)
    gyro z           z = w + bg
    zero-rate (ZUPT) z = w = 0      (only while the wheels are stopped,
                                     makes the gyro bias observable)
"""

import math

import numpy as np

IX, IY, ITH, IV, IW, IBG = range(6)
N = 6


class PlanarEKF:

    def __init__(self, params=None):
        p = dict(
            # process noise (continuous-time spectral densities)
            accel_noise=4.0,          # m/s^2, longitudinal accel std
            yaw_accel_noise=6.0,      # rad/s^2, yaw accel std
            gyro_bias_noise=2e-4,     # rad/s/sqrt(s), bias random walk
            # measurement noise (std devs)
            odom_speed_std=0.05,      # m/s, base
            odom_speed_std_rel=0.05,  # fraction of |v| added on top (slip)
            odom_yaw_rate_std=0.5,    # rad/s, kinematic yaw rate from servo cmd
            gyro_std=0.03,            # rad/s
            zupt_yaw_rate_std=0.005,  # rad/s
            # initial uncertainty
            init_pos_std=0.01,
            init_yaw_std=0.01,
            init_speed_std=0.5,
            init_yaw_rate_std=0.5,
            init_gyro_bias_std=0.05,
            # integration / robustness
            max_step=0.01,            # s, prediction sub-step
            max_gap=0.5,              # s, larger gaps are clamped
            gate_chi2=0.0,            # 1-dof NIS gate, 0 disables
            max_consecutive_rejects=10,
        )
        if params:
            unknown = set(params) - set(p)
            if unknown:
                raise ValueError(f'unknown EKF params: {sorted(unknown)}')
            p.update({k: float(v) for k, v in params.items()})
        self.p = p

        self.x = np.zeros(N)
        self.P = np.diag([
            p['init_pos_std'] ** 2, p['init_pos_std'] ** 2, p['init_yaw_std'] ** 2,
            p['init_speed_std'] ** 2, p['init_yaw_rate_std'] ** 2,
            p['init_gyro_bias_std'] ** 2,
        ])
        self.t = None
        self._rejects = {}
        self.stats = {'clamped_gaps': 0, 'rejected': 0, 'stale': 0}

    # ------------------------------------------------------------------ time

    def initialize(self, t, v=0.0):
        self.t = float(t)
        self.x[IV] = v

    @property
    def initialized(self):
        return self.t is not None

    def advance_to(self, t):
        """Predict the state forward to time t. Returns False if t is stale."""
        if self.t is None:
            self.initialize(t)
            return True
        dt = float(t) - self.t
        if dt < 0.0:
            # Message older than the filter time: apply it without rewinding.
            # Both topics are stamped on the same host at ~50 Hz, so this is rare
            # and the error is at most one sample period.
            self.stats['stale'] += 1
            return False
        if dt > self.p['max_gap']:
            self.stats['clamped_gaps'] += 1
            dt = self.p['max_gap']
        h = self.p['max_step']
        while dt > 1e-9:
            step = min(h, dt)
            self._predict(step)
            dt -= step
        self.t = float(t)
        return True

    def _predict(self, dt):
        x, y, th, v, w, bg = self.x
        thm = th + 0.5 * w * dt
        c, s = math.cos(thm), math.sin(thm)

        self.x[IX] = x + v * c * dt
        self.x[IY] = y + v * s * dt
        self.x[ITH] = th + w * dt

        F = np.eye(N)
        F[IX, ITH] = -v * s * dt
        F[IX, IV] = c * dt
        F[IX, IW] = -0.5 * v * s * dt * dt
        F[IY, ITH] = v * c * dt
        F[IY, IV] = s * dt
        F[IY, IW] = 0.5 * v * c * dt * dt
        F[ITH, IW] = dt

        # Noise inputs: longitudinal accel, yaw accel, bias random walk.
        G = np.zeros((N, 3))
        G[IX, 0] = 0.5 * c * dt * dt
        G[IY, 0] = 0.5 * s * dt * dt
        G[IV, 0] = dt
        G[ITH, 1] = 0.5 * dt * dt
        G[IW, 1] = dt
        G[IBG, 2] = 1.0
        Qc = np.diag([
            self.p['accel_noise'] ** 2 / dt,
            self.p['yaw_accel_noise'] ** 2 / dt,
            self.p['gyro_bias_noise'] ** 2 * dt,
        ])
        # G Qc G^T: accel terms give sigma^2 dt on v (white accel), bias gives sigma^2 dt.
        self.P = F @ self.P @ F.T + G @ Qc @ G.T

    # --------------------------------------------------------------- updates

    def _update_scalar(self, name, z, H, r_std, frozen=()):
        H = np.asarray(H, dtype=float)
        y = z - H @ self.x
        PHt = self.P @ H
        S = H @ PHt + r_std * r_std
        gate = self.p['gate_chi2']
        if gate > 0.0 and y * y / S > gate:
            n = self._rejects.get(name, 0) + 1
            self._rejects[name] = n
            if n <= self.p['max_consecutive_rejects']:
                self.stats['rejected'] += 1
                return False
            # Persistent disagreement: the filter is probably wrong, not the sensor.
        self._rejects[name] = 0
        K = PHt / S
        for i in frozen:
            # Schmidt 'consider' update: this measurement may not move state i.
            # Joseph form below stays valid for any gain.
            K[i] = 0.0
        self.x = self.x + K * y
        I_KH = np.eye(N) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * (r_std * r_std)
        return True

    def update_odom_speed(self, v_meas):
        r = self.p['odom_speed_std'] + self.p['odom_speed_std_rel'] * abs(v_meas)
        H = np.zeros(N)
        H[IV] = 1.0
        return self._update_scalar('odom_v', v_meas, H, r)

    def update_odom_yaw_rate(self, w_meas):
        H = np.zeros(N)
        H[IW] = 1.0
        # The kinematic yaw rate is biased (understeer, servo lag, it uses the
        # *commanded* servo). Letting it correct the gyro bias corrupts the bias
        # while cornering, so the bias is frozen here and is learned only from
        # zero-rate updates while the car is stopped.
        return self._update_scalar('odom_w', w_meas, H, self.p['odom_yaw_rate_std'],
                                   frozen=(IBG,))

    def update_gyro(self, gz):
        H = np.zeros(N)
        H[IW] = 1.0
        H[IBG] = 1.0
        return self._update_scalar('gyro', gz, H, self.p['gyro_std'])

    def update_zero_yaw_rate(self):
        H = np.zeros(N)
        H[IW] = 1.0
        return self._update_scalar('zupt', 0.0, H, self.p['zupt_yaw_rate_std'])

    # ---------------------------------------------------------------- output

    @property
    def yaw_wrapped(self):
        return math.atan2(math.sin(self.x[ITH]), math.cos(self.x[ITH]))
