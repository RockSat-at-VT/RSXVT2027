"""
Accelerometer model for the RockSat-X GNC simulation.

Takes the true state from the CubeSat sim (slide 4 loop) and returns what a
real accelerometer would measure:

    a_true = a_cm_body + alpha x r + omega x (omega x r)      (rigid-body kinematics)
    a_meas = (I + S + M) @ a_true + bias + noise              (sensor errors)
    then saturate and quantize

where
    a_cm_body : specific force at the center of mass, in the body frame
                (zero in free fall / orbit; gravity + motion on the ground or a pendulum)
    r         : sensor position relative to the center of mass, body frame [m]
    S         : diagonal scale-factor error
    M         : off-diagonal misalignment / cross-axis error
    bias      : constant offset + slow random walk
    noise     : white noise from the datasheet noise density

Quaternion convention matches the slides: q = [q0, q1, q2, q3] = [q0, qv],
scalar first, rotating body -> world. omega is body-frame angular velocity.

IMU: STMicroelectronics ISM6HG256X (low-g channel, ±16 g)
Datasheet: DS15340 Rev 1 (September 2026)
URL: https://www.st.com/resource/en/datasheet/ism6hgk256x.pdf

All default parameters below are taken directly from the datasheet using
worst-case (maximum) values unless noted. Two parameters (bias_random_walk,
misalignment_std) are not specified in the datasheet and use conservative
estimates typical for this class of ST industrial MEMS IMU.
"""

import numpy as np

G0 = 9.80665  # standard gravity [m/s^2]


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-first, same as the slides)
# ---------------------------------------------------------------------------
def skew(v):
    """[v]x : the cross-product matrix from the slides."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def quat_to_dcm(q):
    """Rotation matrix body -> world for scalar-first quaternion q."""
    q0, qv = q[0], q[1:]
    return (q0**2 - qv @ qv) * np.eye(3) + 2.0 * np.outer(qv, qv) + 2.0 * q0 * skew(qv)


def quat_derivative(q, omega):
    """q_dot = 1/2 [ -qv^T ; q0 I3 + [qv]x ] omega   (slide 4)."""
    q0, qv = q[0], q[1:]
    Xi = np.vstack((-qv, q0 * np.eye(3) + skew(qv)))
    return 0.5 * Xi @ omega


# ---------------------------------------------------------------------------
# Accelerometer model
# ---------------------------------------------------------------------------
class Accelerometer:
    def __init__(self,
                 rate_hz=100.0,
                 position=(0.05, 0.0, 0.0),          # sensor offset from CoM [m] — confirm with Jhamar
                 # --- ISM6HG256X datasheet values (DS15340 Rev 1, low-g channel ±16 g) ---
                 noise_density=100e-6 * G0,          # 100 µg/√Hz worst-case max, high-perf mode [m/s²/√Hz]
                 bias_initial_std=65e-3 * G0,        # ±65 mg worst-case zero-g offset [m/s²]
                 bias_random_walk=100e-6 * G0,       # ~100 µg/√s — not in datasheet; conservative MEMS estimate [m/s²/√s]
                 scale_factor_std=0.01,              # ±1% sensitivity tolerance (worst-case)
                 misalignment_std=np.deg2rad(1.0),  # ~1° — not in datasheet; conservative MEMS estimate [rad]
                 full_scale=16 * G0,                 # ±16 g low-g channel [m/s²]
                 resolution_bits=16,                 # 0.488 mg/LSB at ±16 g → 32 000 mg / 0.488 ≈ 2^16 steps
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / rate_hz
        self.r = np.asarray(position, dtype=float)

        self.noise_std = noise_density * np.sqrt(rate_hz)   # per-sample white noise
        self.bias_rw_std = bias_random_walk * np.sqrt(self.dt)
        self.full_scale = full_scale
        self.lsb = 2.0 * full_scale / 2**resolution_bits

        # Fixed errors, drawn once per "sensor unit" (like a real part off the line)
        S = np.diag(self.rng.normal(0.0, scale_factor_std, 3))
        M = self.rng.normal(0.0, misalignment_std, (3, 3))
        np.fill_diagonal(M, 0.0)
        self.C = np.eye(3) + S + M                           # error matrix
        self.bias = self.rng.normal(0.0, bias_initial_std, 3)

    # -- truth ---------------------------------------------------------------
    def true_specific_force(self, omega, alpha, a_cm_body=np.zeros(3)):
        """What a perfect accelerometer at position r would read (body frame)."""
        return (a_cm_body
                + np.cross(alpha, self.r)
                + np.cross(omega, np.cross(omega, self.r)))

    # -- measurement ---------------------------------------------------------
    def measure(self, omega, alpha, a_cm_body=np.zeros(3)):
        """One noisy sample. Call once per sensor time step."""
        a_true = self.true_specific_force(omega, alpha, a_cm_body)

        self.bias = self.bias + self.rng.normal(0.0, self.bias_rw_std, 3)
        noise = self.rng.normal(0.0, self.noise_std, 3)

        a_meas = self.C @ a_true + self.bias + noise
        a_meas = np.clip(a_meas, -self.full_scale, self.full_scale)   # saturation
        a_meas = np.round(a_meas / self.lsb) * self.lsb               # quantization
        return a_meas, a_true

    @staticmethod
    def gravity_in_body(q, g_world=np.array([0.0, 0.0, -G0])):
        """Specific force from resting in a gravity field, in the body frame.
        Use for ground tests / the pendulum. In orbit (free fall) pass zeros instead."""
        return quat_to_dcm(q).T @ (-g_world)


# ---------------------------------------------------------------------------
# Demo: the slide-4 attitude loop with the accelerometer attached
# ---------------------------------------------------------------------------
def run_demo(t_final=20.0, in_orbit=True, plot=True):
    dt = 0.01
    I = np.diag([0.010, 0.012, 0.008])       # placeholder CubeSat inertia [kg m^2]
    I_inv = np.linalg.inv(I)

    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.05, -0.02, 0.10])    # [rad/s]
    accel = Accelerometer(rate_hz=1 / dt, seed=0)

    n = int(t_final / dt)
    t = np.arange(n) * dt
    true_log, meas_log = np.zeros((n, 3)), np.zeros((n, 3))

    for k in range(n):
        tau = 1e-4 * np.array([np.sin(0.5 * t[k]), 0.0, 0.0])   # "decide on a torque"

        alpha = I_inv @ (tau - np.cross(omega, I @ omega))     # Euler's equation
        a_cm = np.zeros(3) if in_orbit else Accelerometer.gravity_in_body(q)

        meas_log[k], true_log[k] = accel.measure(omega, alpha, a_cm)

        omega = omega + alpha * dt                              # w_{k+1}
        q = q + quat_derivative(q, omega) * dt                  # q_{k+1}
        q /= np.linalg.norm(q)                                  # keep unit length

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        for i, ax in enumerate(axes):
            ax.plot(t, meas_log[:, i], lw=0.6, alpha=0.6, label="measured")
            ax.plot(t, true_log[:, i], lw=1.5, label="true")
            ax.set_ylabel(f"a_{'xyz'[i]} [m/s²]")
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Accelerometer: true vs measured"
                     + (" (orbit, free fall)" if in_orbit else " (ground, with gravity)"))
        fig.tight_layout()
        plt.show()

    return t, true_log, meas_log


if __name__ == "__main__":
    run_demo(in_orbit=True)
    run_demo(in_orbit=False)