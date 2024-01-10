"""Discrete-time linear Gaussian dynamical system specifications and delay augmentation."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearGaussianSystem:
    """Discrete-time linear Gaussian state-space model:

    x_{t+1} = A x_t + B u_t + w_t,  w_t ~ N(0, W)
    y_t     = C x_t + v_t,          v_t ~ N(0, V)
    Cost:   sum_{t=0}^{N-1} (x_t^T Q x_t + u_t^T R u_t) + x_N^T Q_f x_N
    """
    A: np.ndarray  # [state_dim, state_dim]
    B: np.ndarray  # [state_dim, action_dim]
    C: np.ndarray  # [obs_dim, state_dim]
    Q: np.ndarray  # [state_dim, state_dim]
    R: np.ndarray  # [action_dim, action_dim]
    Q_f: np.ndarray  # [state_dim, state_dim]
    W: np.ndarray  # Process noise covariance [state_dim, state_dim]
    V: np.ndarray  # Observation noise covariance [obs_dim, obs_dim]
    actuator_delay: int = 0  # Number of delay steps

    def __post_init__(self) -> None:
        state_dim = self.A.shape[0]
        action_dim = self.B.shape[1]
        obs_dim = self.C.shape[0]

        if self.A.shape != (state_dim, state_dim):
            raise ValueError(f"A must be square, got {self.A.shape}")
        if self.B.shape != (state_dim, action_dim):
            raise ValueError(f"B shape {self.B.shape} incompatible with state_dim={state_dim}")
        if self.C.shape != (obs_dim, state_dim):
            raise ValueError(f"C shape {self.C.shape} incompatible with state_dim={state_dim}")
        if self.Q.shape != (state_dim, state_dim):
            raise ValueError(f"Q shape {self.Q.shape} incompatible with state_dim={state_dim}")
        if self.R.shape != (action_dim, action_dim):
            raise ValueError(f"R shape {self.R.shape} incompatible with action_dim={action_dim}")
        if self.Q_f.shape != (state_dim, state_dim):
            raise ValueError(f"Q_f shape {self.Q_f.shape} incompatible with state_dim={state_dim}")
        if self.W.shape != (state_dim, state_dim):
            raise ValueError(f"W shape {self.W.shape} incompatible with state_dim={state_dim}")
        if self.V.shape != (obs_dim, obs_dim):
            raise ValueError(f"V shape {self.V.shape} incompatible with obs_dim={obs_dim}")
        if self.actuator_delay < 0:
            raise ValueError(f"actuator_delay must be >= 0, got {self.actuator_delay}")

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def action_dim(self) -> int:
        return self.B.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.C.shape[0]

    def is_controllable(self) -> bool:
        """Check controllability via the Kalman controllability matrix rank."""
        n = self.state_dim
        ctrb_blocks = [self.B]
        cur_ab = self.B
        for _ in range(1, n):
            cur_ab = self.A @ cur_ab
            ctrb_blocks.append(cur_ab)
        ctrb = np.hstack(ctrb_blocks)
        rank = np.linalg.matrix_rank(ctrb)
        return bool(rank == n)

    def is_observable(self) -> bool:
        """Check observability via the Kalman observability matrix rank."""
        n = self.state_dim
        obsv_blocks = [self.C]
        cur_ca = self.C
        for _ in range(1, n):
            cur_ca = cur_ca @ self.A
            obsv_blocks.append(cur_ca)
        obsv = np.vstack(obsv_blocks)
        rank = np.linalg.matrix_rank(obsv)
        return bool(rank == n)

    def augment_for_delay(self) -> "LinearGaussianSystem":
        """Construct state-augmented linear system accounting for actuator delay d >= 1."""
        d = self.actuator_delay
        if d == 0:
            return self

        n = self.state_dim
        m = self.action_dim
        p = self.obs_dim
        aug_dim = n + d * m

        # Augmented state: x_aug = [x_t, u_{t-d}, u_{t-d+1}, ..., u_{t-1}]^T
        A_aug = np.zeros((aug_dim, aug_dim), dtype=self.A.dtype)
        A_aug[:n, :n] = self.A
        A_aug[:n, n : n + m] = self.B  # Delayed action takes effect

        for k in range(d - 1):
            row_idx = n + k * m
            col_idx = n + (k + 1) * m
            A_aug[row_idx : row_idx + m, col_idx : col_idx + m] = np.eye(m, dtype=self.A.dtype)

        B_aug = np.zeros((aug_dim, m), dtype=self.B.dtype)
        B_aug[n + (d - 1) * m : aug_dim, :] = np.eye(m, dtype=self.B.dtype)

        C_aug = np.zeros((p, aug_dim), dtype=self.C.dtype)
        C_aug[:, :n] = self.C

        Q_aug = np.zeros((aug_dim, aug_dim), dtype=self.Q.dtype)
        Q_aug[:n, :n] = self.Q

        Q_f_aug = np.zeros((aug_dim, aug_dim), dtype=self.Q_f.dtype)
        Q_f_aug[:n, :n] = self.Q_f

        W_aug = np.zeros((aug_dim, aug_dim), dtype=self.W.dtype)
        W_aug[:n, :n] = self.W

        return LinearGaussianSystem(
            A=A_aug,
            B=B_aug,
            C=C_aug,
            Q=Q_aug,
            R=self.R,
            Q_f=Q_f_aug,
            W=W_aug,
            V=self.V,
            actuator_delay=0,
        )


def make_damped_harmonic_oscillator(
    mass: float = 1.0,
    spring_k: float = 2.0,
    damping_c: float = 0.5,
    dt: float = 0.05,
    process_noise_std: float = 0.01,
    obs_noise_std: float = 0.02,
    actuator_delay: int = 0,
) -> LinearGaussianSystem:
    """Create a physically grounded discrete-time 1D damped harmonic oscillator."""
    # Continuous dynamics: dx/dt = v, dv/dt = (-k*x - c*v + u)/m
    # Matrix exponential / Euler discretization:
    A_c = np.array([[0.0, 1.0], [-spring_k / mass, -damping_c / mass]], dtype=np.float64)
    B_c = np.array([[0.0], [1.0 / mass]], dtype=np.float64)

    # First-order discretization:
    A = np.eye(2, dtype=np.float64) + A_c * dt
    B = B_c * dt
    C = np.array([[1.0, 0.0]], dtype=np.float64)  # Position observed only

    Q = np.diag([5.0, 0.5]).astype(np.float64)
    R = np.array([[0.1]], dtype=np.float64)
    Q_f = np.diag([10.0, 1.0]).astype(np.float64)

    W = np.diag([process_noise_std**2 * dt, (process_noise_std * 2)**2 * dt]).astype(np.float64)
    V = np.array([[obs_noise_std**2]], dtype=np.float64)

    return LinearGaussianSystem(
        A=A,
        B=B,
        C=C,
        Q=Q,
        R=R,
        Q_f=Q_f,
        W=W,
        V=V,
        actuator_delay=actuator_delay,
    )
