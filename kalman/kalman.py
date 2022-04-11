import numpy as np

class KalmanFilter():

    def __init__(self, timestep, x_init, P_init, R_init, var_s):
        self.x = x_init
        self.P = P_init
        self.R = R_init
        self.F = construct_state_transition_matrix(timestep)
        self.Q = construct_process_noise_matrix(self.F, var_a)
        self.H = construct_observation_matrix()

    # initialization equations
    def construct_state_transition_matrix(self, t):
        return np.array([
        [1, t, 0.5*t**2, 0, 0, 0],
        [0, 1, t, 0, 0, 0],
        [0, 0, 1, 0, 0, 0].
        [0, 0, 0, 1, t, 0.5*t**2],
        [0, 0, 0, 0, 1, t],
        [0, 0, 0, 0, 0, 1]
        ])

    def construct_process_noise_matrix(self, F, var_a):
        Q_a = np.array([
        [t**4/4, t**3/2, t**2/2, 0, 0, 0],
        [t**3/2, t**2, t, 0, 0, 0],
        [t**2/2, t, 1, 0, 0, 0],
        [0, 0, 0, t**4/4, t**3/2, t**2/2],
        [0, 0, 0, t**3/2, t**2, t],
        [0, 0, 0, t**2/2, t, 1]
        ])*var_a

        return np.matmul(np.matmul(F, Q_a), F.T)

    def construct_observation_matrix(self):
        return np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
        ])

    # update equations
    def state_extrapolation(self):
        # TODO: determine if we need w_n
        x_hat = np.matmul(self.F, self.x)
        return x_hat

    def covariance_extrapolation(self):
        return np.matmul(self.F, np.matmul(self.P, self.F.T))

    def compute_kalman_gain(self):
        return np.matmul(self.P, np.matmul(self.H.T, np.linalg.inv(np.matmul(self.H, np.matmul(self.P, self.H.T) + self.R))))

    def state_update(self, z):
        return self.x + np.matmul(self.compute_kalman_gain(), (z - np.matmul(self.H, self.x)))

    def covariance_update(self):
        K = self.compute_kalman_gain()
        return np.matmul((np.eye(K.shape[0]) - np.matmul(K, self.H), np.matmul(self.P, (np.eye(K.shape[0]) - np.matmul(K, self.H)).T))) + np.matmul(K, np.matmul(self.R, K.T))

    def run(self, z):

        compute_kalman_gain()

        state_update(z)

        covariance_update()

        state_extrapolation()

        covariance_extrapolation()
