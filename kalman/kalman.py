import numpy as np

class KalmanFilter():

    # sample initialization:
    # x_init = np.zeros(6)
    # P_init = np.diag(np.full(6, 500))
    # R_init = np.array([[9,0],[0,9]])
    # kf = KalmanFilter(1, x_init, P_init, R_init, 0.2**2)
    def __init__(self, timestep, x_init, P_init, R_init, var_a, gain=1.):
        self.x = x_init
        self.P = P_init
        self.R = R_init
        self.F = self.construct_state_transition_matrix(timestep)
        self.Q = self.construct_process_noise_matrix(self.F, var_a)
        self.H = self.construct_observation_matrix()
        self.gain = gain

    # From https://www.kalmanfilter.net/multiExamples.html
    def construct_state_transition_matrix(self, t):
        return np.array([
        [1, t, 0.5*t**2, 0, 0, 0],
        [0, 1, t, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, t, 0.5*t**2],
        [0, 0, 0, 0, 1, t],
        [0, 0, 0, 0, 0, 1]
        ])

    # From https://www.kalmanfilter.net/multiExamples.html
    def construct_process_noise_matrix(self, t, var_a):
        return np.array([
        [t**4/4, t**3/2, t**2/2, 0, 0, 0],
        [t**3/2, t**2, t, 0, 0, 0],
        [t**2/2, t, 1, 0, 0, 0],
        [0, 0, 0, t**4/4, t**3/2, t**2/2],
        [0, 0, 0, t**3/2, t**2, t],
        [0, 0, 0, t**2/2, t, 1]
        ])*var_a

    # From https://www.kalmanfilter.net/multiExamples.html
    def construct_observation_matrix(self):
        return np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
        ])

    # returns transition matrix times position vector
    def state_extrapolation(self):
        x_hat = np.matmul(self.F, self.x)
        return x_hat

    # extrapolates covariance from transition matrix and covarience
    def covariance_extrapolation(self):
        return np.matmul(self.F, np.matmul(self.P, self.F.T))

    # computes kalman gain
    def compute_kalman_gain(self):
        return np.matmul(self.P, np.matmul(self.H.T, np.linalg.inv(np.matmul(self.H, np.matmul(self.P, self.H.T)) + self.R)))

    # updates location vector based on new observation 'z'
    def state_update(self, z):
        return self.x + np.matmul(self.compute_kalman_gain(), (z - np.matmul(self.H, self.x)))

    # updates covariance
    def covariance_update(self):
        K = self.compute_kalman_gain()*self.gain
        P = np.matmul(np.eye(K.shape[0]) - np.matmul(K, self.H), np.matmul(self.P, (np.eye(K.shape[0]) - np.matmul(K, self.H)).T)) + np.matmul(K, np.matmul(self.R, K.T))
        if np.sum(np.isnan(P)) > 0:
            print('yikes')
            return self.P
        else: return P

    # takes in a new observation and performs one 'iteration' of the kalman filter
    def run(self, z):

        self.x = self.state_update(z)

        self.P = self.covariance_update()

        self.x = self.state_extrapolation()

        self.P = self.covariance_extrapolation()

        return self.x[0], self.x[3]

    # extrapolates one kalman filter iteration without a new observation
    def predict(self):

        self.x = self.state_extrapolation()

        self.P = self.covariance_extrapolation()

        return self.x[0], self.x[3]
