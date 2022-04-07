import numpy as np

class KalmanFilter():

    def __init__(self, timestep, x_init, P_init, R_init):
        self.x = x_init
        self.P = P_init
        self.R = R_init
        self.F = construct_state_transition_matrix(timestep)
        self.Q = construct_process_noise_matrix(self.F, var_a)
        self.H = construct_observation_matrix()

    # initialization equations
    def construct_state_transition_matrix(self, t):
        pass

    def construct_process_noise_matrix(self, F, var_a):
        pass

    def construct_observation_matrix(self):
        pass

    # TODO: figure out if we should update R

    # update equations
    def state_extrapolation(self):
        # TODO: determine if we need w_n
        pass

    def covariance_extrapolation(self):
        pass

    def compute_kalman_gain(self):
        pass

    def state_update(self, z):
        # z = our measurement
        pass

    def covariance_update(self):
        pass
