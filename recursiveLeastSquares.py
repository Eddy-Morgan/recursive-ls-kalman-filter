import numpy as np
class RecursiveLeastSquares():
    #  x0 - initial estimate used to initialize the estimator
    #  P0 - initial estimation error covariance matrix
    #  R - covariance matrix of the measurement noise
    def __init__(self, x0:np.ndarray, P0:np.ndarray, R:np.ndarray) -> None:
        self.x0 = x0
        self.P0 = P0
        self.R = R

        self.t_step = 0

        self.x_k = []
        self.x_k.append(x0)

        self.P_k = []
        self.P_k.append(P0)

        self.K_k = []

        self.errors = []

    def predict(self, y, C:np.ndarray):
        L_k = self.R + C@self.P_k[self.t_step]@C.T
        L_k_inv = np.linalg.inv(L_k)
        #compute gain matrix
        K = self.P_k[self.t_step]@C.T@L_k_inv

        #compute correction
        error = y - C@self.x_k[self.t_step]

        #new estimate
        estimate = self.x_k[self.t_step] + K@error

        #propagate the estimation error covariance 
        P = (np.identity(np.size(self.x0)) - K@C)@self.P_k[self.t_step]

        self.x_k.append(estimate)
        self.P_k.append(P)
        self.K_k.append(K)
        self.errors.append(error)


        self.t_step += 1