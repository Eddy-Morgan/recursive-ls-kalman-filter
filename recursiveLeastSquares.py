import numpy as np
class RecursiveLeastSquares():
    #  x0 - initial estimate used to initialize the estimator
    #  P0 - initial estimation error covariance matrix
    #  R - covariance matrix of the measurement noise
    def __init__(self, x0:np.ndarray, P0:np.ndarray, R:np.ndarray) -> None:
        self.x0 = x0
        self.P0 = P0
        self.R = R

        self.currentTimeStep = 0

        self.estimates = []
        self.estimates.append(x0)

        self.estimationCovarianceMatrices = []
        self.estimationCovarianceMatrices.append(P0)

        self.gainMatrices = []

        self.errors = []

    def predict(self, y, C:np.ndarray):
        L_k = self.R + C@self.estimationCovarianceMatrices[self.currentTimeStep]@C.T
        L_k_inv = np.linalg.inv(L_k)
        #compute gain matrix
        gainMatrix = self.estimationCovarianceMatrices[self.currentTimeStep]@C.T@L_k_inv

        #compute correction
        error = y - C@self.estimates[self.currentTimeStep]

        #new estimate
        estimate = self.estimates[self.currentTimeStep] + gainMatrix@error

        #propagate the estimation error covariance 
        estimationCovarianceMatrix = (np.identity(np.size(self.x0)) - gainMatrix@C)@self.estimationCovarianceMatrices[self.currentTimeStep]

        self.estimates.append(estimate)
        self.estimationCovarianceMatrices.append(estimationCovarianceMatrix)
        self.gainMatrices.append(gainMatrix)
        self.errors.append(error)


        self.currentTimeStep += 1