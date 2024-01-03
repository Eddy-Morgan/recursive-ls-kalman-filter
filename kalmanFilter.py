import numpy as np

class KalmanFilter():
    #x0 - initial guess of the state vector
    #P0 - initial guess of the covariance matrix of the estimation error
    #A,B,C - system matrices describing the system model
    # Q - covariance matrix of the process noise
    # R - covariance matrix of the measurement noise

    def __init__(self, x0:np.ndarray, P0:np.ndarray, A:np.ndarray, B:np.ndarray, C:np.ndarray, Q:np.ndarray, R:np.ndarray) -> None:
        self.x0 = x0
        self.P0 = P0
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        self.currentTimeStep = 0

        self.estimates_aposteriori = []
        self.estimates_aposteriori.append(x0)

        self.estimates_apriori = []

        self.estimationErrorCovarianceMatricesAposteriori = []
        self.estimationErrorCovarianceMatricesAposteriori.append(P0)

        self.estimationErrorCovarianceMatricesApriori = []

        self.GainMatrices = []

        self.errors  = []

    def propagateDynamics(self, inputValue):
        xk_minus = self.A@self.estimates_aposteriori[self.currentTimeStep] + self.B@inputValue
        Pk_minus = self.A@self.estimationErrorCovarianceMatricesAposteriori[self.currentTimeStep]@self.A.T + self.Q

        self.estimates_apriori.append(xk_minus)
        self.estimationErrorCovarianceMatricesApriori.append(Pk_minus)

        self.currentTimeStep +=1

    def computeAposteriorEstimate(self, currentMeasurement):
        #gain matrix
        Kk = self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]@self.C.T@np.linalg.inv(self.R + self.C@self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]@self.C.T)
        
        #prediction error
        error_k = currentMeasurement - self.C@self.estimates_apriori[self.currentTimeStep-1]

        #a posterior estimate
        xk_plus = self.estimates_apriori[self.currentTimeStep-1] + Kk@error_k

        # a posterior error covariance matrix
        Pk_plus = (np.identity(np.size(self.x0)) - Kk@self.C)@self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]

        self.estimates_aposteriori.append(xk_plus)
        self.GainMatrices.append(Kk)
        self.errors.append(error_k)
        self.estimationErrorCovarianceMatricesAposteriori.append(Pk_plus)