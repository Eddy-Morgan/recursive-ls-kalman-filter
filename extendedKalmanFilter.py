import numpy as np

# extended kalman filter for a simple nonlinear pendulum dynamics

class ExtendedKalmanFilter():
    #x0 - initial guess of the state vector
    #P0 - initial guess of the covariance matrix of the estimation error
    # Q - covariance matrix of the process noise
    # R - covariance matrix of the measurement noise
    # dT - discretization period for the forward euler method

    def __init__(self, x0:np.ndarray, P0:np.ndarray, Q:np.ndarray, R:np.ndarray, dT) -> None:
        self.x0 = x0
        self.P0 = P0
        self.Q = Q
        self.R = R
        self.dT = dT

        # model parameter

        # gravitational constant
        self.g = 9.81

        #length of pendulum
        self.l = 1

        self.currentTimeStep = 0
        self.estimates_aposteriori = []
        self.estimates_aposteriori.append(x0)

        self.estimates_apriori = []

        self.estimationErrorCovarianceMatricesAposteriori = []
        self.estimationErrorCovarianceMatricesAposteriori.append(P0)

        self.estimationErrorCovarianceMatricesApriori = []

        self.GainMatrices = []

        self.errors  = []

    def stateSpaceContinuous(self,x,t):
        dxdt = np.array([[x[1,0]],[-(self.g/self.l)*np.sin(x[0,0])]])
        return dxdt
    
    def discreteTimeDynamics(self, x_k):
        x_kp1 = x_k+self.dT*self.stateSpaceContinuous(x_k, self.currentTimeStep*self.dT)
        return x_kp1
    
    def jacobianStateEquation(self, x_k):
        A = np.zeros(shape=(2,2))
        A[0,0] = 1
        A[1,0] = self.dT
        A[0,1] = self.dT*(self.g/self.l)*np.cos(x_k[0,0])
        A[1,1] = 1
        return A

    def jacobianOutputEquation(self, x_k):
        C = np.zeros(shape=(1,2))
        C[0,0] = 1
        return C
    
    def outputEquation(self,x_k):
        return x_k[0]
    
    def propagateDynamics(self):
        xk_minus = self.discreteTimeDynamics(self.estimates_aposteriori[self.currentTimeStep])
        Akm1 = self.jacobianStateEquation(self.estimates_aposteriori[self.currentTimeStep])
        Pk_minus = Akm1@self.estimationErrorCovarianceMatricesAposteriori[self.currentTimeStep]@Akm1.T + self.Q

        self.estimates_apriori.append(xk_minus)
        self.estimationErrorCovarianceMatricesApriori.append(Pk_minus)
        self.currentTimeStep +=1

    def computeAposteriorEstimate(self, currentMeasurement):
        Ck = self.jacobianOutputEquation(self.estimates_aposteriori[self.currentTimeStep])
        #gain matrix
        Kk = self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]@Ck.T@np.linalg.inv(self.R + Ck@self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]@Ck.T)
        
        #prediction error
        error_k = currentMeasurement - self.outputEquation(self.estimates_apriori[self.currentTimeStep-1])

        #a posterior estimate
        xk_plus = self.estimates_apriori[self.currentTimeStep-1] + Kk@error_k

        # a posterior error covariance matrix
        Pk_plus = (np.identity(np.size(self.x0)) - Kk@Ck)@self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]

        self.estimates_aposteriori.append(xk_plus)
        self.GainMatrices.append(Kk)
        self.errors.append(error_k)
        self.estimationErrorCovarianceMatricesAposteriori.append(Pk_plus)