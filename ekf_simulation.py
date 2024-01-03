import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from extendedKalmanFilter import ExtendedKalmanFilter


deltaTime = 0.01

x0 = np.array([np.pi/3 , 0.2])

simulationSteps = 400

totalSimulationTimeVector = np.arange(0, deltaTime*simulationSteps, deltaTime)

def stateSpaceModel(x, t):
    g = 9.81
    l =1 
    dxdt = np.array([x[1],-(g/l)*np.sin(x[0])])
    return dxdt

solutionOde = odeint(stateSpaceModel,x0, totalSimulationTimeVector)

plt.plot(totalSimulationTimeVector, solutionOde[:,0], 'b', label="x1")
plt.plot(totalSimulationTimeVector, solutionOde[:,1], 'r', label="x2")
plt.xlabel('time')
plt.ylabel('x1(t), x2(t)')
plt.legend(loc='best')
plt.show()

