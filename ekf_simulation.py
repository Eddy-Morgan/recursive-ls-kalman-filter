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

# plt.plot(totalSimulationTimeVector, solutionOde[:,0], 'b', label="x1")
# plt.plot(totalSimulationTimeVector, solutionOde[:,1], 'r', label="x2")
# plt.xlabel('time')
# plt.ylabel('x1(t), x2(t)')
# plt.legend(loc='best')


forwardEulerState =np.zeros(shape=(simulationSteps,2))

forwardEulerState[0,0]=x0[0]
forwardEulerState[0,1]=x0[1]

for timeIndex in range(simulationSteps-1):
    forwardEulerState[timeIndex+1,:] = forwardEulerState[timeIndex,:] + deltaTime*stateSpaceModel(forwardEulerState[timeIndex,:],timeIndex*deltaTime)

# plot the comparison results
plt.plot(totalSimulationTimeVector, solutionOde[:, 0], 'r', linewidth=3, label='Angle - ODEINT')
plt.plot(totalSimulationTimeVector, forwardEulerState[:, 0], 'b', linewidth=2, label='Angle- Forward Euler')
plt.legend(loc='best')
plt.xlabel('time [s]')
plt.ylabel('Angle-x1(t)')
plt.grid()


x0guess = np.zeros((2,1))
x0guess[0] = x0[0]+4*np.random.randn()
x0guess[1] = x0[1]+4*np.random.randn()

# initial covariance matrix
P0 = 10*np.eye(2,2)

# discretization_step
dT= deltaTime

Q = 0.0001*np.eye(2,2)

R = np.array([[0.0001]])

extendedKF_object = ExtendedKalmanFilter(x0guess, P0, Q, R, dT)

# simulate online prediction using ekf
for j in np.arange(simulationSteps-1):
    extendedKF_object.propagateDynamics()
    print(solutionOde[j,0])
    extendedKF_object.computeAposteriorEstimate(solutionOde[j,0])

# extract the state estimates in order to plot the results
estimateAngle=[]
estimateAngularVelocity=[]
for j in np.arange(np.size(totalSimulationTimeVector)):
    estimateAngle.append(extendedKF_object.estimates_aposteriori[j][0,0])
    estimateAngularVelocity.append(extendedKF_object.estimates_aposteriori[j][1,0])
   
    
# create vectors corresponding to the true values in order to plot the results
trueAngle=solutionOde[:,0]
trueAngularVelocity=solutionOde[:,1]


# plot the results
steps=np.arange(np.size(totalSimulationTimeVector))
fig, ax = plt.subplots(2,1,figsize=(10,15))
ax[0].plot(steps,trueAngle,color='red',linestyle='-',linewidth=6,label='True angle')
ax[0].plot(steps,estimateAngle,color='blue',linestyle='-',linewidth=3,label='Estimate of angle')
ax[0].set_xlabel("Discrete-time steps k",fontsize=14)
ax[0].set_ylabel("Angle",fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
ax[0].grid()
ax[0].legend(fontsize=14)

ax[1].plot(steps,trueAngularVelocity,color='red',linestyle='-',linewidth=6,label='True angular velocity')
ax[1].plot(steps,estimateAngularVelocity,color='blue',linestyle='-',linewidth=3,label='Angular velocity estimate')
ax[1].set_xlabel("Discrete-time steps k",fontsize=14)
ax[1].set_ylabel("Angular Velocity",fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
ax[1].grid()
ax[1].legend(fontsize=14)
plt.show()


