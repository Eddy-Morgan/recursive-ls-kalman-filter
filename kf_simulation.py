import numpy as np
import matplotlib.pyplot as plt
from kalmanFilter import KalmanFilter

#discretization_step
h= 0.1

# initial value for the simulation
initialPosition=10
initialVelocity=-5
acceleration = 0.5

# measurement noise standard deviation
noiseStd =1

#number of discretization steps
numberTimeSteps =100

# define the system matrices - newtonian system
A = np.array([[1,h,0.5*(h**2)],[0,1,h],[0,0,1]])
B = np.array([[0],[0],[0]])
C = np.array([[1,0,0]])

R = 1*np.array([[1]])
Q = np.zeros((3,3))

# guess initial estimate
x0 = np.array([[0],[0],[0]])
# initial covariance
P0 = 1*np.eye(3)


timeVector = np.linspace(0, (numberTimeSteps-1)*h, numberTimeSteps)

# vectors to store simulation states
position=np.zeros(np.size(timeVector))
velocity=np.zeros(np.size(timeVector))

# simulation system behavior
for i in np.arange(np.size(timeVector)):
    position[i] = initialPosition + initialVelocity*timeVector[i] + (acceleration*timeVector[i]**2)/2
    velocity[i] = initialVelocity + acceleration*timeVector[i]

# add the measurement noise
positionNoisy = position+noiseStd*np.random.randn(np.size(timeVector))

#verify the position vector by plotting the results
plotStep = numberTimeSteps//2
plt.plot(timeVector[0:plotStep], position[0:plotStep], linewidth=4, label="Ideal Position")
plt.plot(timeVector[0:plotStep], positionNoisy[0:plotStep], 'r', label="Observed Position")
plt.xlabel('time')
plt.ylabel('position')
plt.legend()

KalmanFilterObject = KalmanFilter(x0, P0, A, B,C, Q,R)
inputValue = np.array([[0]])

# simulation online prediction
for j in np.arange(np.size(timeVector)):
    KalmanFilterObject.propagateDynamics(inputValue)
    KalmanFilterObject.computeAposteriorEstimate(positionNoisy[j])

#extract estimate to plot
estimates1= []
estimates2= []
estimates3= []

#create vectors corresponding to evolution of parameters
for j in np.arange(np.size(timeVector)):
    estimates1.append(KalmanFilterObject.estimates_aposteriori[j][0])
    estimates2.append(KalmanFilterObject.estimates_aposteriori[j][1])
    estimates3.append(KalmanFilterObject.estimates_aposteriori[j][2])

# create vectors corresponding to the true values in order to plot the results
estimate1true=position
estimate2true=velocity
estimate3true=acceleration*np.ones(np.size(timeVector))


# plot the results
steps=np.arange(np.size(timeVector))
fig, ax = plt.subplots(3,1,figsize=(10,15))
ax[0].plot(steps,estimate1true,color='red',linestyle='-',linewidth=6,label='True value of position')
ax[0].plot(steps,estimates1,color='blue',linestyle='-',linewidth=3,label='True value of position')
ax[0].set_xlabel("Discrete-time steps k",fontsize=14)
ax[0].set_ylabel("Position",fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[0].set_ylim(98,102)  
ax[0].grid()
ax[0].legend(fontsize=14)

ax[1].plot(steps,estimate2true,color='red',linestyle='-',linewidth=6,label='True value of velocity')
ax[1].plot(steps,estimates2,color='blue',linestyle='-',linewidth=3,label='Estimate of velocity')
ax[1].set_xlabel("Discrete-time steps k",fontsize=14)
ax[1].set_ylabel("Velocity",fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,2)  
ax[1].grid()
ax[1].legend(fontsize=14)

ax[2].plot(steps,estimate3true,color='red',linestyle='-',linewidth=6,label='True value of acceleration')
ax[2].plot(steps,estimates3,color='blue',linestyle='-',linewidth=3,label='Estimate of acceleration')
ax[2].set_xlabel("Discrete-time steps k",fontsize=14)
ax[2].set_ylabel("Acceleration",fontsize=14)
ax[2].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,2)  
ax[2].grid()
ax[2].legend(fontsize=14)



plt.show()
