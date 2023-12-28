# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: IRS_Virtual
#     language: python
#     name: irs_virtual
# ---

# %% [markdown]
# ### Maximally Ergodic Trajectory Over a Distribution
# Compute the maximally ergodic trajectory for the system 
#
# $$
# \dot{x} 
# = \begin{bmatrix}
# u_1 \\
# u_2 
# \end{bmatrix}
# $$
# with respect to the normal distribution
# $$
# \phi(x) = \det(2\pi\Sigma)^{-0.5}\exp(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x-\mu)) = \mathcal{N}(x;\mu,\Sigma)
# $$
# where $\mu = 0$ , 
# $
# \Sigma = \begin{bmatrix}
# 2 & 0 \\
# 0 & 2
# \end{bmatrix}
# $
# , and 
# $ 
# x(0) = \begin{bmatrix}
# 0  \\
# 1 
# \end{bmatrix} 
# $ .  
#
# The time horizon is T = 10s. 

# %% [markdown]
# ### Equation
#
# $\xi = (x(t), u(t))$
#
# $$
# J(x(t), u(t)) = q \sum^K_{k_1 = 0} ... \sum^K_{k_n = 0} \Lambda_k 
# \left( 
#     \frac{1}{T} \int^T_0 F_k(x(t))dt - \phi_k 
# \right)^2 + \int^T_0 u(t)^TRu(t)dt
# $$

# %% [markdown]
# $$
# \frac{d}{d\epsilon}J(\xi + \epsilon \zeta)\vert_{\epsilon = 0}
# = 
# \frac{d}{d\epsilon} 
# \left[
# q \sum^K_{k = 0} \Lambda_k 
#     \left( 
#     \frac{1}{T} \int^T_0 F_k(x(s) + \epsilon z(s))dt - \phi_k 
#     \right)^2 
#     + \int^T_0 (u(t) + \epsilon v(t))^T R (u(t) + \epsilon v(t))dt
# \right]_{\epsilon = 0}
# $$
#
#
# $$
# DJ(\xi) \cdot \zeta = q \sum^K_{k = 0} \Lambda_k 
# \left[
#     2
#     \left( 
#     \frac{1}{T} \int^T_0 F_k(x(s))ds - \phi_k 
#     \right) \cdot + \int^T_0 \frac{1}{T} \int^T_0 DF_k(x(t))z(t)dt
# \right]
#
# + \int^T_0 u(t)^T R v(t)dt
#
# $$

# %%
#import the necessary packages
import numpy as np
from numpy.linalg import det 
from numpy.linalg import inv 
from numpy.linalg import norm

import math
from math import cos 
from math import sin
from math import pi
from math import exp

from scipy.integrate import simpson

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
def A(t): 
    #x, y, theta, u1, u2 = state_dict[t]
    return np.array([[0, 0 ],
                     [0, 0 ]
                     ]) 
                     
def B(t): 
    #x, y, theta, u1, u2 = state_dict[t]
    return np.array([[1, 0],
                     [0, 1]
                     ]) 

def euler(P1, Q, R, t, dt):
    #Create the P array based on the time size.
    P = np.ndarray((4, t.size))

    #Assign the terminal condition to the end of the array. 
    P[:,-1] = P1.flatten()

    #Find the slope and work backwards using Euler's integration. 
    for i in range(t.size-1, 0, -1):
        #Find the current P matrix that we have and reshape it for matmul.
        cP = P[:,i].reshape(2,2)

        #Find the slope.
        Pdot = cP @ A(t[i]) + A(t[i]).T @ cP - cP @ B(t[i]) @ inv(R) @ B(t[i]).T @ cP + Q

        #Perform Euler's Inegration and update the next step.
        P[:, i-1] = (cP + Pdot * dt).flatten()

    return P


# %%
def fourier(k, L, x, n):
    '''
    Fourier will take the approximation of a singular n-dimensional k, using the basis function.

    Parameters
    ----------
    k : np.array
        This a singular n-dimensional index for Fourier coefficients.
    L: np.array
        Upper and lower bounds of an n-dimensional search space.
    x: np.array
        Trajectory point.
    n: int
        Number of dimensions in search space. 
    
    Returns
    -------
    prod: float
        Value of the Fourier Basis Function at that index, w.r.t trajectory point.
    '''
    #make sure that k is the same dimensionality as n 
    #also makes sure that k is a numpy array.
    if np.shape(k)[0] != n:
        raise TypeError("Wanted k to be np.array of size n")
    
    if np.shape(x)[0] != n:
        raise TypeError("Wanted x to be np.array of size n")
    
    if np.shape(L) != (2,n):
        raise TypeError("Wanted L to be np.array of size n")
    
    #find h_k. 
    h_k = 1 

    #find the product over the n-dimensional space based on the given
    #k and x(t)
    prod = 1 

    for i in range(n):
        prod *= cos((k[i] * pi)/(L[1, i] - L[0, i]) * (x[i] - L[0, i]))
    
    #return the product normalized by the h_k facotr.
    prod = 1/h_k * prod
    
    return prod


# %%
def DFk(k, x, L, n):
    '''
    Derivative Fourier will take the approximation of a singular n-dimensional k, using the basis function.

    Parameters
    ----------
    k : np.array
        This a singular n-dimensional index for Fourier coefficients.
    L: np.array
        Upper and lower bounds of an n-dimensional search space.
    x: np.array
        Trajectory point.
    n: int
        Number of dimensions in search space. 
    
    Returns
    -------
    vec: np.array
        n-dimensional array of the derivative of the Fourier Basis.
    '''
    #Check any type errors.
    if np.shape(k)[0] != n:
        raise TypeError("Wanted k to be np.array of size n")
    
    if np.shape(x)[0] != n:
        raise TypeError("Wanted x to be np.array of size n")
    
    if np.shape(L) != (2,n):
        raise TypeError("Wanted L to be np.array of size n")
    
    #find h_k 
    h_k = 1 

    #create our storage vector. 
    vec = np.ndarray((n,))

    vec[0] = -1/h_k * sin((k[1] * pi)/(L[1, 0] - L[0, 0]) * (x[0] - L[0, 0])) * cos((k[1] * pi)/(L[1, 1] - L[0, 1]) * (x[1] - L[0, 1]))
    vec[1] = -1/h_k * cos((k[1] * pi)/(L[1, 0] - L[0, 0]) * (x[0] - L[0, 0])) * cos((k[1] * pi)/(L[1, 1] - L[0, 1]) * (x[1] - L[0, 1]))
    
    #This will return a single n-deimvector 
    return vec


# %%

# %%
def a(bounds, trajectory, time, dt, phi_k, q):
    ##Calculation of the 
    #create our ck array.
    n, m = phi_k.shape

    ck = np.zeros((n, m))

    #create an indexable array of Fourier Coefficients.
    k = np.array([n,m])
    rows, cols = np.indices((n, m))

    #find ck using Fourier Basis Functions
    for ii in range(trajectory.shape[1]):
        #iterate over the indices that we have.
        for i in range(n):
            for j in range(m):
                #fourier only accepts numpy arrays so we convert into an index.
                index = np.array([rows[i,j],cols[i,j]])

                #find the ck matrix for the given timestep
                ck[i, j] += fourier(index, bounds, trajectory[:, ii], 2) * dt

    #finsih the integral
    ck = 1/time[-1] * ck    

    #define the ergodic metric
    lamb = (1 + norm(k)**2)**(-1 * ((2 + 1) / 2)) 
    diff = np.subtract(ck, phi_k)
    ans = 2 * q * np.sum(lamb * diff * 1/time[-1] * 

    ) 
    
    
    return ans 

def b(t,R): 
    state = state_dict[t]
    #print(state[3:5,None])
    return 2 * R @ state[2:4,None] 

def r_integrate(r0, P, t, dt):
    #Create the r array based on the time size. 
    r = np.ndarray((2,t.size))

    #Assign the terminal condition to the end of the array. 
    r[:,-1] = r0

    #Find the slope and work backwards using Euler's integration. 
    for i in range(t.size-1, 0, -1):
        #Find the current r matrix that we have and reshape it for matmul.
        cr = r[:,i,None]

        #Find the slope.
        rdot = (A(t[i]) - B(t[i]) @ inv(R) @ B(t[i]).T @ P[:,i].reshape(2,2)).T @ cr + a(t[i],Q) - P[:,i].reshape(2,2) @ B(t[i]) @ inv(R) @ b(t[i],R)

        #Perform Euler's Inegration and update the next step.
        r[:, i-1] = (cr + rdot * dt).flatten()
    return r


# %%
def z_integrate(z0, P, r, t, dt):
    #Create the z array based on the time size. 
    z = np.ndarray((2,t.size))

    #Assign the intitial condition to the beginning of the array. 
    z[:,0] = z0

    #Find the slope and work forwards using Euler's integration. 
    for i in range(0, t.size-1, 1):
        #Find the current z matrix that we have and reshape it for matmul.
        cz = z[:,i,None]

        #Find the slope.
        zdot = A(t[i]) @ cz + B(t[i]) @ (-1 * inv(R) @ B(t[i]).T @ P[:,i].reshape(2,2) @ cz - inv(R) @ B(t[i]).T @ r[:,i,None] - inv(R) @ b(t[i],R))
        # zdot = A(t[i]) @ cz + B(t[i]) @ (-1 * inv(R) @ B(t[i]).T @ P[:,i].reshape(3,3) @ cz  - inv(R) @ b(t[i],R))

        #Perform Euler's Inegration and update the next step.
        z[:, i+1] = (cz + zdot * dt).flatten()
    return z

def v_integrate(P, r, z, t):
    #Create the z array based on the time size. 
    v = np.ndarray((2,t.size))

    #Find the v values based on the current time step. 
    for i in range(0, t.size, 1):
        #Assign the value of v. 
        v[:, i] = (-1 * inv(R) @ B(t[i]).T @ P[:,i].reshape(2,2) @ z[:,i, None] - inv(R) @ B(t[i]).T @ r[:,i,None] - inv(R) @ b(t[i],R)).flatten()
        # v[:, i] = (-1 * inv(R) @ B(t[i]).T @ P[:,i].reshape(3,3) @ z[:,i, None]  - inv(R) @ b(t[i],R)).flatten()

    return v 


# %%
def subspace(distribution):
    '''
    Get the boundaries of a subspace given a n x m distribution in n-dimensional space.

    Parameters
    ----------
    trajectory: np.array
        This is a n x m distribution in n-dimensional space over m samples
    
    Returns
    -------
    bounds: np.array
        This is a 2 x n array that descibes the limits in x_1, x_2, .. x_n dimensions. 
    '''
    #get the dimensions of the trajectory.
    n, m = distribution.shape

    #intitialize bounds to be a 2 x n matrix. 
    bounds = np.ndarray((2,n))

    #set the minimums on top and the maximums on the bottom.
    #round to nice even numbers
    bounds[0, :] = np.floor(np.amin(distribution, axis=1))
    bounds[1, :] = np.ceil(np.amax(distribution, axis=1))

    return bounds


# %%
def phi_k(boundary, mu, cv, resolution, K_f):
    '''
    Define phi_k over the bounds over a normally distributed Guassian distribution with 
    mu and covariance.

    Parameters
    ----------
    boundary: np.array
        This is a 2 x n distribution in n-dimensional space. Assumes [0,:] is minimums,
        [1,:] is maximums.

    mu: np.array
        This is a n-dimensional array of the Guassian distribution mean. 

    cv: np.array
        This is an n x n dimensional array of the Guassian distribution covariance.

    resolution: integer
        This tells us how fine our mesh should be in the distribution discretization    

    k_F: integer
        This tells us how many Fourier coefficients we will have.
    
    Returns
    -------
    pk: np.array
        This is a k_F by k_F dimensioned array of coefficients. 
    '''

    #set up our index arrays for the Fourier Coefficients.
    k = np.array([K_f, K_f])
    k_rows,k_cols = np.indices((k[0],k[1]))

    #intitialize the pk storage.
    pk = np.zeros((k[0],k[1]))

    #create the grid between the boundaries to the specified resolution.
    rangex, dx = np.linspace(boundary[0,0], boundary[1,0] + 1, resolution, retstep=True)
    rangey, dy = np.linspace(boundary[0,1], boundary[1,1] + 1, resolution, retstep=True)
    sub_x, sub_y = np.meshgrid(rangex, rangey)

    #iterate over our subspace indices.
    for ii in range(sub_x.shape[0]):        #number of rows
        for jj in range(sub_y.shape[1]):    #number of columns
            sub_index = np.array([sub_x[ii, jj], sub_y[ii, jj]])

            #find the px value for the given sub_index.
            px = det(2 * pi * cv)**-0.5 * exp(-0.5 * ((sub_index - mu).T) @ inv(cv) @ (sub_index - mu))

            #iterate over our fourier basis coefficient indices.
            for i in range(k[0]):
                for j in range(k[1]):

                    #fourier only accepts numpy arrays, so we convert into an index.
                    k_index = np.array([k_rows[i,j],k_cols[i,j]])

                    #find the pk matrix for the given timestep.         
                    pk[i,j] += px * fourier(k_index, boundary, sub_index, 2) * dy * dx

    return pk


# %%
def J(trajectory, bounds, time, dt, phi_k, q, R, u, P1):
    '''
    Define the cost of the ergodicity of a trajectory.

    Parameters
    ----------
    trajectory: np.array
        This a n x m trajectory in n-dimensional space over m timesteps

    bounds: np.array
        This is a 2 x n distribution in n-dimensional space. Assumes [0,:] is minimums,
        [1,:] is maximums.

    time: np.array
        This is a m shaped array.
    
    dt: float
        Timestep.

    phi_k: np.array
        This is a k_F by k_F dimensioned array of coefficients. 

    q: float
        This tells us how much to weigh our ergodic metric in the cost function

    R: np.array
        Must be n x n dimensions. Weighs importance control signals.

    u: np.array
        This a n x m control array in n-dimensional space over m timesteps
    
    P1: np.array
        Must be n x n dimensions.
        
    Returns
    -------
    cost: float
        A measure of the current cost based on the ergodic metric.
    '''
    #create our ck array.
    n, m = phi_k.shape

    ck = np.zeros((n, m))

    #create an indexable array of Fourier Coefficients.
    k = np.array([n,m])
    rows, cols = np.indices((n, m))

    #find ck using Fourier Basis Functions
    for ii in range(trajectory.shape[1]):
        #iterate over the indices that we have.
        for i in range(n):
            for j in range(m):
                #fourier only accepts numpy arrays so we convert into an index.
                index = np.array([rows[i,j],cols[i,j]])

                #find the ck matrix for the given timestep
                ck[i, j] += fourier(index, bounds, trajectory[:, ii], 2) * dt

    #finsih the integral
    ck = 1/time[-1] * ck    

    #define the ergodic metric
    lamb = (1 + norm(k)**2)**(-1 * ((2 + 1) / 2)) 
    epsilon = np.sum(lamb * (np.subtract(ck, phi_k))**2)

    #calculate the integral 
    integral = np.ndarray((time.size)) 
    for i in range(0, time.size, 1):
        integral[i] = u[:,i,None].T @ R @ u[:,i,None]

    #get the cost
    cost = q * epsilon + simpson(integral, time) + (0.5 * (trajectory[:,-1, None]).T @ P1 @ (trajectory[:,-1,None]))[0,0]
    return cost


# %%
##Setup our constants
Q  = np.diag([1,1])

#When r is equal to zero then Q must be equal to P1
R  = np.diag([1,1])

P1 = np.diag([1,1])

#The terminal condition for r is that it is equal to zero. 
rT = np.zeros((2,))
#The initial condition for z is that it is equal to zero.
z0 = np.zeros((2,))

# %%
#Distribution and Initialization of Constants---------------------------------
#creation of the distribution
mu = np.zeros((2))
covar= np.diag((2,2))
target = np.random.multivariate_normal(mu, covar, 100000).T
bounds = subspace(target)

#creation of our time boundary
t = 10
timestep = 1000
time, dt = np.linspace(0,t,timestep,retstep=True)

#initial control guess
control_guess = np.vstack((np.full_like(time, 1), 
                           np.full_like(time, -1)
                           ))

#intial trajectory creation
traj_guess = np.zeros((2 * time.size))
traj_guess[0:2] = np.array([0, 1])

#update our trajectory based on our control signals
for i in range(0,len(traj_guess)-2, 2):
    #calculate the states based on the control velocities.
    ii = int(i/2)
    traj_guess[i + 2] = traj_guess[i]       + dt * control_guess[0, ii]
    traj_guess[i + 3] = traj_guess[i + 1]   + dt * control_guess[1, ii]

#reshape to represent states. 
traj_guess = traj_guess.reshape(time.size, 2).T

#initial guess
initial_guess = np.concatenate((traj_guess, control_guess))
state_dict = dict(zip(time,initial_guess.T))

#calculate phi_k
pk = phi_k(bounds, mu, covar, 100, 10)

#calculate the current cost of the trajectory. 
cost = J(traj_guess, bounds, time, dt, pk, 1, R, control_guess, P1)

# %%
#Begin solving the Ricotti Equations. 
P_All = euler(P1, Q, R, time, dt)

#Begin solving for r based on the terminal condition. 
r_All = r_integrate(rT, P_All, time, dt) 

#Begin solving for z based on the initial condtion. Solve for v, once we have z. 
z_All = z_integrate(z0, P_All, r_All, time, dt)
v_All = v_integrate(P_All, r_All, z_All, time)

# %%
##iLQR Algorithm---------------------------------------------------------------------------------------------
#Initialize the constants that we need. 
alpha = 0.4 
beta = 0.7
epsilon = 10 ** -7
q = 1 

#iterators
xi = 1
j = 0 

#Create our changing trajectory.
trajectory = np.copy(traj_guess)
controls = np.copy(control_guess)

#Create an objective function list. 
ob_Arr = []
obj_Arr = []

while xi > epsilon: 
    #Begin solving the Ricotti Equations. 
    P_All = euler(P1, Q, R, time, dt)

    #Begin solving for r based on the terminal condition. 
    r_All = r_integrate(rT, P_All, time, dt) 

    #Begin solving for z based on the initial condtion. Solve for v, once we have z.
    #This is the optimal descent direction.  
    z_All = z_integrate(z0, P_All, r_All, time, dt)
    v_All = v_integrate(P_All, r_All, z_All, time)

    #Compute the current cost of the trajectory and the directional derivative in the new direction.
    cost = J(trajectory[0:3,:], optimal, controls, time, Q, R, P1)
    d_J = dJ(trajectory[0:3,:], optimal, controls, time, z_All, v_All, P1)

    #Add the current objective.
    obj_Arr.append(cost)

    print("initial cost is:", cost)
    print("intial directional is:", d_J)

    n = 0 
    maxiter = 0
    gamma = beta ** n 

    #Compute the descent direction based on the new direction z, v. 
    while True:
        print("n is:",n)
        #Update the controls
        temp_controls = controls + gamma * v_All

        #Update the trajectory based on the new controls.
        update = np.zeros((3, time.size))

        #Create our trajectory vector with the controls added in. 
        update = np.concatenate((update, temp_controls)).T.flatten()
        update[0:3] = np.array([0,0,math.pi/2])

        #Create our updated trajectory -- make sure that it is scalable to larger sizes
        for ii in range(0,(5 * len(time)-5),5):
            #Calculate the intial velocities.
            update[ii + 5] = update[ii] + dt * update[ii + 3] * cos(update[ii + 2])
            update[ii + 6] = update[ii + 1] + dt * update[ii + 3] * sin(update[ii + 2])
            update[ii + 7] = update[ii + 2] + dt * update[ii + 4]  

        #Reshape our trajectory. 
        update = update.reshape(time.size,5).T

        #Find the cost of the new trajectory. 
        updated_cost = J(update[0:3,:],optimal,temp_controls,time, Q, R, P1)
        ob_Arr.append(updated_cost)
        print("updated_cost is", updated_cost)
        print("condition is:", cost + alpha * gamma * d_J)

        #Break out of the loop if our condition is met or if we have had too many iterations. 
        if updated_cost < cost + alpha * gamma * d_J:
            break 

        maxiter += 1
        if maxiter > 1000:
            break

        #If we don't break out of the loop, we just do it again. 
        n += 1 
        gamma = beta ** n 

    #Exit the loop if we have too many iterations. 
    if maxiter > 1000:
        break
    
    #When we break out of the loop.
    #Update our trajectory and the dictionary so that we can continue iterating. 
    trajectory = np.copy(update)

    #Update our controls based on our new trajectory.
    controls = trajectory[3:5,:]

    #Update the dictionary being called so that the loop can continue iterating.
    state_dict = dict(zip(time,trajectory.T)) 
    
    j += 1
    xi = abs(dJ(trajectory[0:3,:], optimal, controls, time, z_All, v_All, P1))
    print("j is:",j)
        

# %%
##Plots-------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1,2)
#Plotting the reference trajectory
axs[0].plot(ob_Arr[0:100])
axs[0].set_title("Objective Count in Armijo")

#Plotting the control signal 
axs[1].plot(obj_Arr, "tab:orange")
axs[1].set_title("Objective Overall")

# %%

##Plots-------------------------------------------------------------------------------------------------
fig, axs = plt.subplots()
divider = make_axes_locatable(axs) 

#Creation of the colormap and the colorbar. 
cax = divider.append_axes('right', size='5%', pad=0.05)
im = axs.hist2d(target[0,:], target[1,:], bins=100, cmap="binary")
fig.colorbar(im[3], cax=cax, orientation='vertical')
axs.set_aspect("equal")
axs.set_title("Distribution")

# #Plot the intitial distribution on the second figure.
# axs[1].scatter(target[0,:], target[1,:], c='orange')
# axs[1].axis("equal")
# axs[1].set_title("Samples")

#Plot the trajectory as well. 
axs.plot(traj_guess[0, :],traj_guess[1, :])
axs.axis("equal")
axs.set_title("Trajectory")
