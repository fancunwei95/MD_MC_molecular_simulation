#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------

from particleset import ParticleSet
"""
The ParticleSet class is designed to hold the position, velocity and accelerations of a set of particles. Initialization methods init_pos_cubic(cube_length) and init_vel(temperature) are provided for your convenience.

pset = ParticleSet(natom) will initialize a particle set pset
pset.size() will return the number of particles in pset

| --------------------------------------------------------- | ------------------- |
|                   for access of                           |    use method       |
| --------------------------------------------------------- | ------------------- |
| all particle positions in an array of shape (natom,ndim)  |    pset.all_pos()   |
| all particle velocities                                   |    pset.all_vel()   |
| all particle accelerations                                |    pset.all_accel() |
| particle i position in an array of shape (ndim)           |    pset.pos(i)      |
| particle i velocity                                       |    pset.vel(i)      |
| particle i acceleration                                   |    pset.accel(i)    |
| --------------------------------------------------------- | ------------------- |

| ----------------------------- | ------------------------------------ | 
|           to change           |             use method               |
| ----------------------------- | ------------------------------------ |
| all particle positions        |  pset.change_all_pos(new_pos_array)  |
| particle i position           |  pset.change_pos(i,new_pos)          |
| ditto for vel and accel       |  pset.change_*(i,new_*)              |
| ----------------------------- | ------------------------------------ |
"""

# Routines to ensure periodic boundary conditions that YOU must write.
# ------------------------------------------------------------------------

def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary conditions. The simulation box is a cube of size box_length^ndim centered around the origin vec{0}. """
     
    new_pos = mypos.copy()
    ndim    = mypos.shape[0]
    L = box_length
    for i in range(ndim):
	absolute = new_pos[i] # + L*0.5
        while absolute > 0.5*L :
	   absolute = absolute - (L)
        while absolute <= - 0.5*L :
	   absolute = absolute + (L)
        new_pos[i] = absolute # - L*0.5
	#new_pos[i] = get_in_ran(new_pos[i], box_length) 
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle. Unlike the distance function, here you will return a vector instead of a scalar """
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    L = box_length
    ndim = pset.ndim()
    # i.e. r_i - r_j
    for k in range(ndim):
    	initial_disp = posi[k] - posj[k] ## kth component of Displacement vector from i to j 
    	while (initial_disp > L/2.0): ## Test if out of range
    	    initial_disp = initial_disp - L
    	while (initial_disp < -1*L/2.0):  ## Test if out of range
            initial_disp = initial_disp + L
    	disp[k] = initial_disp

    return disp

def distance(iat, jat, pset, box_length):
   
    return np.linalg.norm(displacement(iat,jat,pset,box_length))

# end def distance

# The Verlet time-stepping algorithm that YOU must write, dt is time step
# ------------------------------------------------------------------------
def verlet_next_pos(pos_t,vel_t,accel_t,dt):
    """
    We want to return position of the particle at the next moment t_plus_dt
    based on its position, velocity and acceleration at time t.  
    """
    pos_t_plus_dt = pos_t.copy()
    pos_t_plus_dt = pos_t_plus_dt+vel_t*dt+0.5*accel_t*dt*dt
    #for i in range(len(pos_t_plus_dt)):
    #    pos_t_plus_dt[i] = pos_t_plus_dt[i]+vel_t[i]*dt+0.5*accel_t[i]*dt*dt

    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    vel_t_plus_dt = vel_t_plus_dt +0.5*(accel_t + accel_t_plus_dt)*dt
    
    #for i in range(len(vel_t_plus_dt)):
    #	vel_t_plus_dt[i] = vel_t_plus_dt[i] +0.5*(accel_t[i] + accel_t_plus_dt[i])*dt
    return vel_t_plus_dt
# end def verlet_next_vel

# We want Lennard-Jones forces. YOU must write this.
# ------------------------------------------------------------------------
def internal_force(iat,pset,box_length):
    """
    We want to return the force on atom 'iat' when we are given a list of 
    all atom positions. Note, pos is the position vector of the 
    1st atom and pos[0][0] is the x coordinate of the 1st atom. It may
    be convenient to use the 'displacement' function above. For example,
    disp = displacement( 0, 1, pset, box_length ) would give the position
    of the 1st atom relative to the 2nd, and disp[0] would then be the x coordinate
    of this displacement. Use the Lennard-Jones pair interaction. Be sure to avoid 
    computing the force of an atom on itself.
    F = 4*(12*1/(r^13)-6*1/(r^7))
    """

    pos = pset.all_pos()  # positions of all particles
    mypos = pset.pos(iat) # position of the iat th particle
    natom = pset.size()
    dim = pset.ndim()
    force = np.zeros(dim)
    for jat in range(natom):
	if jat!=iat :
	    r = displacement(iat,jat,pset,box_length)
	    d = np.linalg.norm(r)
	    d6 = (d*d*d*d*d*d)
            force_mag = 24.0*(2.0/(d*d6*d6) - 1.0/(d*d6))     ##- 1.0/(d*d6))
	    for m in range(dim):
	        force[m] +=  r[m]/d*force_mag
    # calculate force
    return force
# end def internal_force

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    vel = pset.all_vel() # all particle velocies
    mass = pset.mass()
    tot_kinetic   = 0.0
    tot_potential = 0.0 
    for i in range(natom):
	tot_kinetic = tot_kinetic + 0.5*mass*np.linalg.norm(pset.vel(i))**2
	for j in range(i+1,natom):
	    d = distance(i, j, pset, box_length)
	    d6i = 1/(d*d*d*d*d*d)
	    tot_potential = tot_potential + 4.0*(d6i*d6i-d6i)
    
    tot_energy = tot_kinetic + tot_potential
    return (tot_kinetic, tot_potential, tot_energy)

def compute_potential(pset,box_length):
    natom = pset.size()  # number of particles
    mass = pset.mass()
    tot_potential = 0.0 
    for i in range(natom):
	for j in range(i+1,natom):
	    d = distance(i, j, pset, box_length)
	    d6i = 1/(d*d*d*d*d*d)
	    tot_potential = tot_potential + 4.0*(d6i*d6i-d6i)
    return tot_potential
    

def temp(tot_kinetic):
    natom = pset.size()
    return 2.0*tot_kinetic/3.0/natom

def momentum(pset):
    tot_momentum = np.zeros(pset.ndim())
    mass = pset.mass()
    natom = pset.size()
    for i in range(natom):
	tot_momentum = tot_momentum + mass*pset.vel(i)
    return tot_momentum
### gr_plot ##########################################################

def initial_gr(gr_bin, low_b, upp_b):
    count_list = np.zeros((2,gr_bin))
    dr = float(upp_b - low_b)/(gr_bin)
    for m in range(len(count_list[0])):
	count_list[0][m] = low_b + dr*(m+0.5)
    return count_list

def update_gr(pset,box_length,count_list,dt):
    natom = pset.size()
    pair_num = natom*(natom-1)/2
    dist_list = np.zeros(pair_num)
    count = 0
    for iat in range(natom-1):
	for jat in range(iat+1,natom):
	    dist = distance(iat, jat, pset, box_length)
	    dist_list[count] = dist
	    count = count +1
    dist_list.sort()
    dr = count_list[0][2] - count_list[0][1]            
    upp_b =  count_list[0][-1]+0.5*dr     
    m = 0
    volume  = box_length **( pset.pos(0).shape[0] )
    for n in range(pair_num):
        this_d = dist_list[n]
	if this_d > upp_b:
	    break	
	upper = count_list[0][m]+0.5*dr 
	while (this_d>= upper):
	    m = m +1
	    upper = count_list[0][m]+ 0.5*dr    
	this_r = count_list[0][m]  
        count_list[1][m] = count_list[1][m] + dt/(2*np.pi*(this_r)*(this_r)*natom*natom/volume*dr)
    return count_list

def plot_gr(gr_arr, total_time, box_length, show = True):
    plt.plot(gr_arr[0]/box_length, gr_arr[1]/total_time,"r")
    plt.xlabel("r/L")
    plt.ylabel("g(r)")
    if show:
    	plt.show()
    
    return 

##### Structure Factor Function ###############################################
def legal_kvecs(maxk,box_length):
    ndim = pset.ndim()
    kvecs = np.array([np.array(k_pos) for k_pos in itertools.product(range(maxk+1),repeat = ndim) ])
    fac = 2.0*np.pi/box_length
    kvecs = fac * kvecs
    return kvecs

def rhok(kvec,pset):
    #v_real = 0.0
    #v_imag = 0.0
    v = 0.0
    for i in range(pset.size()):
	k_r = np.dot(kvec,pset.pos(i))
	v =v + np.exp(-1.0j*k_r)
	#v_real = v_real + np.cos(k_r)
	#v_imag = v_imag + np.sin(k_r)
    return v     #v_real, v_imag

def update_sk(sk_arr,pset,dt):
    natom = pset.size()
    for i in range(len(sk_arr[0])):
        v = rhok(sk_arr[0][i],pset)
	sk_arr[1][i] = sk_arr[1][i] + dt/natom *(np.absolute(v))*(np.absolute(v))
    return sk_arr

def initial_sk(maxk,box_length):
    kvecs = legal_kvecs(maxk,box_length)
    sk_arr = np.zeros(len(kvecs))
    return [kvecs, sk_arr]

def plot_sk(SK_arr, T, show = True):
    kvecs = SK_arr[0]
    sk_arr = SK_arr[1]*(1.0/T)
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
	kmag    = unique_kmags[iukmag]
	idx2avg = np.where(kmags==kmag)
	unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    plt.plot(unique_kmags[1:],unique_sk[1:])
    plt.xlabel("k_mag")
    plt.ylabel("sk")
    if show:
    	plt.show()
    return 

#### velocity correlation ###########################################

def cv(v_0,pset):
    v_list = pset.all_vel()
    result = 0.0
    N = pset.size()
    for i in range(N):
	result = result + np.dot(v_0[i],v_list[i])/np.dot(v_0[i],v_0[i])
    return result/float(N)

def average_cv(cv_arr):
    sum_cv = 0.0
    for i in range(len(cv_arr)):
	sum_cv = sum_cv + cv_arr[i]
	cv_arr[i] = sum_cv/float(i+1)
    return cv_arr

def plot_cv(cv_arr,dt,show=True):
    cv_arr = average_cv(cv_arr)
    x = np.zeros(len(cv_arr))
    for i in range(len(cv_arr)):
	x[i] = i*dt
    diff = D(cv_arr,dt)
    plt.plot(x,cv_arr, "r", label = "D ="+str(diff) )
    #plt.plot(x, D_arr, "b", label = "diffusion")
    plt.ylabel("cv")
    plt.xlabel("time")
    plt.legend()  #bbox_to_anchor = (1.3,1))
    if show:
    	plt.show()
    return 

def D(cv_arr,dt):
    result = 0.0
    #D_arr =np.zeros(len(cv_arr))
    for i in range(len(cv_arr)):
	result = result + cv_arr[i]*dt
	#D_arr[i] = result
    return result

def PLOT(gr_arr, sk_arr, cv_arr, temp_arr, total_time, cutoff, box_length, dt):
    plt.subplot(221)
    plot_gr(gr_arr, total_time-cutoff, box_length, False)
    
    plt.subplot(222)
    plot_sk(sk_arr, total_time-cutoff,False)
    
    plt.subplot(223)
    plot_cv(cv_arr,dt, False)

    plt.subplot(224)
    plot_temp(temp_arr,dt,False)
    plt.show()
    return

#### change random velocity ########################################

def plot_temp(temp_arr,dt,show=True):
    #cv_arr = average_cv(temp_arr)
    x = np.zeros(len(temp_arr))
    for i in range(len(temp_arr)):
	x[i] = i*dt
    plt.figure(1)
    plt.plot(x,temp_arr)
    plt.xlabel("time")
    plt.ylabel("temperature")
    '''
    plt.figure(2)
    plt.plot(x,temp_arr*1.5*num_atoms)
    plt.xlabel("time")
    plt.ylabel("kinetic energy")
    '''
    if show:
    	plt.show()
    return 

def plot_mome(mome_arr, dt, show=True):
    x = np.zeros(len(mome_arr))
    for i in range(len(mome_arr)):
	x[i] = i*dt
    plt.plot(x,mome_arr)
    plt.xlabel("time")
    plt.ylabel("total_momentum")
    if show:
    	plt.show()
    return 

### mian ###########################################################
if __name__ == '__main__':
 
    #print (sys.argv)
    num_atoms   = 64
    mass        = 48.0
    temperature = 0.728
    box_length  = 4.0 #4.2323167
    nsteps      = 1000  #int(round(10.0/step_size))
    dt          = 0.032 
    
    gr_bin      = 200  
    Temp        = 0.75  # the canonical temperature
    cutoff      = 6.4 
    sigma       = np.sqrt(Temp/mass)

    total_time  = dt* nsteps
    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(Temp)       #temperature
    
    # creat arries to do time average 
    vel_0 = pset.all_vel()                             # get the initial velocity
    gr_arr = initial_gr(gr_bin, 0.0, box_length*0.5)   # gr arry with gr_bin slots 
    sk_arr = initial_sk(5,box_length)                  # sk arry 
    cv_arr = np.zeros(nsteps)                          # velocity correlation arry
    temp_arr = np.zeros(nsteps)                        # zeros temperature
    mome_arr = np.zeros(nsteps) 
    for iat in range(pset.size()):
	pset.change_accel(iat,internal_force(iat,pset,box_length)/mass)
    # molecular dynamics simulation loop
    for istep in range(nsteps):
	    # calculate properties of the particles
	com_energy = compute_energy(pset,box_length)
	temp_now = temp(com_energy[0])

	print(istep, com_energy)
	
	cv_step = cv(vel_0,pset)
	cv_arr[istep] = cv_step
	temp_arr[istep] = temp_now
	mome_arr[istep] = np.linalg.norm(momentum(pset))
	# update positions
        for iat in range(num_atoms):
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            pset.change_pos(iat,new_pos)
        # end for iat
	if (istep*dt >= cutoff) : 
	    gr_arr = update_gr(pset,box_length,gr_arr,dt)
	    sk_arr = update_sk(sk_arr,pset,dt)
	# update velocities
        for iat in range(num_atoms):
            new_accel = (internal_force(iat,pset,box_length))/mass
	    my_next_vel = verlet_next_vel( pset.vel(iat), pset.accel(iat), new_accel, dt )
            pset.change_vel( iat, my_next_vel )
            pset.change_accel(iat,new_accel) 
	    if (np.random.random()<0.50):
                 ran_vel = np.random.normal(0.0,sigma,3)
	         pset.change_vel(iat, ran_vel)

    #plot_gr(gr_arr, total_time-cutoff, box_length)
    #plot_sk(sk_arr, total_time-cutoff)
    #plot_cv(cv_arr,dt)
    PLOT(gr_arr, sk_arr, cv_arr, temp_arr, total_time, cutoff, box_length, dt)
    #plot_mome(mome_arr,dt)
    #plot_temp(temp_arr,dt)
    # end for istep
# end __main__
