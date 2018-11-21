#!/usr/bin/env python
from __future__ import print_function
import numpy as np
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

def get_in_ran(num,L):
    absolute = num + L*0.5
    while absolute > L :
	absolute = absolute - (L)
    while absolute <= 0.0 :
	absolute = absolute + (L)
    return absolute - L*0.5

def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary conditions. The simulation box is a cube of size box_length^ndim centered around the origin vec{0}. """
     
    new_pos = mypos.copy()
    ndim    = mypos.shape[0]
    for i in range(ndim):
	new_pos[i] = get_in_ran(new_pos[i], box_length) 
     
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle. Unlike the distance function, here you will return a vector instead of a scalar """
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    ndim = pset.pos(iat).shape[0]
    for i in range(ndim):
	comp = abs(posi[i]-posj[i])
	flag = int((box_length-comp) < comp)
	sign = (1.0 - 2.0*int(posi[i]<posj[i]))*(1.0-2.0*flag)
	comp = comp - (2.0*comp -box_length)*flag 
	disp[i] = sign * comp
    # i.e. r_i - r_j
    return disp
# end def distance

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
    for i in range(len(pos_t_plus_dt)):
	pos_t_plus_dt[i] = pos_t_plus_dt[i]+vel_t[i]*dt+0.5*accel_t[i]*dt*dt

    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    for i in range(len(vel_t_plus_dt)):
	vel_t_plus_dt[i] = vel_t_plus_dt[i] +0.5*(accel_t[i] + accel_t_plus_dt[i])*dt
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
    force = np.zeros(pset.ndim())
    for jat in range(natom):
	#if (np.array_equal(mypos,pos[jat])):
	#    continue
	if jat!=iat :
	    r = displacement(iat,jat,pset,box_length)
	    d = np.linalg.norm(r)
	    d6 = d*d*d*d*d*d
            force_mag = 24.0*(2.0/(d*d6*d6) - 1.0/(d*d6))
	    for m in range(len(force)):
	        force[m] = force[m] + r[m]/d * force_mag

    # calculate force

    return force
# end def internal_force

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    pos = pset.all_pos() # all particle positions 
    vel = pset.all_vel() # all particle velocies

    tot_kinetic   = 0.0
    tot_potential = 0.0 
    
    for i in range(natom):
	tot_kinetic = tot_kinetic + 0.5*mass*np.linalg.norm(pset.vel(i))*np.linalg.norm(pset.vel(i))
	for j in range(i+1,natom):
	    
	    #if (np.array_equal(pset.pos(i),pset.pos(j))) :
	    #	continue
	    d = distance(i, j, pset, box_length)
	    d6i = 1/(d*d*d*d*d*d)
	    tot_potential = tot_potential + 4.0*(d6i*d6i-d6i)
    #tot_potential = 0.5*tot_potential
    tot_energy = tot_kinetic + tot_potential
    return (tot_kinetic, tot_potential, tot_energy)
# end def compute_energy

# Output visulization data
def VMDOut(R):
    outFile=open("myTrajectory.xyz","a")
    for i in range(0,len(R)):
        outFile.write(str(i)+" "+str(R[i][0])+" "+str(R[i][1])+" "+str(R[i][2])+"\n")
    outFile.close()
# End output


if __name__ == '__main__':
 
    print (sys.argv)
    step_size = float(sys.argv[1])
    #step_number = int(round(float(sys.argv[2])))
    num_atoms   = 64
    mass        = 48.0
    temperature = 0.728
    box_length  = 4.2323167
    nsteps      = int(round(10.0/step_size))
    dt          = step_size
    file_name = "out_"+str(dt)
    fo = open(file_name,"w")
    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature)
    for iat in range(pset.size()):
	pset.change_accel(iat,internal_force(iat,pset,box_length)/mass)
    # molecular dynamics simulation loop
    for istep in range(nsteps):

        # calculate properties of the particles
        energy_output = compute_energy(pset,box_length)
	#print(istep, energy_output)
	fo.write(str(istep)+"\t")
	for x in range(len(energy_output)):
	    fo.write(str(energy_output[x])+"\t")
	fo.write("\n")
        # update positions
        for iat in range(num_atoms):
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            pset.change_pos(iat,new_pos)
        # end for iat
        
        # Q/ When should forces be updated?
        #new_accel = get_accel(pset,box_length,mass)

        # update velocities
        for iat in range(num_atoms):
            new_accel = (internal_force(iat,pset,box_length))/mass
	    my_next_vel = verlet_next_vel( pset.vel(iat), pset.accel(iat), new_accel, dt )
            pset.change_vel( iat, my_next_vel )
            pset.change_accel(iat,new_accel)
    	    
    fo.close()
	
	# end for iat
    # end for istep
# end __main__
