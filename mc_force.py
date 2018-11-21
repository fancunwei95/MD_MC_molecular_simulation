from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
from particleset import ParticleSet 
from information import Info
import random
 
def update_position(pset,iat,sigma):
    old_ptcl_pos = pset.pos(iat)
    new_ptcl_pos = old_ptcl_pos + np.random.normal(0.0,sigma,pset.ndim())
    pset.change_pos(iat,new_ptcl_pos)
    return old_ptcl_pos

def update_position_force(pset,iat,sigma,force,mass,dt_2):
    old_ptcl_pos = pset.pos(iat)
    x_adjust = force/(2.0*mass)*dt_2
    del_x = x_adjust + np.random.normal(0.0,sigma,pset.ndim())
    new_ptcl_pos = old_ptcl_pos + del_x
    pset.change_pos(iat,new_ptcl_pos)
    return old_ptcl_pos, del_x, x_adjust

def accept(delta_V,T):
    A = min(1.0,(1.0/np.exp(delta_V/T)))
    if np.random.uniform()< A:
        return True
    return False

def accept_force(delta_V, T, del_x, x_adjust, x_adjust_new, sigma):
    num_1 = np.linalg.norm(del_x - x_adjust)
    num_2 = np.linalg.norm(del_x + x_adjust_new)
    T_ratio = np.exp((num_1*num_1 - num_2*num_2)/(2*sigma*sigma))
    A_test = T_ratio/(np.exp(delta_V/T))
    A = min(1.0,A_test)
    if np.random.uniform()< A:
        return True
    return False

def initial_gr(gr_bin, low_b, upp_b):
    count_list = np.zeros((2,gr_bin))
    dr = float(upp_b - low_b)/(gr_bin)
    for m in range(len(count_list[0])):
	count_list[0][m] = low_b + dr*(m+0.5)
    return count_list

def update_gr(pset, info, box_length,count_list):
    natom = pset.size()
    pair_num = natom*(natom-1)/2
    dist_list = np.zeros(pair_num)
    count = 0
    for iat in range(natom-1):
	for jat in range(iat+1,natom):
	    dist_list[count] = info._distance[iat,jat]
	    count = count +1
    dist_list.sort()
    dr = count_list[0][2] - count_list[0][1]            
    upp_b =  count_list[0][-1]+0.5*dr     
    m = 0
    volume  = box_length **( pset.pos(0).shape[0] )
    for n in range(pair_num):
        this_d = dist_list[n]
	if this_d >= upp_b:
	    break	
	upper = count_list[0][m]+0.5*dr 
	while (this_d> upper):
	    m = m +1
	    upper = count_list[0][m]+ 0.5*dr    
	this_r = count_list[0][m]  
        count_list[1][m] = count_list[1][m] + 1.0/(2*np.pi*(this_r)*(this_r)*natom*natom/volume*dr)
    return count_list

def plot_gr(gr_arr, total_sweeps, box_length, show = True):
    plt.plot(gr_arr[0]/box_length, gr_arr[1]/total_sweeps,"r")
    plt.xlabel("r/L")
    plt.ylabel("g(r)")
    if show:
    	plt.show()
    return 

def legal_kvecs(maxk,box_length):
    ndim = pset.ndim()
    kvecs = np.array([np.array(k_pos) for k_pos in itertools.product(range(maxk+1),repeat = ndim) ])
    fac = 2.0*np.pi/box_length
    kvecs = fac * kvecs
    return kvecs

def rhok(kvec,pset):
    v = 0.0
    for i in range(pset.size()):
	k_r = np.dot(kvec,pset.pos(i))
	v =v + np.exp(-1.0j*k_r)
    return v     

def update_sk(sk_arr,pset):
    natom = pset.size()
    for i in range(len(sk_arr[0])):
        v = rhok(sk_arr[0][i],pset)
	sk_arr[1][i] = sk_arr[1][i] + 1.0/natom *(np.absolute(v))*(np.absolute(v))
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

def monte_carlo(pset, info, sweep_numbers, T, gr_bins, box_length, dt_2,  out = True):
    potential_list = np.zeros(sweep_numbers)
    natom = pset.size()
    mass = pset.mass()
    sweep = 1
    success = 0.0
    gr_arr = initial_gr(gr_bin, 0.0, box_length*0.5)  
    sk_arr = initial_sk(5,box_length)
    while sweep <= sweep_numbers :
        for iat in range(natom):
            #old_posi = update_position(pset,iat,sigma)   
            iat_force = info.get_ptcl_force(iat)
            old_posi, del_x, x_adjust = update_position_force(pset,iat,sigma,iat_force,mass, dt_2)
            delta_V = info.trial_update_ptcl(iat)
            x_adjust_new = info.get_ptcl_force_new(iat)/(2*mass) *dt_2
            #if accept(delta_V,T):
            if accept_force(delta_V, T, del_x, x_adjust, x_adjust_new, sigma):
                info.update_ptcl_potential(iat)
                info.update_ptcl_distance(iat)
                info.update_ptcl_displacement(iat)
                info.update_ptcl_extra(iat)
                success = success + 1.0
            else:
                pset.change_pos(iat,old_posi)
            potential_list[sweep-1] = info.potential()
        print (sweep, info.potential())
        sweep = sweep + 1
        #gr_arr = new_update_gr(pset, info, box_length,gr_arr)
        #sk_arr = update_sk(sk_arr,pset)
    #plot_gr(gr_arr, sweep_numbers, box_length)
    #plot_sk(sk_arr, sweep_numbers)
    return success / (sweep_numbers*natom)

def new_update_gr(pset, info, box_length,count_list):
    dr = count_list[0][2] - count_list[0][1]            
    natom = pset.size()
    volume  = box_length **( pset.pos(0).shape[0] )
    upp_b =  count_list[0][-1]+0.5*dr 
    for iat in range(natom-1):
        for jat in range(iat+1,natom):
            this_d = info.distance()[iat,jat]
            if this_d>= upp_b:
                continue
	    m = int(np.floor(this_d/dr))
	    this_r = count_list[0][m]  
            count_list[1][m] = count_list[1][m] + 1.0/(2*np.pi*(this_r)*(this_r)*natom*natom/volume*dr)
    return count_list


if __name__ == '__main__':
 
    num_atoms   = 64
    mass        = 48.0
    temperature = 2.0
    box_length  = 4.0 
    
    gr_bin      = 200  
    sigma       = 0.06
    sweep_num   = 2000
    dt_2        = 0.1
    np.random.seed(1)
    
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature)

    info = Info(pset,box_length)
    info.update_all()
    
    #update_position(pset,i,sigma)
    #info.update_ptcl(i)
    print (monte_carlo(pset, info, sweep_num, temperature, gr_bin,  box_length, dt_2, out = True))
    #print (info.potential())
    #print (info.potential())
