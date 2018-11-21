import numpy as np

def calc_disp(iat, jat, pset, box_length):
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    L = box_length
    ndim = pset.ndim()
    # i.e. r_i - r_j
    for k in range(ndim):
    	initial_disp = posi[k] - posj[k] 
    	while (initial_disp > L/2.0): ## Test if out of range
    	    initial_disp = initial_disp - L
        while (initial_disp < -1*L/2.0):  ## Test if out of range
            initial_disp = initial_disp + L
    	disp[k] = initial_disp

    return disp

def calc_dist(iat, jat, pset, box_length):
    return np.linalg.norm(displacement(iat,jat,pset,box_length))

class Info:
    def __init__(self,pset,box_length):
        self._box_length = box_length
        self._pset =pset
        self._displacement = np.zeros([pset.size(),pset.size(),pset.ndim()])
        self._distance = np.zeros([pset.size(),pset.size()])
        self._potential_list = np.zeros([pset.size(),pset.size()])
        self._extra_list = np.zeros([pset.size(),pset.size(),3])
        self._extra_name = ["d2i","d6i","d12i"]

        self._displacement_new = np.zeros([pset.size(),pset.ndim()])
        self._distance_new = np.zeros(pset.size())
        self._potential_list_change = np.zeros(pset.size())
        self._extra_list_new = np.zeros([pset.size(),3])
        self._potential_change = 0.0
        self._potential = 0.0

    def pset(self):
        return self._pset

    def box_legnth(self):
        return self._box_length
    
    def distance(self):
        return self._distance

    def displacement(self):
        return self._displacement
    
    def potential(self):
        return self._potential
    
    def potential_change(self):
        return self._potential_change

    def potential_list(self):
        return self._potential_list  
    
    def extra_list(self):
        return self._extra_name

    def calc_disp(iat, jat, pset, box_length):
        posi = pset.pos(iat)
        posj = pset.pos(jat)
        disp = posi.copy()
        L = box_length
        ndim = pset.ndim()
        # i.e. r_i - r_j
        for k in range(ndim):
    	    initial_disp = posi[k] - posj[k] 
    	    while (initial_disp > L/2.0): ## Test if out of range
    	        initial_disp = initial_disp - L
            while (initial_disp < -1*L/2.0):  ## Test if out of range
                initial_disp = initial_disp + L
    	    disp[k] = initial_disp
        return disp

    def update_all(self):
        natom = self._pset.size()
        new_potential = 0.0
        for iat in range(natom-1):
            for jat in range(iat+1, natom):
                disp = calc_disp(iat,jat,self._pset, self._box_length)
                self._displacement[iat,jat,:] = disp[:]
                self._displacement[jat,iat,:] = -1.0*disp[:]
                d = np.linalg.norm(disp)
                self._distance[iat,jat] = d
                self._distance[jat,iat] = d
                d2i = 1/(d*d)
                d6i = d2i*d2i*d2i
                d12i = d6i*d6i
	        this_potential = 4.0*(d12i-d6i)
                new_potential = new_potential + this_potential
                self._potential_list[iat,jat] = this_potential
                self._potential_list[jat,iat] = this_potential
                self._extra_list[iat,jat,:] = [d2i,d6i,d12i]
                self._extra_list[jat,iat,:] = [d2i,d6i,d12i]
        self._potential = new_potential
        return

    def trial_update_ptcl(self,iat):
        natom = self._pset.size()
        for jat in range(natom):
            if iat == jat:
                self._displacement_new[iat,:] = 0.0
                self._distance_new[iat] = 0.0
                self._potential_list_change[iat] = 0.0
                self._extra_list_new[iat,:] = 0.0
                continue
            disp = calc_disp(iat,jat,self._pset, self._box_length)
            self._displacement_new[jat,:] = disp
            d = np.linalg.norm(disp)
            self._distance_new[jat] = d
            d2i = 1.0/(d*d)
            d6i = d2i*d2i*d2i
            d12i = d6i*d6i
            this_potential = 4.0*(d12i-d6i)
	    self._extra_list_new[jat,:] = [d2i,d6i,d12i]
            del_potential = this_potential- self._potential_list[iat,jat]
            self._potential_list_change[jat] = del_potential
        self._potential_change = np.sum(self._potential_list_change)
        return self._potential_change
    
    def update_ptcl_displacement(self,iat):
        self._displacement[iat,:,:] = self._displacement_new
        self._displacement[:,iat,:] = -1.0* self._displacement_new
        
    def update_ptcl_distance(self,iat):
        self._distance[iat,:] = self._distance_new
        self._distance[:,iat] = self._distance_new

    def update_ptcl_potential(self,iat):
        self._potential_list[iat,:] = self._potential_list[iat,:] + self._potential_list_change
        self._potential_list[:,iat] = self._potential_list[iat,:]
        self._potential = self._potential + self._potential_change 

    def update_ptcl_extra(self,iat):
        self._extra_list[iat,:,:] = self._extra_list_new
        self._extra_list[:,iat,:] = -1.0* self._extra_list_new
    
    def update_ptcl_all(self,iat):
        update_ptcl_displacement(self,iat)
        update_ptcl_distance(self,iat)
        update_ptcl_potential(self,iat)
        update_ptcl_extra(self,iat)
    
    def get_ptcl_force(self,iat):
        d2i = self._extra_list[iat,:,0]
        d6i = self._extra_list[iat,:,1]
        d12i = self._extra_list[iat,:,2]
        factor = 24.0*(2.0*d12i-d6i)*d2i
        disp = self._displacement[iat,:,:]
        return np.sum(disp*((factor[np.newaxis]).T),axis = 0)
    
    def get_ptcl_force_new(self,iat):
        d2i = self._extra_list_new[:,0]
        d6i = self._extra_list_new[:,1]
        d12i = self._extra_list_new[:,2]
        factor = 24.0*(2.0*d12i-d6i)*d2i
        disp = self._displacement_new[:,:]
        return np.sum(disp*((factor[np.newaxis]).T),axis = 0)
