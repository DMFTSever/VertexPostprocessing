#!/usr/bin/python

import h5py
import numpy as np
import argparse
import scipy.integrate as integ

# import filename from command line

parser = argparse.ArgumentParser(description='calc physical chi_charge and chi_spin')
parser.add_argument('file')
args = parser.parse_args()
filename = args.file

f = h5py.File(filename, 'r')

#def printname(name):
#    print name
#f.visit(printname)

dat_ntn0 = np.array(f['stat-001/ineq-001/ntau-n0/value'])
dat_tausus = np.array(f['.axes/tausus'])
dtau = dat_tausus[1] - dat_tausus[0]
print 'delta_tau =', dtau

chi_tau_charge = dat_ntn0[0,1,0,1,:] + dat_ntn0[0,1,0,0,:] + dat_ntn0[0,0,0,1,:] + dat_ntn0[0,0,0,0,:] - 1.
chi_tau_spin = dat_ntn0[0,1,0,1,:] - dat_ntn0[0,1,0,0,:] - dat_ntn0[0,0,0,1,:] + dat_ntn0[0,0,0,0,:]

#TODO

#chiname = 'Chi-beta12-5-U575.dat'
#chifile = open(chiname,'w')

#print >> chifile, "# File:", filename
#print >> chifile, "# tau, Chim(tau), Chic(tau)"

#chidim = chi_tau_charge.shape[0]

#for i in range(chidim):
#        print >> chifile, dat_tausus[i],"\t",chi_tau_spin[i],"\t", chi_tau_charge[i]

#chifile.close()

chi_phys_charge = 1/2.*integ.simps(chi_tau_charge, dx = dtau)
chi_phys_spin = 1/2.*integ.simps(chi_tau_spin, dx = dtau)

print 'chi_phys_charge =', chi_phys_charge, ', chi_phys_spin =', chi_phys_spin

# stat-001/ineq-001/ntau-n0/error

exit()
