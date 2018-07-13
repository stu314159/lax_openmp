#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:29:37 2018

@author: stu
"""
from __future__ import print_function
import sys
sys.path.insert(1,'.')

import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import numba


parser = argparse.ArgumentParser(prog='lax_solver.py',
                                 description='solve linear scalar wave equation')
                                 
parser.add_argument('N',type=int)
parser.add_argument('Num_ts',type=int)
parser.add_argument('solver',type=int)
parser.add_argument('Write_output',type=int)
args = parser.parse_args()

@numba.jit()
def lax_numba(F,dx,dt,Num_ts):
    f_tmp = np.copy(F)
    #x_range = np.int32(F.size);
    x_ind = np.arange(F.size,dtype=np.int32);
    x_p = np.roll(x_ind,-1)
    x_m = np.roll(x_ind,1)
    
    for ts in range(Num_ts):
        #if (ts%5000)==0:
        #    print("Executing time step number %d"%ts)
        f_tmp = 0.5*(F[x_p]+F[x_m]) - (dt/(2.*dx))*(F[x_p]-F[x_m])
        F = f_tmp[:]
    return F[:]
        
        


def lax_numpy(F,dx,dt,Num_ts):
    x_ind = np.arange(F.size,dtype=np.int32);
    x_p = np.roll(x_ind,-1)
    x_m = np.roll(x_ind,1)
    #f_even = np.copy(F)
    #f_odd = np.copy(F)
    f_tmp = np.copy(F)
    for ts in range(Num_ts):
        if (ts%5000)==0:
            print("Executing time step number %d"%ts)
         
        f_tmp = 0.5*(F[x_p]+F[x_m]) - (dt/(2.*dx))*(F[x_p]-F[x_m])
        F = f_tmp;
        
    
    return F[:]
        


    
N = args.N;
Num_ts = args.Num_ts;
solver = args.solver;
Write_output = args.Write_output

# initialize the problem
x_left = -10.; # left boundary
x_right = 10.; # right boundary
u = 1;  # wave speed

X = np.linspace(x_left,x_right,num=N,dtype=np.float64)
dx = X[1]-X[0];
dt = 0.6*dx/u;

# initialize array
F = np.exp(-X*X)

time_start = time.time();
# solver 1 = numpy vectorized
if (solver == 1): # regular numpy
    F = lax_numpy(F,dx,dt,Num_ts);
if (solver == 2): # numba jit
    F = lax_numba(F,dx,dt,np.int32(Num_ts));


time_stop = time.time();
elapsed_time = time_stop - time_start;
print("Elapsed time: %g seconds"%(elapsed_time))

if Write_output==1:
    p1 = plt.plot(X,F)
    plt.show()
