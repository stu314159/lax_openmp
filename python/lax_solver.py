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
from numba import cuda


parser = argparse.ArgumentParser(prog='lax_solver.py',
                                 description='solve linear scalar wave equation')
                                 
parser.add_argument('N',type=int)
parser.add_argument('Num_ts',type=int)
parser.add_argument('solver',type=int)
parser.add_argument('Write_output',type=int)
args = parser.parse_args()

@cuda.jit('void(float64[:],float64[:],float64,float64,int32)')
def lax_numba_cuda(F_out,F_in,dx,dt,N):
     # get thread data
     tx = cuda.threadIdx.x;
     bx = cuda.blockIdx.x;
     bw = cuda.blockDim.x;
     tid = tx + bx*bw;
     x_p = tid+1; x_m = tid-1;
     if(x_m<0):
         x_m = N-1;
     if(x_p == N):
         x_p = 0;
         
     if (tid<N):
        F_out[tid] = 0.5*(F_in[x_p]+F_in[x_m])-(dt/(2.*dx))*(F_in[x_p]-F_in[x_m])

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
u = 1.;  # wave speed

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
if (solver == 3): # numba cuda jit
    F_even = np.copy(F);
    F_odd = np.copy(F);
    # copy array to gpu
    dF_even = cuda.to_device(F_even)
    dF_odd = cuda.to_device(F_odd)
    
    # configure thread grid
    threads_per_block = 256;
    num_blocks = np.ceil(float(N)/float(threads_per_block))
    griddim = int(num_blocks);
    blockdim = threads_per_block
    
    # launch kernel
    for ts in range(Num_ts):
        
        if (ts%2)==0:
            lax_numba_cuda[griddim,blockdim](dF_odd,dF_even,
                          dx,dt,np.int32(N));
        else:
            lax_numba_cuda[griddim,blockdim](dF_even,dF_odd,
                          dx,dt,np.int32(N));
    
    
    # copy data back to host
    if (Num_ts%2==0):
        F = dF_even.copy_to_host();
    else:
        F = dF_odd.copy_to_host();


time_stop = time.time();
elapsed_time = time_stop - time_start;
print("Elapsed time: %g seconds"%(elapsed_time))

if Write_output==1:
    p1 = plt.plot(X,F)
    plt.show()
