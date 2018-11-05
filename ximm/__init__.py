"""An emulator for the matter correlation function based on the Aemulus simulations.
"""
import numpy as np
import george
import scipy.optimize as op

#Training data standard deviation in ln(xi_mm)
ln_xi_stddev = 1.609

#Number of principle components
Npc = 12

#Load in all the data we need, which includes pre-computed matrices
r             = np.loadtxt("radii.txt")
scale_factors = np.load("scale_factors.npy")
ln_xi_mean    = np.load("ln_xi_mean.npy")
phis          = np.load("phis.npy")
