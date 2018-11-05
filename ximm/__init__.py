"""An emulator for the matter correlation function based on the Aemulus simulations.
"""
import numpy as np
import george
from george.kernels import *
import scipy.optimize as op
from classy import Class
import cluster_toolkit.xi as ctxi

#Load in all the data we need, which includes pre-computed matrices
radii         = np.loadtxt("radii.txt")
scale_factors = np.load("scale_factors.npy")
ln_xi_mean    = np.load("ln_xi_mean.npy")
phis          = np.load("phis.npy")
cosmologies   = np.load("cosmologies.npy")
redshifts     = 1./scale_factors - 1.
Nr = len(radii)
Nz = len(redshifts)

#Training data standard deviation in ln(xi_mm)
ln_xi_stddev = 1.609

#Number of principle components and number of cosmological parameters
Npc          = len(phis)
N_cos_params = len(cosmologies[0])
metric       = np.load("metric.npy")
gp_params    = np.load("gp_parameters.npy")
weights      = np.load("weights.npy")

#Create the gaussian processes for each principle component
gplist = []
for i in range(Npc):
    kernel = 1.*(ExpSquaredKernel(metric, ndim=N_cos_params) +
                 Matern32Kernel(metric, ndim=N_cos_params))
    gp = george.GP(kernel=kernel, mean=0)
    gp.set_parameter_vector(gp_params[i])
    gp.compute(cosmologies)
    gplist.append(gp)

#Cosmological parameters: Omega_bh^2 Omega_ch^2 w0 ns ln10As H0[km/s/Mpc] Neff
def predict(params):
    #Weigth for each principle component for these parameters
    weights_predicted = np.zeros((Npc))
    for i in range(Npc):
        weights_predicted[i] = gplist[i].predict(weights[i], np.atleast_2d(params))[0]
    #Loop over weights and add to the prediction
    r2ximm_diff = np.zeros((Nz*Nr))
    print(weights_predicted.shape, phis.shape)
    for i in range(Npc):
        r2ximm_diff += weights_predicted[i] * phis[i]
    r2ximm_diff *= ln_xi_stddev
    r2ximm_diff += ln_xi_mean
    #Create the output
    ximm = np.zeros((Nz, Nr))
    #Compute xi_nl from CLASS and the toolkit
    kmin = 1e-5
    kmax = 10
    k = np.logspace(np.log10(kmin), np.log10(kmax), num=1000) #Mpc^-1
    obh2, och2, w, ns, H0, Neff, sigma8 = params
    h = H0/100.
    Omega_b = obh2/h**2
    Omega_c = och2/h**2
    Omega_m = Omega_b+Omega_c
    #print(h, sigma8, ns, w, Omega_b, Omega_c, Neff)
    params = {'output': 'mPk',
              'h': h, 'sigma8':sigma8, 'n_s': ns,
              'w0_fld': w, 'wa_fld': 0.0, 'Omega_b': Omega_b,
              'Omega_cdm': Omega_c, 'Omega_Lambda': 1.- Omega_m,
              'N_eff': Neff,
              'P_k_max_1/Mpc':10., 'z_max_pk':5.,'non linear':'halofit'}
    cosmo = Class()
    cosmo.set(params)
    print("CLASS is computing")
    cosmo.compute()
    print("\tCLASS done")
    r2xinl = np.zeros((Nz, Nr))
    kh = k/h #h/Mpc
    for i in range(Nz):
        P = np.array([cosmo.pk(ki, redshifts[i]) for ki in k])*h**3
        r2xinl[i] = radii**2 * ctxi.xi_mm_at_R(radii, kh, P)
        ximm[i] = (r2ximm_diff[i*Nr :(i+1)*Nr] + r2xinl[i])/radii**2
    #Return the full prediction
    return ximm

#Test it
test_ind = 0
cos = cosmologies[test_ind]
ximm = predict(cos)
import matplotlib.pyplot as plt
for i in range(Nz):
    plt.loglog(radii, radii**2*ximm[i])
plt.show()
