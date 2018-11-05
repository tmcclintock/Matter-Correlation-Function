"""An emulator for the matter correlation function based on the Aemulus simulations.
"""
import numpy as np
import george
from george.kernels import *
import scipy.optimize as op
from classy import Class
import cluster_toolkit.xi as ctxi
import os, inspect
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/"

#Load in all the data we need, which includes pre-computed matrices
radii         = np.load(data_path+"radii_xi_mm_emu.npy")
scale_factors = np.load(data_path+"scale_factors.npy")
ln_xi_mean    = np.load(data_path+"ln_xi_mean.npy")
phis          = np.load(data_path+"phis.npy")
cosmologies   = np.load(data_path+"cosmologies.npy")
training_cos  = np.delete(cosmologies,4,1) #delete ln10As

redshifts     = 1./scale_factors - 1.
Nr = len(radii)
Nz = len(redshifts)

#Training data standard deviation in ln(xi_mm)
ln_xi_stddev = 1.609

#Number of principle components and number of cosmological parameters
Npc          = len(phis)
N_cos_params = len(training_cos[0])
metric       = np.load(data_path+"metric.npy")
gp_params    = np.load(data_path+"gp_parameters.npy")
weights      = np.load(data_path+"weights.npy")

#Create the gaussian processes for each principle component
gplist = []
for i in range(Npc):
    kernel = 1.*(ExpSquaredKernel(metric, ndim=N_cos_params) +
                 Matern32Kernel(metric, ndim=N_cos_params))
    gp = george.GP(kernel=kernel, mean=0)
    gp.set_parameter_vector(gp_params[i])
    gp.compute(training_cos)
    gplist.append(gp)

#Cosmological parameters: Omega_bh^2 Omega_ch^2 w0 ns ln10As H0[km/s/Mpc] Neff
class ximm_emulator(object):
    def __init__(self, parameters):
        #Check the inputs
        assert parameters is not None, "Must supply cosmological parameters."
        assert len(parameters) is 7, "Only seven cosmological parameters supported."
        #Class parameters
        self.parameters = parameters
        #Emulator things
        self.gplist = gplist
        self.radii = radii
        self.scale_factors = scale_factors
        self.redshifts = redshifts
        self.phis = phis
        self.weights = weights
        self.Npc = Npc
        self.Nr = Nr
        self.Nz = Nz
        self.ln_xi_stddev = ln_xi_stddev
        self.ln_xi_mean = ln_xi_mean
        #Call the class setup
        self.setup_class()

    def setup_class(self):
        obh2, och2, w, ns, ln10As, H0, Neff = self.parameters
        h = H0/100.
        Omega_b = obh2/h**2
        Omega_c = och2/h**2
        Omega_m = Omega_b+Omega_c
        params = {'output': 'mPk',
                  'h': h, 'ln10^{10}A_s': ln10As, 'n_s': ns,
                  'w0_fld': w, 'wa_fld': 0.0, 'Omega_b': Omega_b,
                  'Omega_cdm': Omega_c, 'Omega_Lambda': 1.- Omega_m,
                  'N_eff': Neff,
                  'P_k_max_1/Mpc':10., 'z_max_pk':5.,'non linear':'halofit'}
        cosmo = Class()
        cosmo.set(params)
        print("CLASS is computing")
        cosmo.compute()
        print("\tCLASS done")
        self.class_cosmo_object = cosmo
        return

    def predict(self, params=None):
        #If we have new parameters
        if params is not None and params is not self.parameters:
            assert len(params) is 7, "Only seven cosmological parameters supported."
            self.parameters = params
            setup_class()
        else:
            params = self.parameters
        #Weight for each principle component for these parameters
        weights_predicted = np.zeros((self.Npc))
        for i in range(self.Npc):
            weights_predicted[i] = self.gplist[i].predict(self.weights[i], np.atleast_2d(params))[0]
        #Loop over weights and add to the prediction
        r2ximm_diff = np.zeros((self.Nz*self.Nr))
        for i in range(Npc):
            r2ximm_diff += weights_predicted[i] * self.phis[i]
        r2ximm_diff *= self.ln_xi_stddev
        r2ximm_diff += self.ln_xi_mean
        #Create the output
        ximm   = np.zeros((self.Nz, self.Nr))
        r2xinl = np.zeros((self.Nz, self.Nr))
        #Compute xi_nl from CLASS and the toolkit
        kmin = 1e-5
        kmax = 10
        k = np.logspace(np.log10(kmin), np.log10(kmax), num=1000) #Mpc^-1
        h = self.parameters[5]/100. #Hubble constant
        kh = k/h #h/Mpc
        for i in range(Nz):
            P = np.array([self.class_cosmo_object.pk(ki, self.redshifts[i]) for ki in k])*h**3
            r2xinl[i] = radii**2 * ctxi.xi_mm_at_R(radii, kh, P)
            ximm[i] = (r2ximm_diff[i*Nr :(i+1)*Nr] + r2xinl[i])/self.radii**2
        #Return the full prediction
        return ximm

    def xi_mm_at_z(self, redshift, params=None):
        raise Exception("xi_mm at arbitrary redshift not implemented yet.")
        assert redshift >= 0, "Redshift must be >= 0."
        assert redshift <= 3, "Redshift must be <= 3."
        ximm = self.predict(params)
        return 0

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import aemulus_data as AD
    #Test it
    test_ind = 0
    cos = AD.test_box_cosmologies()[test_ind][:-1] #remove sigma8
    emu = ximm_emulator(cos)
    ximm = emu.predict(cos)
    #xiz = emu.xi_mm_at_z(1)
    for i in range(Nz):
        plt.loglog(radii, radii**2*ximm[i])
    plt.show()
