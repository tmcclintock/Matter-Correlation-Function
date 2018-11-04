"""An emulator for the matter correlation function based on the Aemulus simulations.
"""
import numpy as np
import george
import scipy.optimize as op

#Training data standard deviation in ln(xi_mm)
ln_xi_stddev = 1.609

#Number of principle components
Npc = 12
