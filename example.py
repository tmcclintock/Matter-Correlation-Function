from ximm_emulator import *
import matplotlib.pyplot as plt
#cos is an array with:
#Omega_b*h^2, Omega_cdm*h^2, w_0, n_s, ln(10^10 * A_s), H0, N_eff
cos = np.array([ 2.32629e-02, 1.07830e-01, -7.26513e-01, 9.80515e-01, 3.03895e+00, 6.32317e+01, 2.95000e+00])

#Make the emulator
emu = ximm_emulator(cos)
#Predict xi_mm. This always comes out at these ten red redshifts at these radii
ximm = emu.predict(cos)
radii = emu.get_radii()
redshifts = emu.get_redshifts()

for i in range(len(redshifts)):
    plt.loglog(radii, radii**2*ximm[i])
plt.show()
