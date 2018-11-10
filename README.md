# Matter-Correlation-Function
An emulator for the matter correlation function, based on the Aemulus simulation suite. The emulator takes in a set of cosmological parameters, and returns the matter correlation function evaluated between [0.1, 75] Mpc/h comoving at fifty radii, and at ten redshifts spaced between z=0 and z=3.

At present, we are currently implementing a method to interpolate the returned curves at arbitrary redshifts, using a weighted average between adjacent snapshots.

Install using
```
python setup.py install
```