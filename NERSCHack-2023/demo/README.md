# Demo simulation

One of our base simulations for this hackathon is the 3x2v p=1 LAPD gyrokinetic
simulation (see E. Shi et al., "Gyrokinetic continuum simulation of turbulence in
a straight open-field-line plasma", *Journal of Plasma Physics*,
**83**, 1–27 (2017)) on one GPU. We provide an input file and a Perlmutter job
script in this folder. This simulation takes about 8 hours on one NVIDIA A100 GPU
to enter the turbulent stage (i.e. to reach about 1.8 ms). An example of the log
file written by slurm when the simulation completes is in ```slurm-903819.out```.
And an example of the electron density at 3 ms can be plotted with postgkyl
using
```
pgkyl ls2-lapd-3x2v-p1_elc_gridDiagnostics_300.bp -d M0 interp sel --z2 0. pl -a -x 'x (m)' -y 'y (m)' --clabel '$n_e$ (m$^{-3}$)'
```
which results in the figure in ```s2-lapd-3x2v-p1_elc_M0_300_z2eq0p0.png```. 

A faster case can be run by one or more of the
following:
- Commenting out ```coll``` Lua tables in each of the electron and ion tables.
  This turns off collisions, and should reach 5 ms in about 2.4 hours on an A100.
- Reducing ```finalTime``` and ```numFrames``` proportionally.
- Reducing the resolution (```cells```). This may not be advised since we
  haven't tested this sim at lower res yet.

