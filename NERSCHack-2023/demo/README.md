# Demo simulation

One of our base simulations for this hackathon is the 3x2v p=1 LAPD gyrokinetic
simulation (see E. Shi et al., "Gyrokinetic continuum simulation of turbulence in
a straight open-field-line plasma", *Journal of Plasma Physics*,
**83**, 1–27 (2017)) on one GPU. We provide an input file and a Perlmutter job
script in this folder. This simulation takes about 8 hours on one NVIDIA A100 GPU
to enter the turbulent stage. A faster case can be run by commenting out ```coll```
Lua tables in each of the electron and ion tables (this turns off collisions).

