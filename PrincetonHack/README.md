# Notes for use in the Princeton GPU Hackathon, June 2020

Please add notes to this file/directory for things specific to the Princeton PU Hackathon to be held in June 2020. Important first steps:

- Please __register__ at this website: [Registration](https://docs.google.com/forms/d/e/1FAIpQLSfenTYRopA83xixsLQ2g3kJztokgB9GH-3dtSRM-nDAeJH0kQ/viewform). 

- You can get access to Accent training system. See information at [Accent](https://bit.ly/ptown_gpuhack2020_ascent).

- We will also likely use our own Portal resources for the hackathon. Those who have PPPL IDs, please make sure you have access to Portal and can build the code on Portal. Machine files are checked in also. __Petr__ I can get you access to Portal also.

# Division of labor

- __Ammar__ Overall lead, C++ looping infrastructure, CUDA/C++/Lua bindings and assitance to various parts of the code.

- __Noah__ Overall co-lead; build and infrastructure; gyrokinetics, including GK moments. Poisson solver is lower priority, though it is clearly needed to get the full GK solver to work.

- __Jimmy__ Vlasov-Maxwell lead. Valsov Eq object; Maxwell Eq object; current accumulation; Vlasov Moments

- __Mana__ Moment calculations; binOps and related functions; supporting code in VM and GK code

- __Petr__ LBO kernels and various RDG based kernels ports.

- __Liang__ Fluid kernels; Euler and Tenmoment objects; Source updater

# GPU Work Phases (Old Notes from Ammar)

## Phase I: Infrastructure work, independent of GPUs:

- Getting the fields to work between Lua and C++ transparently. This needs some work in making the Range, Grid and Field objects transparent between Lua and C++. This is not hard as the C++ side of thing needs access only to very few things (no setup, sync or IO or things like that are needed). So the C++ side can be very simple collection of methods that just call things setup in Lua.

- Getting a "Universal Looping" mechanism implemented in C++. Basically I envision three looping mechanisms: first, over cells. Second, over faces+cells. Third: over configuration space with inner loop over velocity space, (moment calculators). I am planning a C++ std library style loops that would take C++ functions as inputs. These functions could be Lua supplied C or CUDA kernels, or other kernels or code written by hand in C++. I need to understand how CUDA looping works to get this correct, but I think if we allow composition at the Lua level we can dynamically pick the looping mechanism and kernels by detecting the platform in the Lua layer.

## Phase II: Updaters with new looping mechanism:

- Move all our major updaters to this new looping mechanisms. These are: HyperDistCont, DistFuncMomentCalc, DistFuncIntegratedMomentCalc. Also, we need to move things like accumulating currents that are done in regular Lua code to these loop structures too.

- Regenerate kernels to be cell-based and not face+cell based (fuse the surface and vol kernels into one).

- Test everything and make sure it works on all platforms of interest.

For the Lua <-> C++ bridge I am planning to use the sol3 library (see https://github.com/ThePhD/sol2). This is an amazing library. BUT it needs c++17 support. At one point I was using this library in G2 but then we found that the Intel C++ compilers were barfing on some of the code. (ICE messages). However, that was 2 years ago and perhaps everything is fine now. We can test. We can simply abondon Intel Compilers also.

## Phase III: GPUs specific work

- Have Maxima generate CUDA code. We will have cleaned up the Maxima code in Phase II and so this should be easy, I think.

- Figure out the hybrid communication patterns: write code to copy skin cells to CPU, do MPI and then copy data to GPU and set ghost cells.

- Copy data to CPU for IO. I am thinking about some sort of asynchronous IO mechanism, in which we would trigger a copy to CPU but immediately return and continue with advance loop while transfer and IO is complete. I do not know if this is possible, but if it is, it will be very useful. (That is, trigger a copy and keep going).


