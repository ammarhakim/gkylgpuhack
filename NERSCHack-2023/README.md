# 2023 NERSC GPU hackathon

The goal of participating in this GPU hackathon is to:

1. Allow newer members of our team to develop GPU programming skills (in the context of Gkeyll).
2. Develop further GPU capabilities for the gyrokinetic solver in Gkeyll.
3. Measure, understand and, if possible, enhance the performance scaling of the gyrokinetic solver in Gkeyll.

A summary of the gyrokinetic model and some code capabilities can be found in `gkeyll_gyrokinetics_summary.pdf`.

## On the Gkeyll code

The Gkeyll code consists 4 github repositories: gkylcas, gkylzero, gkyl and postgkyl. In this hackathon we
will primarily use the [gkylzero](https://github.com/ammarhakim/gkylzero/) and
[gkyl](https://github.com/ammarhakim/gkyl/) repositories, and one can install postgkyl via conda (see the
[gkeyll docs](https://gkeyll.readthedocs.io/en/latest/). The gkylzero code is a compiled C (and C++ for Cuda)
layer, which builds a library used by the gkyl code; gkyl is meant to just be a dynamic driver written in Lua,
but at the moment some legacy C++ kernels, MPI and NCCL remain there (we wish to move these to gkylzero).

**On the streets you may hear people saying g0 to refer to gkylzero, g2 to refer to gkyl, and g0g2 to mean
g2 calling g0 under the hood**.

In this hackathon we will use g0 and g2 in the following branches:

- g0: gpuHackathon_2023
- g2: g0-merge_gpuHackathon_2023

Instructions for installing the codes is in the README of each of these. We will primarily work on Perlmutter
using the `ntrain4` and `ntrain4_g` accounts we've been given access to for this event, and on Princeton's
stellar-amd cluster.

## Hackathon dev tasks

We identified the following components of the code that we wish to develop during the hackathon, each listing
the people that will participate in it:

1. Neutral models (charge exchange, ionization, recycling): Tess Bernard, Jimmy Juno.
2. Moving MPI/NCCL from g2 to g0: Manaure Francisquez, Ammar Hakim.
3. Poisson solver for simulations w/ multiple GPUs: Maxwell Rosen, Tony Qian, Manaure Francisquez.
4. Twist-shift BCs: Akash Shukla, Manaure Francisquez.
5. Fixing Maxwellians: Dingyun Liu, Ammar Hakim.
6. Measuring performance scaling: TBD, likely multiple people.

We expand on some of these below and/or on other subfolders.

### Moving MPI/NCCL from g2 to g0

Presently the MPI and NCCL used by g2 for communication in parallel (distributed) simulations is in g2,
wrapping these libraries in Lua. However some work has begun to create an object in g0 which manages
communication (`gkyl_comm`), and uses MPI/NCCL in g0, not in g2. This is preferable since the Lua
wrapping of MPI/NCCL is cumbersome and ambiguous.

This work needs to be completed, possibly before attempting to adapt Poisson solvers to work in simulations
with multiple GPUs, perhaps by:

- Wrapping `gkyl_comm` in g2 as it is now, and try to replace some of the
  CPU/MPI operations currently done with g2's MPI.
- Extend `gkyl_comm` with features it doesn't yet have that are needed for
  CPU/MPI runs.
- Extend `gkyl_comm` to support NCCL.

### Poisson solver for multi-GPU simulations

Presently the gyrokinetic solver uses g0's `fem_poisson_perp` solver to solver a 2D perpendicular
Poisson problem on x-y planes, and the `fem_parproj` solver to solve a 1D FEM projection problem
(smoothing, or continuity enforcing) in the z direction. Each of these are linear problems that
lead to a system of equations whose matrix needs to be inverted/factorized to solve it. Presently
we do this either on a single CPU with SuperLU, or on a single GPU with cuSolver.

We'd like to run simulations in which the RHS of this linear problem is distributed across
multiple GPUs, and have ths solution also be distribution across GPUs. It's unclear what are the
steps that need to be followed here, but some useful ones could be:

- Explore if multi-GPU cuSolver has what we need and is mantained/developed by
  NVIDIA still.
- Explore the use of the [ginkgo](https://ginkgo-project.github.io/) library
  to solve multi-GPU linear problems.
- Add infrastructure which allgathers (see Mpi_Allgather) the RHS of the linear
  problem, so that each GPU can solve the whole problem.

### Twist-shift BCs

Twist-shift boundary conditions (BCs) are used in tokamak simulations. See an
[arxiv paper](https://arxiv.org/abs/2110.02249) we wrote on this. Ultimately the BC is applied as
series of small matrix-vector multiplications; i.e. the solution in one cell is a linear combination
of the distribution in several other cells (vectors), with linear coefficients (matrices) that
result from DG interpolations.

Presently this is entirely in g2. We'd like to move the TwistShift updater to g0, but
perhaps the first thing to do is to simply have g2 compute those matrices but have g0
store them, and have g0 apply them via these mat-vec multiples.

### Measuring performance

We need to understand where g2 and g0 are spending their time, and how this changes with problem
size and with number of GPUs. At first, when the gyrokinetic solver doesn't run with multiple
GPUs, we may focus on:

- Going through the g2 App and making sure the timers are set up correctly.
- Measure performance of single GPU runs.
