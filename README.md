# What

This repo contains some documents from the Gkeyll GPU
Hackathon. Please add GPU tutorial links and other useful materials
here.

# Links to useful GPU tutorials and papers

- See Princeton University [GPU Intro
  github](https://github.com/PrincetonUniversity/gpu_programming_intro)
  page.

- The most comprehensive documentation is [NVIDIA's C Programing
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

- Paper with gory details on [Volta GPU Architectures via
  Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)

- Blog post on implementing [finite-difference codes on
  GPUs](https://devblogs.nvidia.com/finite-difference-methods-cuda-cc-part-1/)

# Logging onto Adroit GPU node

All people with a valid PU email ID should have access to Adroit. You
need to enable Duo Authenticate to access Adroit, though, and need to
be VPN-ed to the lab's network (or be within the PU or PPPL
network). 

Please build the dependencies for Adroit by running:

 ./machines/mkdeps.adroit.sh

and configure the code using:

 ./machines/configure.adroit.sh

Note that the Adroit node adroit-h11g1 has the Volta GPU that we wish
to use. This node has 4 NVIDIA V100 GPUs with 32 GB of memory
each. See the specs for the
[V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184). Each
GPU has 80 streaming multiprocessors (SM) and 64 CUDA cores per SM
(and 8 Tensor Cores per SM).

Add this line to your Slurm script to use a V100 GPUs:

```
#SBATCH --gres=gpu:tesla_v100:1
```

You can logon to the GPU node:

  ssh adroit-h11g1


To see information about the GPU installed you can do:

  nvidia-smi -q
