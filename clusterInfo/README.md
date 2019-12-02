# Cluster information

For this hackathon we are primarily targeting Tesla V100 GPUs. Here is some
useful information on using the GPUs on some clusters.

# Working with Adroit GPU node

Everyone with a valid PU email ID should have access to Adroit. You
need to enable Duo Authenticate to access Adroit, though, and need to
be VPN-ed to the lab's network (or be within the PU or PPPL network).

Please build the dependencies for Adroit by running:

```
 ./machines/mkdeps.adroit.sh
```

and configure the code using:

```
  ./machines/configure.adroit.sh
```

Once you have installed gkeyll, one can launch a job on the compute node
with a GPU (see the sample job script in the folder). This node has 4
NVIDIA V100 GPUs with 32 GB of memory each. See the specs for the
[V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184). Each
GPU has 80 streaming multiprocessors (SM) and 64 CUDA cores per SM
(and 8 Tensor Cores per SM).

Add this line to your Slurm script to use a V100 GPUs:

```
#SBATCH --gres=gpu:tesla_v100:1
```

You can also logon to the GPU node directly:

```
  ssh adroit-h11g1
```  


To see gory information about the GPU do:

```
  nvidia-smi -q
```

The CUDA dev tools are part of the cudatoolkit/10.1 module. It is
loaded when you configure Gkeyll using the Adroit machine
file. However, you can also load it manually using:

```
  module load cudatoolkit/10.1
```

This will add the nvcc compiler into the path and also define a bunch
of env variables with location of the libraries.
