# Cluster information

For this hackathon we are primarily targeting Tesla V100 GPUs. Here is some
useful information on using the GPUs on some clusters.

# Working with Adroit GPU node

Everyone with a valid PU email ID should have access to Adroit. You
need to enable Duo Authenticate to access Adroit, though, and need to
be VPN-ed to the lab's network (or be within the PU or PPPL network).

## Useful Adroit/SLURM commands

List the nodes: `snodes`.

List the gpu nodes: `snodes all gpu`.

Check your diskspace availability: `checkquota`.

## Building Gkeyll on Adroit
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

The CUDA dev tools are part of the cudatoolkit/10.1 module. It is
loaded when you configure Gkeyll using the Adroit machine
file. However, you can also load it manually using:

```
  module load cudatoolkit/10.1
```

This will add the nvcc compiler into the path and also define a bunch
of env variables with location of the libraries.

## Submitting batch jobs

Add this line to your Slurm script to use a V100 GPUs:

```
#SBATCH --gres=gpu:tesla_v100:1
```

## Interactive jobs

### Running CPU code
If you are only going to run cpu/host code, then you can request an interactive job with
```
salloc -N 1 -n 1 --gres=gpu:tesla_v100:1 -t 00:30:00 --mem=8G
```
where `-N` indicates the number of nodes, `-n` the number of (MPI) tasks, `-t` the duration of the job, and `--mem` the amount of memory you desire. Note that if you are only running CPU code you don't really need to speciy the resource `--gres=gpu:tesla_v100:1`.

### Running GPU code
The prefered way to run GPU code interactively is to request a job with
```
salloc -N 1 --gres=gpu:tesla_v100:1 -t 00:30:00 srun -n1 -N1 --mem-per-cpu=0 --pty --preserve-env --mpi=none --gres=gpu:0 /bin/bash
```
The idea here is that one has to start an interactive job that doesn’t consume the GPU, and once in the interactive session do use the GPU with your run command:
```
srun gkyl cudaFile.lua
```
You may have to provide the direct path to the `gkyl` command, depending on your local configuration.

In principle you can also logon to the GPU node directly:
```
  ssh adroit-h11g1
```  
However, one should not ssh onto a node and run code without requesting an interactive job as explained above. 

Sometimes it is useful to ssh into a node to see what the current GPU utilization is (e.g. to see if there's a GPU available) with
```
  nvidia-smi
```
or o see gory information about the GPU with
```
  nvidia-smi -q
```

# Working with Portal GPU node

Everyone that can VPN into PPPL should have access to the Portal cluster at PPPL.
You can log in to portal with the command:

```
  ssh ppplusername@portalc7.pppl.gov
```
**Note that we are now using the CentOS 7 login node portalc7. This is because the GPU node that
we want to use (gpusrv02) is running CentOS 7.**

Once logged in to Portal, you will need to modify whichever profile file is sourced
to set up your modules (see which file is sourced in ~/.login). For example, modify
~/.cshrc to load:

```
  module load gcc
  module load openmpi
  module swap gcc gcc/8.4.0
  module load cuda/10.2
  module load git
```
Note that we need load the gcc(9) module only so that we can load the openmpi module. We then swap to the gcc/8.4.0 module.
**Do not use the default gcc module, as it gives gcc/9.3.0, which is incompatible with the CUDA runtime.**

Also ensure that /sbin/ in in your PATH. This is needed to get
ldconfig which is needed by luajit to properly create its shared
libraries.

Even though we load these modules in the mkdeps and configure shell scripts, we still need to
modify the profile file to load the modules, since calling module load from within a shell
script cannot change the environment outside of the shell script.

After setting up your modules run the mkdeps file in the machines directory:

```
 ./machines/mkdeps.portal.sh
```

and then the configure file:

```
  ./machines/configure.portal.sh
```

Then build with the usual

```
./waf build install
```

to finish the installation of Gkyl. To access the Nvidia Volta node, ssh directly onto the node,

```
  ssh gpusrv02
```

and to check that the installation has been successful, run the test_Cuda.lua unit test,

```
  ~/gkylsoft/gkyl/bin/gkyl ~/gkyl/Unit/test_Cuda.lua
```

## If luajit fails to link due to -fPIC flag

Often, the compiler will complain that libluajit.a can't be used to
build shared library. This is due to missing ldconfig. If you see
this, then we need to manually link the shared libraries for Lua. We
can do this by,

```
 cd ~/gkylsoft/luajit/lib
 ln -sf libluajit-5.1.so.2.1.0 libluajit-5.1.so && ln -sf libluajit-5.1.so.2.1.0 libluajit-5.1.so.2
```


