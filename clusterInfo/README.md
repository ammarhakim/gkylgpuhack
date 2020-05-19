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
and see what the current GPU utilization is (e.g. to see if there's a GPU available) with
```
  nvidia-smi
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

# Working with Portal GPU node

Everyone that can VPN into PPPL should have access to the Portal cluster at PPPL.
You can log in to portal with the command:

```
  ssh ppplusername@portal.pppl.gov
```

Once logged in to Portal, you will need to modify whichever profile file is sourced
to set up your modules (see which file is sourced in ~/.login). For example, modify
~/.cshrc to load:

```
  module load gcc/7.3.0
  module load openmpi
  module load cuda
  module load git
```

Also ensure that /sbin/ in in your PATH. This is needed to get
ldconfig which is needed by luajit to properly create its shared
libraries.

This modification is necessary as the module load command does not work inside a shell script,
such as those we use to build the dependencies and configure the system. Note that the module load
commands are included anyway in the mkdeps.portal.sh and configure.portal.sh files as a reference.

Although Portal contains the Intel compiler, gcc 7.3 and cuda 10.1 are a consistent build on Portal 
(DO NOT USE gcc 9.1), and preferable for ease-of-build. Likewise it is recommended that you load git
if doing any development work as the git in /usr/bin/ is too old to make commits, whereas the module git
is newer.

After setting up your modules run the mkdeps file in the machines directory:

```
 ./machines/mkdeps.portal.sh
```


```
  ./machines/configure.portal.sh
```

and finish the installation of Gkyl. To access the Nvidia Volta node, ssh directly onto the node,

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


