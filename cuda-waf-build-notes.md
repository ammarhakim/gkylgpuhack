# Notes about the CUDA-aware waf build system

## How the waf system works with CUDA

A new `cutools` module (see `waf_tools/cutools.py`) has been added to the `waf` build system to enable compiling and linking CUDA code. This has rules for how to build various types of files with `nvcc`.

### Compiling C/CUDA code into object files
If `waf` finds a file with a `*.cu` extension, it will compile it using `nvcc`. However, in Gkeyll we have a lot of existing code that lives in `*.cpp` files that we would like to compile with `nvcc`. This can be enabled using `features=nvcc` in the wscript build command, as shown in the example below. This will compile the files with `nvcc -x cu -c -dc ...`, which creates device object files.

### Linking CUDA-compiled object files into shared libraries
To use `nvcc` to link the object files that were compiled with `nvcc` into a shared library, we add `cushlib` to the features list in the wscript build command. This will link the object files into a shared library with `nvcc -shared ...`.

## Getting existing .cpp kernels built with CUDA

To enable existing kernels that reside in .cpp files to be built by CUDA, one must do the following:

**1.** In the header file containing the prototypes for a kernel that needs to be built by CUDA, you must add  
```
#include "GkCudaConfig.h"
```  
to the top of the header file. You must also add the preprocessors  
```
__host__ __device__
```  
to the beginning of each prototype that should be built with CUDA. For an example, see `Updater/momentCalcData/DistFuncMomentCalcModDecl.h`. 
The `MomentCalc1x1vSer_M0_P1` function has been enabled for CUDA compilation via
```
__host__ __device__ void MomentCalc1x1vSer_M0_P1(const double *w, const double *dxv, const double *f, double *out);
```

**2.** You must also add the `__host__ __device__` preprocessors to the kernel in the .cpp file. 
For example, in `Updater/momentCalcData/DistFuncMomentCalcSer1x1v.cpp`, we have
```
__host__ __device__ void MomentCalc1x1vSer_M0_P1(const double *w, const double *dxv, const double *f, double *out)
{
  const double volFact = dxv[1]/2;
  out[0] += 1.414213562373095*f[0]*volFact;
  out[1] += 1.414213562373095*f[1]*volFact;
}
```
**Other than adding the preprocessors, no other changes to the kernel are needed. With these changes, the** `MomentCalc1x1vSer_M0_P1` **function
can now be called from within a CUDA** `__global__` **kernel.**

**3.** You must also add a new section in the `wscript` file containing source (.cpp) files to be built by CUDA. For example, in `Updater/wscript`, we have
```
    # CUDA specific code
    if bld.env['CUTOOLS_FOUND']:
        cusources = 'momentCalcData/DistFuncMomentCalcSer1x1v.cpp momentCalcData/DistFuncMomentCalcDeviceCommon.cu momentCalcData/DistFuncMomentCalcSer1x1vDevice.cu'

        bld(source = cusources,
            name = 'updater_cu', target='updater_cu',
            includes = '. ../Cuda ../Grid ../Lib momentCalcData', features = 'cxx nvcc cushlib'
        )
```

where we have added `momentCalcData/DistFuncMomentCalcSer1x1v.cpp` to the `cusources` list, since this is the file that contains the 
`MomentCalc1x1vSer_M0_P1` kernel that we instrumented with CUDA preprocessors above. As we enable more of the momentCalc kernel files for 
CUDA via the processes above, the `cusources` list will grow.

**Note that the `momentCalcData/DistFuncMomentCalcSer1x1v.cpp` file will now be built twice: 
once with the C++ compiler (via the standard build scheme), and another time with ``nvcc``.**


