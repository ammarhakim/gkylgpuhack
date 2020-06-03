Random dev notes as we move Gkeyll to GPU...
- dynamic memory allocation inside kernels absolutely **kills** performance. avoid it at all costs!
- C++ objects that contain virtual methods cannot be passed into a device kernel. you can instantiate objects with virtual methods inside a kernel, but this may affect performance.
- C++ std library methods are extremely limited and/or non-existent on device. best to avoid them.
- nvcc cannot do device-code linking across a shared library boundary. need to nvlink all dependent object files into single shared library or object file.
- ...