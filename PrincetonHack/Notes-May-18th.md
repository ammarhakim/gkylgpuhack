# Notes, kickoff meeting with mentors: May 18th 2020

- List of teams and mentors for is on [this page](https://researchcomputing.princeton.edu/gpu-hackathon-2020/list-of-teams).

- We will send the mentors github links and give them write access if they wish. For now we have agreed to use Portal and/or Adriot or Tiger at PU to avoid trouble with Power9 chips. Eventually we will support Power9 chips also but at present the JIT compiler does not fully work on PPC64le (IBM as usual went and changed their endianness on their newer chips. However, some forums claim that Power9 can be used in either endianness but seems almost certain that the OS on Summit is using PPC64le. Perhaps in the future one can imagine using a container technology to get around this issue).

- As homework we will read up on ["CUDA C++ Best Practices Guide"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). Everyone should spend some time on the basic CUDA documentation linked on our private hackathon github page. Some familiarity with the API is also good, though not essential for most of the team. (Though Noah, Mana and A need to be throughly familiar this).

- We will explore if the profilers work with dynamic languages. Once the control goes to the GPU then there seems no reason why the profiler can't get a hold of the code and profile it. However, this remains to be seen. We perhaps have a stand-alone C++ executable that calls the kernels in a loop which the mentors and we can use to run via the profilers.

- Doris asked if we are thinking about multi-GPUs. I think this should be "automatic" if we are careful and design the right abstractions. But I need to think about this more carefully, in particular if we can async the ghost-cell transfer, run the kernels and then complete the transfers. Perhaps it will "just work". Someone mentioned ["CUDA-aware MPI"](https://devblogs.nvidia.com/introduction-cuda-aware-mpi/). Not sure if this is useful but could be on a single node with multiple GPUs.

- After the internal team meeting on Tuesday May 19th we will send a more detailed list to the mentors about what we want to achieve so all of us have some time to go over needed background.
