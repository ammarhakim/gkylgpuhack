# Notes, GPU team meeting: May 19th 2020

- Next meeting will be held on __May 26th at 10:00 am__. We will attempt to complete some or all of the infrastructure work listed below before then.

- Each person responsible for equation objects etc will split the kernels into multiple files with named by polyOrder. This should be done ASAP to assist in the tasks below. __Noah__ will add a flag to the top-level waf gkyl tool to disable compiling all higher than p=1 kernels. This needs reorg of listing the p=1 kernel code into their own lists in each `wscript` then conditionally compiling them as needed. Default will be to build everything.

- Something is not right with CUDA on CentOS 7 nodes on Portal. This needs to be fixed ASAP. (AH sent email to Prentice).

- A templated wrapper will be added to allow calling various C-style kernels from C++ code. AH will do this and provide a pure-C++ driven executable that calls these kernels in a loop. We and mentors can use to study properties of the kernels via device profilers. __Jimmy__ will lead this effort and collect stats and work with mentors to see if we some reorg of the kernels leads to better performance etc.

- A non-templated Eq base class will be introduced. The other Eq objects will derive from this and hook into the kernels provided in the previous tasks. AH will take a shot at this and __Noah__ will work on it also on weekend/next-week.

- Nested ranges are needed to do double indexing: first, given a linear index call `invIndexer` and then feed this into the field's `indexer()` objects to get the linear index. See notes below.

- AH suggested adding a "ghost-object" to all Lua objects that work on the GPU. For example, `CartField` would have a ghost object that encapsulates the various range objects, the pointer to the GPU data etc. Then, this is passed to the GPU code, for example, updaters etc. Key objects like this are (some of these are already wrapped in one form or the other): `Range`, `Grid`, `CartField` and the various updaters we want to put on the device.

- AH says that the base Updater object should be "GPU aware". Then if GPU-specific init() advance() etc are provided by children it should call them. Hence, transparently updaters will run on device if the updater is ported to it, but run on host if not. Sounds magical.

- __Mana__ will explore `binOp`s on device. This needs inverting small matrices: can we not just use some pure-C linear solver code that we just compile and use on the device?

- Once some of this infra work is done and demonstrated on May 26th then each Updater author will start working on moving things to this architectures.
