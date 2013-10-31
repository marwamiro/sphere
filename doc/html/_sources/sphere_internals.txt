sphere internals
================

The *sphere* executable has the following options:

.. command-output:: ../../sphere --help

The most common way to invoke *sphere* is however via the Python API (e.g. :py:func:`sphere.run`, :py:func:`sphere.render`, etc.).




\subsection{The *sphere* algorithm}
\label{subsec:spherealgo}
The *sphere*-binary is launched from the system terminal by passing the simulation ID as an input parameter; \texttt{./sphere\_<architecture> <simulation\_ID>}. The sequence of events in the program is the following:
#. System check, including search for NVIDIA CUDA compatible devices (\texttt{main.cpp}).
  
#. Initial data import from binary input file (\texttt{main.cpp}).
  
#. Allocation of memory for all host variables (particles, grid, walls, etc.) (\texttt{main.cpp}).
  
#. Continued import from binary input file (\texttt{main.cpp}).

#. Control handed to GPU-specific function \texttt{gpuMain(\ldots)} (\texttt{device.cu}).
  
#. Memory allocation of device memory (\texttt{device.cu}).
  
#. Transfer of data from host to device variables (\texttt{device.cu}).
  
#. Initialization of Thrust\footnote{\url{https://code.google.com/p/thrust/}} radix sort configuration (\texttt{device.cu}).
  
#. Calculation of GPU workload configuration (thread and block layout) (\texttt{device.cu}).

#. Status and data written to \verb"<simulation_ID>.status.dat" and \verb"<simulation_ID>.output0.bin", both located in \texttt{output/} folder (\texttt{device.cu}).
  
#. Main loop (while \texttt{time.current <= time.total}) (functions called in \texttt{device.cu}, function definitions in seperate files). Each kernel call is wrapped in profiling- and error exception handling functions:
  
  
  #. \label{loopstart}CUDA thread synchronization point.
  
  #. \texttt{calcParticleCellID<<<,>>>(\ldots)}: Particle-grid hash value calculation (\texttt{sorting.cuh}).
  
  #. CUDA thread synchronization point.
  
  #. \texttt{thrust::sort\_by\_key(\ldots)}: Thrust radix sort of particle-grid hash array (\texttt{device.cu}).
  
  #. \texttt{cudaMemset(\ldots)}: Writing zero value (\texttt{0xffffffff}) to empty grid cells (\texttt{device.cu}).
  
  #. \texttt{reorderArrays<<<,>>>(\ldots)}: Reordering of particle arrays, based on sorted particle-grid-hash values (\texttt{sorting.cuh}).
  
  #. CUDA thread synchronization point.

  #. Optional: \texttt{topology<<<,>>>(\ldots)}: If particle contact history is required by the contact model, particle contacts are identified, and stored per particle. Previous, now non-existant contacts are discarded (\texttt{contactsearch.cuh}).
  
  #. CUDA thread synchronization point.
  
  #. \texttt{interact<<<,>>>(\ldots)}: For each particle: Search of contacts in neighbor cells, processing of optional collisions and updating of resulting forces and torques. Values are written to read/write device memory arrays (\texttt{contactsearch.cuh}).
  
  #. CUDA thread synchronization point.
    
  #. \texttt{integrate<<<,>>>(\ldots)}: Updating of spatial degrees of freedom by a second-order Taylor series expansion integration (\texttt{integration.cuh}).

  #. CUDA thread synchronization point. 

  #. \texttt{summation<<<,>>>(\ldots)}: Particle contributions to the net force on the walls are summated (\texttt{integration.cuh}).

  #. CUDA thread synchronization point.

  #. \texttt{integrateWalls<<<,>>>(\ldots)}: Updating of spatial degrees of freedom of walls (\texttt{integration.cuh}).
  
  #. Update of timers and loop-related counters (e.g. \texttt{time.current}), (\texttt{device.cu}).
  
  #. If file output interval is reached:
  
	\item Optional write of data to output binary (\verb"<simulation_ID>.output#..bin"), (\texttt{file\_io.cpp}).
	\item Update of \verb"<simulation_ID>.status#..bin" (\texttt{device.cu}).
  
      \item Return to point \ref{loopstart}, unless \texttt{time.current >= time.total}, in which case the program continues to point \ref{loopend}.
  
  
#. \label{loopend}Liberation of device memory (\texttt{device.cu}).

#. Control returned to \texttt{main(\ldots)}, liberation of host memory (\texttt{main.cpp}).
  
#. End of program, return status equal to zero (0) if no problems where encountered.


Numerical algorithm
-------------------
The *sphere*-binary is launched from the system terminal by passing the simulation ID as an input parameter; \texttt{./sphere\_<architecture> <simulation\_ID>}. The sequence of events in the program is the following:
  
#. System check, including search for NVIDIA CUDA compatible devices (\texttt{main.cpp}).
  
#. Initial data import from binary input file (\texttt{main.cpp}).
  
#. Allocation of memory for all host variables (particles, grid, walls, etc.) (\texttt{main.cpp}).
  
#. Continued import from binary input file (\texttt{main.cpp}).

#. Control handed to GPU-specific function \texttt{gpuMain(\ldots)} (\texttt{device.cu}).
  
#. Memory allocation of device memory (\texttt{device.cu}).
  
#. Transfer of data from host to device variables (\texttt{device.cu}).
  
#. Initialization of Thrust\footnote{\url{https://code.google.com/p/thrust/}} radix sort configuration (\texttt{device.cu}).
  
#. Calculation of GPU workload configuration (thread and block layout) (\texttt{device.cu}).

#. Status and data written to \verb"<simulation_ID>.status.dat" and \verb"<simulation_ID>.output0.bin", both located in \texttt{output/} folder (\texttt{device.cu}).
  
#. Main loop (while \texttt{time.current <= time.total}) (functions called in \texttt{device.cu}, function definitions in seperate files). Each kernel call is wrapped in profiling- and error exception handling functions:
  
  
  #. \label{loopstart}CUDA thread synchronization point.
  
  #. \texttt{calcParticleCellID<<<,>>>(\ldots)}: Particle-grid hash value calculation (\texttt{sorting.cuh}).
  
  #. CUDA thread synchronization point.
  
  #. \texttt{thrust::sort\_by\_key(\ldots)}: Thrust radix sort of particle-grid hash array (\texttt{device.cu}).
  
  #. \texttt{cudaMemset(\ldots)}: Writing zero value (\texttt{0xffffffff}) to empty grid cells (\texttt{device.cu}).
  
  #. \texttt{reorderArrays<<<,>>>(\ldots)}: Reordering of particle arrays, based on sorted particle-grid-hash values (\texttt{sorting.cuh}).
  
  #. CUDA thread synchronization point.

  #. Optional: \texttt{topology<<<,>>>(\ldots)}: If particle contact history is required by the contact model, particle contacts are identified, and stored per particle. Previous, now non-existant contacts are discarded (\texttt{contactsearch.cuh}).
  
  #. CUDA thread synchronization point.
  
  #. \texttt{interact<<<,>>>(\ldots)}: For each particle: Search of contacts in neighbor cells, processing of optional collisions and updating of resulting forces and torques. Values are written to read/write device memory arrays (\texttt{contactsearch.cuh}).
  
  #. CUDA thread synchronization point.
    
  #. \texttt{integrate<<<,>>>(\ldots)}: Updating of spatial degrees of freedom by a second-order Taylor series expansion integration (\texttt{integration.cuh}).

  #. CUDA thread synchronization point. 

  #. \texttt{summation<<<,>>>(\ldots)}: Particle contributions to the net force on the walls are summated (\texttt{integration.cuh}).

  #. CUDA thread synchronization point.

  #. \texttt{integrateWalls<<<,>>>(\ldots)}: Updating of spatial degrees of freedom of walls (\texttt{integration.cuh}).
  
  #. Update of timers and loop-related counters (e.g. \texttt{time.current}), (\texttt{device.cu}).
  
  #. If file output interval is reached:
  
	* Optional write of data to output binary (\verb"<simulation_ID>.output#..bin"), (\texttt{file\_io.cpp}).
        * Update of \verb"<simulation_ID>.status#..bin" (\texttt{device.cu}).
  
  #. Return to point \ref{loopstart}, unless \texttt{time.current >= time.total}, in which case the program continues to point \ref{loopend}.
  
  
#. \label{loopend}Liberation of device memory (\texttt{device.cu}).

#. Control returned to \texttt{main(\ldots)}, liberation of host memory (\texttt{main.cpp}).
  
#. End of program, return status equal to zero (0) if no problems where encountered.



The length of the computational time steps (\texttt{time.dt}) is calculated via equation \ref{eq:dt}, where length of the time intervals is defined by:

.. math::
   \Delta t = 0.075 \min \left( m/\max(k_n,k_t) \right)

where :math:`m` is the particle mass, and :math:`k` are the elastic stiffnesses. 
The time step is set by this relationship in :py:func:`initTemporal`. 
This equation ensures that the elastic wave (traveling at the speed of sound) is resolved a number of times while traveling through the smallest particle.

\subsubsection{Host and device memory types}
\label{subsubsec:memorytypes}
A full, listed description of the *sphere* source code variables can be found in appendix \ref{apx:SourceCodeVariables}, page \pageref{apx:SourceCodeVariables}. There are three types of memory types employed in the *sphere* source code, with different characteristics and physical placement in the system (figure \ref{fig:memory}). 

The floating point precision operating internally in *sphere* is defined in \texttt{datatypes.h}, and can be either single (\texttt{float}), or double (\texttt{double}). Depending on the GPU, the calculations are performed about double as fast in single precision, in relation to double precision. In dense granular configuraions, the double precision however results in greatly improved numerical stability, and is thus set as the default floating point precision. The floating point precision is stored as the type definitions \texttt{Float}, \texttt{Float3} and \texttt{Float4}. The floating point values in the in- and output datafiles are \emph{always} written in double precision, and, if necessary, automatically converted by *sphere*.

Three-dimensional variables (e.g. spatial vectors in `E^3`) are in global memory stored as \texttt{Float4} arrays, since these read and writes can be coalesced, while e.g. \texttt{float3}'s cannot. This alone yields a `\sim`20`\times` performance boost, even though it involves 25\% more (unused) data.


\paragraph{Host memory} is the main random-access computer memory (RAM), i.e. read and write memory accessible by CPU processes, but inaccessible by CUDA kernels executed on the device. 


\paragraph{Device memory} is the main, global device memory. It resides off-chip on the GPU, often in the form of 1--6 GB DRAM. The read/write access from the CUDA kernels is relatively slow. The arrays residing in (global) device memory are prefixed by ``dev_`` in the source code. 

\marginpar{Todo: Expand section on device memory types}

\paragraph{Constant memory} values cannot be changed after they are set, and are used for scalars or small vectors. Values are set in the ``transferToConstantMemory(...)}`` function, called in the beginning of \texttt{gpuMain(\ldots)} in \texttt{device.cu}. Constant memory variables have a global scope, and are prefixed by ``devC_`` in the source code.



%\subsection{The main loop}
%\label{subsec:mainloop}
%The *sphere* software calculates particle movement and rotation based on the forces applied to it, by application of Newton's law of motion (Newton's second law with constant particle mass: `F_{\mathrm{net}} = m \cdot a_{\mathrm{cm}}`). This is done in a series of algorithmic steps, see list on page \pageref{loopstart}. The steps are explained in the following sections with reference to the *sphere*-source file; \texttt{sphere.cu}. The intent with this document is \emph{not} to give a full theoretical background of the methods, but rather how the software performs the calculations.


\subsection{Performance}
\marginpar{Todo: insert graph of performance vs. np and performance vs. `\Delta t`}.
\subsubsection{Particles and computational time}

\subsection{Compilation}
\label{subsec:compilation}
An important note is that the \texttt{C} examples of the NVIDIA CUDA SDK should be compiled before *sphere*. Consult the `Getting started guide`, supplied by Nvidia for details on this step.

*sphere* is supplied with several Makefiles, which automate the compilation process. To compile all components, open a shell, go to the \texttt{src/} subfolder and type \texttt{make}. The GNU Make will return the parameters passed to the individual CUDA and GNU compilers (\texttt{nvcc} and \texttt{gcc}). The resulting binary file (\texttt{sphere}) is placed in the *sphere* root folder. ``src/Makefile`` will also compile the raytracer.


C++ reference
-------------
.. doxygenclass:: DEM
   :members:


