.. sphere documentation master file, created by
   sphinx-quickstart on Wed Nov 14 12:56:58 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sphere's documentation!
==================================
This is the official documentation for the *sphere* discrete element modelling software. It presents the theory behind the discrete element method (DEM), the structure of the software source code, and the Python API for handling simulation setup and data analysis.

*sphere* is developed by Anders Damsgaard Christensen under supervision of David Lunbek Egholm and Jan A. Piotrowski, all of the department of Geology, Aarhus University, Denmark. This document is a work in progress, and is still in an early state. 

Contact: Anders Damsgaard Christensen, http://cs.au.dk/~adc, adc@geo.au.dk


Contents:

.. toctree::
   :maxdepth: 2

   introduction
   dem
   python_api
   cpp



sphere work flow
================
After compiling the \texttt{SPHERE} binary (see sub-section \ref{subsec:compilation}), the procedure of a creating and handling a simulation is typically arranged in the following order:
	\item Setup of particle assemblage, physical properties and conditions using the Python API, described in section \ref{sec:ModelSetup}, page \pageref{sec:ModelSetup}.
	\item Execution of \texttt{SPHERE} software, which simulates the particle behavior as a function of time, as a result of the conditions initially specified in the input file. Described in section \ref{sec:Simulation}, page \pageref{sec:Simulation}.
	\item Inspection, analysis, interpretation and visualization of \texttt{SPHERE} output in Python. Described in section \ref{sec:DataAnalysis}, page \pageref{sec:DataAnalysis}.

\subsection{The \texttt{SPHERE} algorithm}
\label{subsec:spherealgo}
The \texttt{SPHERE}-binary is launched from the system terminal by passing the simulation ID as an input parameter; \texttt{./sphere\_<architecture> <simulation\_ID>}. The sequence of events in the program is the following:
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



Sphere algorithm
================
The \texttt{SPHERE}-binary is launched from the system terminal by passing the simulation ID as an input parameter; \texttt{./sphere\_<architecture> <simulation\_ID>}. The sequence of events in the program is the following:
  
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
\begin{equation}
\label{eq:dt}
\Delta t = 0.17 \min \left( m/\max(k_n,k_t) \right)
\end{equation}
where $m$ is the particle mass, and $k$ are the elastic stiffnesses. This equation ensures that the elastic wave (traveling at the speed of sound) is resolved a number of times while traveling through the smallest particle.

\subsubsection{Host and device memory types}
\label{subsubsec:memorytypes}
A full, listed description of the \texttt{SPHERE} source code variables can be found in appendix \ref{apx:SourceCodeVariables}, page \pageref{apx:SourceCodeVariables}. There are three types of memory types employed in the \texttt{SPHERE} source code, with different characteristics and physical placement in the system (figure \ref{fig:memory}). 

The floating point precision operating internally in \texttt{SPHERE} is defined in \texttt{datatypes.h}, and can be either single (\texttt{float}), or double (\texttt{double}). Depending on the GPU, the calculations are performed about double as fast in single precision, in relation to double precision. In dense granular configuraions, the double precision however results in greatly improved numerical stability, and is thus set as the default floating point precision. The floating point precision is stored as the type definitions \texttt{Float}, \texttt{Float3} and \texttt{Float4}. The floating point values in the in- and output datafiles are \emph{always} written in double precision, and, if necessary, automatically converted by \texttt{SPHERE}.

Three-dimensional variables (e.g. spatial vectors in $E^3$) are in global memory stored as \texttt{Float4} arrays, since these read and writes can be coalesced, while e.g. \texttt{float3}'s cannot. This alone yields a $\sim$20$\times$ performance boost, even though it involves 25\% more (unused) data.

\begin{figure}[htbp]
\label{fig:memory}
\begin{center}
\begin{small}
\begin{tikzpicture}[scale=1, node distance = 2cm, auto]
    % Place nodes
    \node [harddrive] (freadbin) {\textbf{Hard Drive}:\\[1mm] Input binary: \verb"input/"\\\verb"<sim_ID>.bin"};
    		
		\node [processor, below of=freadbin, node distance=2cm] (cpu) {\textbf{CPU}};
		
    \node [harddrive, below of=cpu, node distance=2.5cm] (fwritebin) {\textbf{Hard Drive}:\\[1mm] Output binaries: \texttt{output/}\\ \verb"<sim_ID>."\\\verb"output#..bin"};
    
		\node [mem, right of=cpu, node distance=2.5cm] (host) {\textbf{Host memory} (RAM) E.g. \texttt{x}};
		
		\node [mem, right of=host, node distance=3.5cm] (textures) {\textbf{Textures} (ROM)\\ E.g. \verb"tex_x"};
		\node [mem, above of=textures, node distance=2cm] (device) {\textbf{Device} (RAM)\\ E.g. \verb"dev_x"};
		\node [mem, below of=textures, node distance=2cm] (constant) {\textbf{Constant} (ROM)\\ E.g. \verb"devC_dt"};
		
		
		\node [processor, right of=textures, node distance=4cm] (gpu) {\textbf{GPU}\\multi-\\processors\\\vrule width 2.1cm};
		
	
        \draw node [above of=gpu, shape aspect=1, diamond, draw, node distance=0.3cm] {\vrule width 2.4cm};
        \draw node [below of=gpu, shape aspect=1, diamond, draw, node distance=0.3cm] {\vrule width 2.4cm};
        
        \node [mem, above of=gpu, node distance=3cm] (local) {\textbf{Local registers} per thread\\ (RAM)};

    	\node [mem, below of=gpu, node distance=3cm] (shared) {\textbf{Shared} per grid\\ (RAM) 48kb};
    
    % Place hardware description
    \node [above of=freadbin, node distance=1.5cm] {\large Host system};
    %\node [above of=device, node distance=4cm] {\Large CUDA device};
    \node [at={(7.0,1.5)}] {\large CUDA device};
    
    \node [at={(4.0,0.8)}, rotate=90] {PCIe $\times$16 Gen2};
    \path [draw] (4.3, 2.0) -- (4.3,-0.5);
    \path [draw] (4.3,-3.5) -- (4.3,-6.5);
    
    \node [at={(6.0,-6.3)}] {Off-chip};
    
    \path [draw, gray] (8.0, 0.5) -- (8.0,-0.5);
    \path [draw, gray] (8.0,-4.5) -- (8.0,-6.5);
    
    \node [at={(10.0,-6.3)}] {On-chip};
    
    % Draw lines
    \path [draw, -latex', thick] (freadbin) -- (cpu);
    
    \path [draw, -latex'] (cpu) -- (host);
    \path [draw, -latex'] (host) -- (cpu);

    \path [draw, -latex', dashed] (host) -- (device);
    \path [draw, -latex', dashed] (device) -- (host);

    \path [draw, -latex', dashed] (host) -- (constant);
    \path [draw, -latex', dashed] (constant) -- (host);
    %\path [draw, -latex'] (constant) -- (device);
    
    \path [draw, -latex', dashed] (host) -- (textures);
    \path [draw, -latex', dashed] (textures) -- (host);
    
    %\path [draw, -latex', dashed] (host) -- (shared);
    %\path [draw, -latex', dashed] (shared) -- (host);
    
    \path [draw, -latex'] (device) -- (gpu);
    \path [draw, -latex'] (gpu) -- (device);
    
    \path [draw, -latex'] (textures) -- (gpu);
    \path [draw, -latex'] (gpu) -- (textures);
    \node [at={(7.7,-1.8)}] {\footnotesize Cached};
    \node [at={(7.7,-2.2)}] {\footnotesize reads};
    
    \path [draw, -latex'] (constant) -- (gpu);
    %\path [draw, -latex'] (gpu) -- (constant);
    \node [at={(7.7,-2.9)}, rotate=25] {\footnotesize Cached};
    \node [at={(7.9,-3.2)}, rotate=25] {\footnotesize reads};
    
    \path [draw, -latex'] (shared) -- (gpu);
    \path [draw, -latex'] (gpu) -- (shared);
    \node [at={(9.85,-3.9)}] {\footnotesize Cached reads};
    %\node [at={(8.0,-4.2)}, rotate=45] {\footnotesize reads};
    
    \path [draw, -latex'] (local) -- (gpu);
    \path [draw, -latex'] (gpu) -- (local);
    
    
    %\path [draw, -latex'] (device) -- (shared);

    \path [draw, -latex', thick] (cpu) -- (fwritebin);
    
    % Bandwith text
    \node [at={(4.2,-2.3)}] (host-dev) {8 GB/s};
    %\node [at={(3,-3)}] (PCIe) {(PCIe Gen2)};
    %\node [at={(6,-2.3)}] (dev-dev) {89.6 GB/s};

\end{tikzpicture}
\end{small}

\caption{Flow chart of system memory types and communication paths. RAM: Random-Access Memory (read + write), ROM: Read-Only Memory. Specified communication path bandwidth on test system (2010 Mac Pro w. Quadro 4000 noted.}
\end{center}

\end{figure}


\paragraph{Host memory} is the main random-access computer memory (RAM), i.e. read and write memory accessible by CPU processes, but inaccessible by CUDA kernels executed on the device. 


\paragraph{Device memory} is the main, global device memory. It resides off-chip on the GPU, often in the form of 1--6 GB DRAM. The read/write access from the CUDA kernels is relatively slow. The arrays residing in (global) device memory are prefixed by ``dev_`` in the source code. 

\marginpar{Todo: Expand section on device memory types}

\paragraph{Constant memory} values cannot be changed after they are set, and are used for scalars or small vectors. Values are set in the ``transferToConstantMemory(...)}`` function, called in the beginning of \texttt{gpuMain(\ldots)} in \texttt{device.cu}. Constant memory variables have a global scope, and are prefixed by ``devC_`` in the source code.



%\subsection{The main loop}
%\label{subsec:mainloop}
%The \texttt{SPHERE} software calculates particle movement and rotation based on the forces applied to it, by application of Newton's law of motion (Newton's second law with constant particle mass: $F_{\mathrm{net}} = m \cdot a_{\mathrm{cm}}$). This is done in a series of algorithmic steps, see list on page \pageref{loopstart}. The steps are explained in the following sections with reference to the \texttt{SPHERE}-source file; \texttt{sphere.cu}. The intent with this document is \emph{not} to give a full theoretical background of the methods, but rather how the software performs the calculations.


\subsection{Performance}
\marginpar{Todo: insert graph of performance vs. np and performance vs. $\Delta t$}.
\subsubsection{Particles and computational time}

\subsection{Compilation}
\label{subsec:compilation}
An important note is that the \texttt{C} examples of the NVIDIA CUDA SDK should be compiled before \texttt{SPHERE}. Consult the `Getting started guide`, supplied by Nvidia for details on this step.

\texttt{SPHERE} is supplied with several Makefiles, which automate the compilation process. To compile all components, open a shell, go to the \texttt{src/} subfolder and type \texttt{make}. The GNU Make will return the parameters passed to the individual CUDA and GNU compilers (\texttt{nvcc} and \texttt{gcc}). The resulting binary file (\texttt{sphere}) is placed in the \texttt{SPHERE} root folder. ``src/Makefile`` will also compile the raytracer.





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

