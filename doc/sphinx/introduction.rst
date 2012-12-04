Introduction
============
The \texttt{SPHERE}-software is used for three-dimensional discrete element method (DEM) particle simulations. The source code is written in \texttt{C++} and \texttt{CUDA C}, and compiled by the user. The main computations are performed on the graphics processing unit (GPU) using NVIDIA's general purpose parallel computing architecture, CUDA. 

The ultimate aim of the \texttt{SPHERE} software is to simulate soft-bedded subglacial conditions, while retaining the flexibility to perform simulations of granular material in other environments. The requirements to the host computer are:
\begin{itemize}
  \item UNIX, Linux or Mac OS X operating system.
  \item GCC, the GNU compiler collection.
  \item A CUDA-enabled GPU with compute capability 1.1 or greater\footnote{See \url{http://www.nvidia.com/object/cuda_gpus.html} for an official list of NVIDIA CUDA GPUs.}.
  \item The CUDA Developer Drivers and the CUDA Toolkit\footnote{Obtainable free of charge from \url{http://developer.nvidia.com/object/cuda_3_2_downloads.html}}.
\end{itemize}
For simulation setup and data handling, a Python distribution of a recent version is essential. Required Python modules include Numpy\footnote{\url{http://numpy.scipy.org}}. There is however no requirement of Python on the computer running the \texttt{SPHERE} calculations, i.e. model setup and data analysis can be performed on a separate device. Command examples in this document starting with the symbol '\verb"$"' are executed in the shell of the operational system, and '\verb">>>"' means execution in Python. All numerical values in this document, the source code, and the configuration files are typeset with strict respect to the SI unit system.


