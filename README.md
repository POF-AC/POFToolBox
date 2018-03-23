# POFToolBox
Toolset for modeling the transmission behavior of SI-POF


A. Introduction
------------

The aim of this project is a realistic modeling of the transmission behavior of step-index poylmer optical fiber.

Step-index polymer optical fibers (SI-POF) are deployed in both analog sensing and data transmission systems. 
The optical transmission behavior of these fibers is complex and affected by intrinsic influences like modal dispersion, scattering and attenuation as well as extrinsic influences like the launching condition and the angular sensitivity of the receiver. Since a proper modeling of the transmission behavior is important to evaluate the suitability of the fiber for a specific application, this project contains a novel fiber model which considers all the previously mentioned impacts. Furthermore the model distinguishes scattering and attenuation for propagating rays not only by their propagating angle Theta_z but also by the skewness Theta_phi. It is therefore possible to distinguish between guided, tunneling and refracted modes. The model uses combined data for scattering and attenuation and computes the impulse response of the transmission system which can be transferred to the frequency domain to derive the amplitude and phase response.

WARNING

The entire project is developed on Linux and has never been tested on any other platform. While most of the code should run on other operating systems with little to no change, some files will need adjustments. For example the file GnuplotInterface.java expects the gnuplot executable to reside in "usr/bin/gnuplot". Furthermore the scripts to build the cudacore are shell scripts.

WARNING 2

You will need to build the model from source because no binaries are provided. This is not intended to hassle you but especially working with the CUDA implementation requieres you to adjust the source code (see section C).


B. Dependencies
------------
1. The model uses JGnuplot to plot its data. In order to work you need to have Gnuplot installed as well.

http://jgnuplot.sourceforge.net/

2. CUDA / JCuda

If you want to use NVIDIAs CUDA for the matrix transitions you need the CUDA Toolkit to compile the computation core:

https://developer.nvidia.com/cuda-toolkit

For accessing the computation core from Java this project relies on JCuda:

http://www.jcuda.org/

3. Scatter data

The model requiers scatter data in order to operate. As of now we only offer scatter data for a 16 cm piece of the SI-POF Asahi DB-1000:

http://www.asahi-kasei.co.jp/ake-mate/pof/en/product/data-communication.html

Be aware that these data are not provided by Asahi nor are they involved in this project. The data were obtained from a far field measurement at the POF-AC and may contain errors.

Unpack the file AsahiDB1000.tar.bz2 from the folder scatterfiles. The path to the scatterfiles has to be considered in the source code.

C. Compiling the model and working with it
---------------------------------------

1. Working with the Java implementation

Create a Java project with your favorite IDE and add all source folders that reside under "src". In order to compile and run the Java code, you need to satisfy the dependencies litsted in B.

The file demo/demojava.java contains an example how to use the model.

2. Working with the cuda core

All files concerning CUDA are stored in the folder "cudacore". All constants like the step number of the impulse responses are adjustable in the file "cudacore/src/Defines.cu". Inside the folder "cudacore" you will find shell scripts to build the cuda core. Use "build.sh" or "buildclang.sh" to compile the cuda code into an executable. Execute the resulting executable to obtain the sizes of the structs we use in cuda. The exact sizes have to be adjusted in the file "democuda.java" which also shows how to use the model with the cuda core. Use the shell scripts "buildptx.sh" or "buildclangptx.sh" to create the actual cuda core that is needed my the model. Whereever you place the core, you have to adjust the Java source code to that path as well.


