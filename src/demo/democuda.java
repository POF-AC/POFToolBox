/*
 	Copyright (C) 2018 Polymer Optical Fiber Application Center (POF-AC), 
 		Technische Hochschule Nürnberg
 		
 	Written by Thomas Becker
 	
	This file is part of POFToolBox.

    POFToolBox is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    POFToolBox is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with POFToolBox.  If not, see <http://www.gnu.org/licenses/>.

    Diese Datei ist Teil von POFToolBox.

    POFToolBox ist Freie Software: Sie können es unter den Bedingungen
    der GNU General Public License, wie von der Free Software Foundation,
    Version 3 der Lizenz oder (nach Ihrer Wahl) jeder späteren
    veröffentlichten Version, weiterverbreiten und/oder modifizieren.

    POFToolBox wird in der Hoffnung, dass es nützlich sein wird, aber
    OHNE JEDE GEWÄHRLEISTUNG, bereitgestellt; sogar ohne die implizite
    Gewährleistung der MARKTFÄHIGKEIT oder EIGNUNG FÜR EINEN BESTIMMTEN ZWECK.
    Siehe die GNU General Public License für weitere Details.

    Sie sollten eine Kopie der GNU General Public License zusammen mit diesem
    Programm erhalten haben. Wenn nicht, siehe <http://www.gnu.org/licenses/>.
*/
package demo;

import static jcuda.driver.JCudaDriver.*;

import java.io.IOException;
import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import model.FiberProps;
import model.GlobalModelSettings;
import model.RefractiveIndexHelper;
import model.RefractiveIndexHelper.Material;
import model.threeDimensional.Angle.ModalDistribution2DArea;
import model.threeDimensional.Angle.LightSourceAngularDistribution.LightSourceAngularEnum;
import model.threeDimensional.Discrete.DiscreteImpulseResponse;
import model.threeDimensional.Discrete.Matrix;
import model.threeDimensional.Discrete.MatrixCell;
import model.threeDimensional.cuda.MyJCudaDriver;
import plot.MyJGnuplot;

/**
 * This class demonstrates the use of the model with CUDA.
 * 
 * The communication with CUDA relies on JCuda. Since we need to allocate memory for CUDA,
 * we need to know the exact sizes of the structs used in CUDA. Since this is compiler dependent,
 * the CUDA code has to be compiled on the target computer. When compiled as an executable,
 * the binary prints the required struct sizes. 
 * 
 * @author Thomas Becker
 *
 */
public class democuda 
{
	/**
	 * This method starts the simulation with CUDA.
	 * 
	 * @param args	Command line arguments
	 * 
	 * @throws IOException	Exception that might arise during the simulation
	 */
	public static void main(String[] args) throws IOException 
	{
		// The path to the scatterfiles
		String strPathToScatterFiles = "/path/to/scatterfiles/";
		
		// The path for the results
		String strPathToResults = "/path/to/results/";
		
		// This flag decides if we use impulse responses with 851 steps <code>true</code> of 85 steps <code>false</code>
		boolean bUseLongIR = false;
		
		// If this flag is <code>true</code> we allocate the memory for CUDA on the host
		// it is highly recommended to allcoate the memory on the graphics card if possible to speed up the model
		boolean bMemoryOnHost = true;
		
		// 12m of fiber takes 75 steps
		int nCycles = 75;
		
		// set the refractive indices of the core
		double dblGroupIndex = RefractiveIndexHelper.getGroupIndexForWavelength(Material.PMMA, 650.0E-9);
		GlobalModelSettings.getInstance().setCoreGroupIndex(dblGroupIndex);
		
		double dblIndex = RefractiveIndexHelper.getGroupIndexForWavelength(Material.PMMA, 650.0E-9);
		GlobalModelSettings.getInstance().setCoreIndex(dblIndex);
			
		// The following variables contain the structsizes needed by CUDA.
		int nSizeMatrix;
		int nSizeNTS;
		int nSizeAH;
		int nSAMPLES_IR;
		
		// this String contains the name of the ptx file containing the CUDA code
		String ptxFileName;
		
		/*
		 * WARNING:
		 * 
		 *  The following struct sizes were obtained with
		 *  
		 *  nvcc: NVIDIA (R) Cuda compiler driver
		 *	Copyright (c) 2005-2016 NVIDIA Corporation
		 *	Built on Tue_Jan_10_13:22:03_CST_2017
		 *	Cuda compilation tools, release 8.0, V8.0.61
		 *
		 *	The sizes are very likely to differ on another platform and the sizes have to be adjusted
		 *	to the real needs!
		 */
		
		// sizes for 851
		if(bUseLongIR)
		{
			nSizeMatrix = 1140217536;
			nSizeNTS = 27240;
			nSizeAH = 54512;
			nSAMPLES_IR = 851;
			ptxFileName = "Matrix851.ptx";
		}
		else
		{
			// sizes for 85
			nSizeMatrix = 629154592;
			nSizeNTS = 2728;
			nSizeAH = 54512;
			nSAMPLES_IR = 85;
			ptxFileName = "Matrix85.ptx";
		}
		
		// // Fiber length is 16 cm according to the scatter data
		double dblFiberLength = 0.16;
		FiberProps fp = new FiberProps(GlobalModelSettings.getInstance().getCoreIndex(), 1.42, 1.0, dblFiberLength, 0.49E-3);
		
		// area light source with all mode categories
		ModalDistribution2DArea mps = new ModalDistribution2DArea(fp, 0.0, (Math.PI/2.0), 1.0, false, LightSourceAngularEnum.Y);
		
		
		// init Matrix
		Matrix m1 = new Matrix(851, 98, 0.0, dblFiberLength, 1.0, GlobalModelSettings.getInstance().getCoreIndex(), 85.0, 0.0);
		
		try
		{
			// load the scatterfiles
			m1.loadScatterFiles(strPathToScatterFiles);
			
			// load the initial power distribution
			m1.loadInitialPowerDistribution(mps, false, 0, false);
				
			// init CUDA 
			JCudaDriver.setExceptionsEnabled(true);
			
		     // Initialize the driver and create a context for the first device.
		    cuInit(0);
		    CUdevice device = new CUdevice();
		    cuDeviceGet(device, 0);
		    CUcontext context = new CUcontext();
		    cuCtxCreate(context, 0, device);
		    
		    // Load the ptx file.
		    CUmodule module = new CUmodule();
		    cuModuleLoad(module, ptxFileName);
	
		    // Obtain function pointers
		    CUfunction fConstructNormTimeSlicer = new CUfunction();
		    cuModuleGetFunction(fConstructNormTimeSlicer, module, "constuctNormTimeSlicer");
		    
		    CUfunction fConstructAngleHelper = new CUfunction();
		    cuModuleGetFunction(fConstructAngleHelper, module, "constructAngleHelper");
		    
		    CUfunction finitMatrix = new CUfunction();
		    cuModuleGetFunction(finitMatrix, module, "initMatrix");
		    
		    CUfunction fLoadScatterData = new CUfunction();
		    cuModuleGetFunction(fLoadScatterData, module, "loadScatterData");
		    
		    CUfunction ftransferIRs = new CUfunction();
		    cuModuleGetFunction(ftransferIRs, module, "transferIRs");
		    
		    CUfunction fprepareMatrixForReloadsA = new CUfunction();
		    cuModuleGetFunction(fprepareMatrixForReloadsA, module, "prepareMatrixForReloadsA");
		    
		    CUfunction fMatrixReloadedA = new CUfunction();
		    cuModuleGetFunction(fMatrixReloadedA, module, "MatrixReloadedA");
		    
		    CUfunction fprepareMatrixForReloadsB = new CUfunction();
		    cuModuleGetFunction(fprepareMatrixForReloadsB, module, "prepareMatrixForReloadsB");
		    
		    CUfunction fMatrixReloadedB = new CUfunction();
		    cuModuleGetFunction(fMatrixReloadedB, module, "MatrixReloadedB");
		    
		    CUfunction fprepareStepC = new CUfunction();
		    cuModuleGetFunction(fprepareStepC, module, "prepareStepC");
		    
		    CUfunction fMatrixReloadedC2 = new CUfunction();
		    cuModuleGetFunction(fMatrixReloadedC2, module, "MatrixReloadedC");
		    
		    CUfunction fGetMatrixData = new CUfunction();
		    cuModuleGetFunction(fGetMatrixData, module, "GetMatrixData");
		    
		    CUfunction fGetDIRData = new CUfunction();
		    cuModuleGetFunction(fGetDIRData, module, "GetDIRData");
		    
		    // Block1 Construct NormTimeSlicer
		    CUdeviceptr dpNTS = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dpNTS, nSizeNTS, bMemoryOnHost);
			
			Pointer kernelParameters = Pointer.to(
		            Pointer.to(dpNTS)
		        );
	
	        // Call the kernel function.
	        int blockSizeX = 1;
	        int gridSizeX = 1;
	        long nStart = System.nanoTime();
	        cuLaunchKernel(fConstructNormTimeSlicer,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        long nEnd = System.nanoTime();
	        System.out.println("Construct NTS: " + (nEnd-nStart)/1E9 + "s");
			// End Block1
	        
	        // Block2 Construct AngleHelper
		    CUdeviceptr dpAH = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dpAH, nSizeAH, bMemoryOnHost);
			
			kernelParameters = Pointer.to(
		            Pointer.to(dpAH)
		        );
	
	        // Call the kernel function.
	        blockSizeX = 1;
	        gridSizeX = 1;
	        nStart = System.nanoTime();
	        cuLaunchKernel(fConstructAngleHelper,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        nEnd = System.nanoTime();
	        System.out.println("Construct AH: " + (nEnd-nStart)/1E9 + "s");
			// End Block2
	        
	        // Block3 initMatrix
		    CUdeviceptr dpM1 = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dpM1, nSizeMatrix, bMemoryOnHost);
			
			CUdeviceptr dDistanceAbsolute = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dDistanceAbsolute, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dDistanceAbsolute, Pointer.to(new double[]{0}), Sizeof.DOUBLE);
			
			CUdeviceptr dDistanceDelta = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dDistanceDelta, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dDistanceDelta, Pointer.to(new double[]{0.16}), Sizeof.DOUBLE);
			
			CUdeviceptr dnOutside = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dnOutside, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dnOutside, Pointer.to(new double[]{1.0}), Sizeof.DOUBLE);
			
			CUdeviceptr dnInside = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dnInside, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dnInside, Pointer.to(new double[]{GlobalModelSettings.getInstance().getCoreIndex()}), Sizeof.DOUBLE);
			
			CUdeviceptr dTZMax = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dTZMax, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dTZMax, Pointer.to(new double[]{m1.getTZMax()}), Sizeof.DOUBLE);
			
			CUdeviceptr dTPMin = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dTPMin, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dTPMin, Pointer.to(new double[]{m1.getTPMin()}), Sizeof.DOUBLE);
			
			CUdeviceptr dTPMax = new CUdeviceptr();
			MyJCudaDriver.cuMemAlloc(dTPMax, Sizeof.DOUBLE, bMemoryOnHost);
			cuMemcpyHtoD(dTPMax, Pointer.to(new double[]{m1.getTPMax()}), Sizeof.DOUBLE);
			
			kernelParameters = Pointer.to(
		            Pointer.to(dpM1),
		            Pointer.to(dDistanceAbsolute),
		            Pointer.to(dDistanceDelta),
		            Pointer.to(dnOutside),
		            Pointer.to(dnInside),
		            Pointer.to(dTZMax),
		            Pointer.to(dTPMin),
		            Pointer.to(dTPMax)
		        );
	
	        // Call the kernel function.
	        blockSizeX = 32;
	        int blockSizeY = 1;
	        
	        gridSizeX = 32;
	        int gridSizeY = 98;
	        nStart = System.nanoTime();
	        cuLaunchKernel(finitMatrix,
	            gridSizeX,  gridSizeY, 1,      // Grid dimension
	            blockSizeX, 1, blockSizeY,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        nEnd = System.nanoTime();
	        System.out.println("init Matrix: " + (nEnd-nStart)/1E9 + "s");
			// End Block3
	        
	        
	        // Block4 loadScatterData
	        CUdeviceptr dinTZ = new CUdeviceptr();
	 	   	MyJCudaDriver.cuMemAlloc(dinTZ, 1 * Sizeof.INT, bMemoryOnHost);
	 	   	   
	 	   
	 	   	CUdeviceptr dinTP = new CUdeviceptr();
	 	   	MyJCudaDriver.cuMemAlloc(dinTP, 1 * Sizeof.INT, bMemoryOnHost);
	 	   
	 	   	CUdeviceptr diSD = new CUdeviceptr();
	 	   	MyJCudaDriver.cuMemAlloc(diSD, 851 * Sizeof.DOUBLE, bMemoryOnHost);
	 	   	
	 	    CUdeviceptr dTZInside = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dTZInside, Sizeof.DOUBLE, bMemoryOnHost); 
	
	        // Call the kernel function.
	        blockSizeX = 851;
	        gridSizeX = 1;
	        nStart = System.nanoTime();
	        for(int i = 0; i < 851; i++)
	 	   	{
	        	for(int j = 0; j < 98; j++)
	 		   	{
	        		double[] dblData = m1.getCells()[i][j].getScatterCell().getData();
	 			   	cuMemcpyHtoD(diSD, Pointer.to(dblData),851 * Sizeof.DOUBLE);
	 			   	cuMemcpyHtoD(dinTZ, Pointer.to(new int[]{i}), Sizeof.INT);
	 			   	cuMemcpyHtoD(dinTP, Pointer.to(new int[]{j}), Sizeof.INT);
	 			   	cuMemcpyHtoD(dTZInside, Pointer.to(new double[]{m1.getCells()[i][j].getScatterCell().getThetaZInside()}), Sizeof.DOUBLE);
	 		     
	 			   	kernelParameters = Pointer.to(
	 		
	 			            Pointer.to(dinTZ),
	 			            Pointer.to(dinTP),
	 			            Pointer.to(diSD),
	 			            Pointer.to(dpM1),
	 			            Pointer.to(dTZInside)
	 			        );
	 		
	 			        // Call the kernel function.
	
	 			        cuLaunchKernel(fLoadScatterData,
	 			            gridSizeX,  1, 1,      // Grid dimension
	 			            blockSizeX, 1, 1,      // Block dimension
	 			            0, null,               // Shared memory size and stream
	 			            kernelParameters, null // Kernel- and extra parameters
	 			        );
	 			        cuCtxSynchronize();
	 		   	}
	 	   	}
	        nEnd = System.nanoTime();
	        System.out.println("Load Scatterdata: " + (nEnd-nStart)/1E9 + "s");
			// End Block4
	        
	        // Block 5 transfer IR Values
	        CUdeviceptr dZeroValue = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dZeroValue, 851*Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	double[] dblZeroValues = new double[851];
		   	nStart = System.nanoTime();
		   	for(int j = 0; j < 98; j++)
			{
		   		for(int i = 0; i < 851; i++)
			   	{	
			   		dblZeroValues[i] = m1.getCells()[i][j].getDIR().getPeaks().get(0)[1];
			   	}
		   		
		   		cuMemcpyHtoD(dinTP, Pointer.to(new int[]{j}), Sizeof.INT);
		   		cuMemcpyHtoD(dZeroValue, Pointer.to(dblZeroValues), 851*Sizeof.DOUBLE);
		   		
				kernelParameters = Pointer.to(
			            Pointer.to(dinTP),
			            Pointer.to(dZeroValue),
			            Pointer.to(dpM1)
			        );
		
		        // Call the kernel function.
		        blockSizeX = 851;
		        gridSizeX = 1;
		        
		        cuLaunchKernel(ftransferIRs,
		            gridSizeX,  1, 1,      // Grid dimension
		            blockSizeX, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cuCtxSynchronize();
			}
	        nEnd = System.nanoTime();
	        System.out.println("Transfer IRs: " + (nEnd-nStart)/1E9 + "s");
			// End Block5
	        
	        // Block6 MatrixReloadedA
	        CUdeviceptr dpM2 = new CUdeviceptr();
			int nR = MyJCudaDriver.cuMemAlloc(dpM2, nSizeMatrix, bMemoryOnHost);
			
			kernelParameters = Pointer.to(
		            Pointer.to(dpM1),
		            Pointer.to(dpM2),
		            Pointer.to(dpNTS),
		            Pointer.to(dpAH)
		        );
	
	        // Call the kernel function.
			blockSizeX = 8;
		    blockSizeY = 8;
		        
		    gridSizeX = 128;
		    gridSizeY = 16;
	        
	        nStart = System.nanoTime();
	        cuLaunchKernel(fMatrixReloadedA,
	            gridSizeX,  gridSizeY, 1,      // Grid dimension
	            blockSizeX, blockSizeY, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        nEnd = System.nanoTime();
	        System.out.println("Matrix ReloadedA: " + (nEnd-nStart)/1E9 + "s");
			// End MatrixReloadeA
	        
	        // Block7 MatrixReloadedB
	        kernelParameters = Pointer.to(
		            Pointer.to(dpM1),
		            Pointer.to(dpM2),
		            Pointer.to(dpNTS),
		            Pointer.to(dpAH)
		        );
	
	        // Call the kernel function.
	        blockSizeX = 8;
		    blockSizeY = 8;
		        
		    gridSizeX = 128;
		    gridSizeY = 16;
	        nStart = System.nanoTime();
	        cuLaunchKernel(fMatrixReloadedB,
	            gridSizeX,  gridSizeY, 1,      // Grid dimension
	            blockSizeX, blockSizeY, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        nEnd = System.nanoTime();
	        System.out.println("Matrix ReloadedB: " + (nEnd-nStart)/1E9 + "s");
			// End MatrixReloadedB
	        
	        CUdeviceptr dpStepTimes = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dpStepTimes, nSAMPLES_IR*Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr dpStepWidth = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dpStepWidth, 851*Sizeof.DOUBLE, bMemoryOnHost);
	        
		   	// Block8 MatrixReloadedC
	        
	        // Call the kernel function.
	        blockSizeX = 1;
		    blockSizeY = 32;
		        
		    gridSizeX = 1;
		    gridSizeY = 32;
		    
		    CUdeviceptr nTZ = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(nTZ, Sizeof.INT, bMemoryOnHost);
		   	
		   	CUdeviceptr nTP = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(nTP, Sizeof.INT, bMemoryOnHost);
		   	
		   	kernelParameters = Pointer.to(
		            Pointer.to(dpM1),
		            Pointer.to(dpM2),
		            Pointer.to(dpNTS),
		            Pointer.to(dpAH),
		            Pointer.to(nTZ),
		            Pointer.to(nTP),
		            Pointer.to(dpStepTimes),
		            Pointer.to(dpStepWidth)
		        );
	       
		    for(int i = 0; i < 851; i++)
		    {
		    	for(int j = 0; j < 98; j++)
		
		    	{
		    		int nTZn[] = new int[]{i};
		            cuMemcpyHtoD(nTZ,Pointer.to(nTZn), Sizeof.INT);
		            
		            int nTPn[] = new int[]{j};
		            cuMemcpyHtoD(nTP,Pointer.to(nTPn), Sizeof.INT);
		            
		    		kernelParameters = Pointer.to(
				            Pointer.to(dpM1),
				            Pointer.to(dpM2),
				            Pointer.to(dpNTS),
				            Pointer.to(dpAH),
				            Pointer.to(nTZ),
				            Pointer.to(nTP),
				            
				            Pointer.to(dpStepTimes),
				            Pointer.to(dpStepWidth));
				            
			        cuLaunchKernel(fMatrixReloadedC2,
			        		gridSizeX,  gridSizeY, 1,      // Grid dimension
			                blockSizeX, blockSizeY, 1,      // Block dimension
			            0, null,               // Shared memory size and stream
			            kernelParameters, null // Kernel- and extra parameters
			        );
			        cuCtxSynchronize();
			       
			    }
		    }
		    nEnd = System.nanoTime();
	        System.out.println("Matrix ReloadedC: " + (nEnd-nStart)/1E9 + "s");
			// End MatrixReloadedC
	        
	        // Additional runs
	        for(int nRuns = 0; nRuns < nCycles-1; nRuns++)
	        {
		        // Reload Matrix!!!        
		        CUdeviceptr ftemp = dpM1;
		        dpM1 = dpM2;
		        dpM2 = ftemp;
		        
		    	CUdeviceptr dpMinTime = new CUdeviceptr();
			   	MyJCudaDriver.cuMemAlloc(dpMinTime, Sizeof.DOUBLE, bMemoryOnHost);
			   	
			   	CUdeviceptr dpMaxTime = new CUdeviceptr();
			   	MyJCudaDriver.cuMemAlloc(dpMaxTime, Sizeof.DOUBLE, bMemoryOnHost);

			   	// Block sw1 PrepareMatrixForReload
		        kernelParameters = Pointer.to(
			            Pointer.to(dpM2),
			            Pointer.to(dpMinTime),
			            Pointer.to(dpMaxTime)
			        );
		        
		        // Call the kernel function.
		        blockSizeX = 1;
			    blockSizeY = 1;
			        
			    gridSizeX = 1;
			    gridSizeY = 1;
		        
		        cuLaunchKernel(fprepareMatrixForReloadsA,
		            gridSizeX,  gridSizeY, 1,      // Grid dimension
		            blockSizeX, blockSizeY, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cuCtxSynchronize();
		
		        // Call the kernel function.
		        blockSizeX = 8;
			    blockSizeY = 16;
			        
			    gridSizeX = 128;
			    gridSizeY = 8;
		        nStart = System.nanoTime();
		        cuLaunchKernel(fprepareMatrixForReloadsB,
		            gridSizeX,  gridSizeY, 1,      // Grid dimension
		            blockSizeX, blockSizeY, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cuCtxSynchronize();
		        System.out.println("Matrix Reloaded again half"+ (nEnd-nStart)/1E9 + "s");
		        
		        nStart = System.nanoTime();
		        
		        // Block 8 a
		        nStart = System.nanoTime();
			   	
		        kernelParameters = Pointer.to(
			            Pointer.to(dpM1),
			            Pointer.to(dpNTS),
			            Pointer.to(dpStepTimes),
			            Pointer.to(dpStepWidth)
			        );
	
		        // Call the kernel function.
		        blockSizeX = 8;
			    blockSizeY = 1;
			        
			    gridSizeX =  128;
			    gridSizeY = 1;
		        nStart = System.nanoTime();
		        cuLaunchKernel(fprepareStepC,
		        		gridSizeX,  gridSizeY, 1,      // Grid dimension
		                blockSizeX, blockSizeY, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cuCtxSynchronize();
		        nEnd = System.nanoTime();
		        System.out.println("Matrix prepare C: " + (nEnd-nStart)/1E9 + "s");
		        // End Block 8 a

				// End sw1
		        // Block8 MatrixReloadedC
		        
			    nStart = System.nanoTime();   
		
		        // Call the kernel function.
		        blockSizeX = 8;
			    blockSizeY = 8;
			        
			    gridSizeX = 107;
			    gridSizeY = 107;
		       
			    for(int i = 0; i < 851; i++)
			    {
			    	for(int j = 0; j < 98; j++)
			    	{
			    		int nTZn[] = new int[]{i};
			            cuMemcpyHtoD(nTZ,Pointer.to(nTZn), Sizeof.INT);
			            
			            int nTPn[] = new int[]{j};
			            cuMemcpyHtoD(nTP,Pointer.to(nTPn), Sizeof.INT);
			            
			    		kernelParameters = Pointer.to(
					            Pointer.to(dpM1),
					            Pointer.to(dpM2),
					            Pointer.to(dpNTS),
					            Pointer.to(dpAH),
					            Pointer.to(nTZ),
					            Pointer.to(nTP),
					            
					            Pointer.to(dpStepTimes),
					            Pointer.to(dpStepWidth));
					            
				        cuLaunchKernel(fMatrixReloadedC2,
				        		gridSizeX,  gridSizeY, 1,      // Grid dimension
				                blockSizeX, blockSizeY, 1,      // Block dimension
				            0, null,               // Shared memory size and stream
				            kernelParameters, null // Kernel- and extra parameters
				        );
				    }
			    }
			    
			    cuCtxSynchronize();
	        
		        nEnd = System.nanoTime();
		        System.out.println("Matrix Reloaded again: " + (nEnd-nStart)/1E9 + "s");
				// End MatrixReloadedC
	        }
	        
	                
	        // Block9 GetBackMatrixData
	        CUdeviceptr dnTPMaxIndex = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dnTPMaxIndex, Sizeof.INT, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblTPMax = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblTPMax, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblTPMin = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblTPMin, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr dnTZMaxIndex = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dnTZMaxIndex, Sizeof.INT, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblTZMax = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblTZMax, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblDistanceAbs = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblDistanceAbs, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblDistanceDelta = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblDistanceDelta, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblCoreIndex = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblCoreIndex, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblSurroundIndex = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblSurroundIndex, Sizeof.DOUBLE, bMemoryOnHost);

	        kernelParameters = Pointer.to(
	        		Pointer.to(dpM2),
		            Pointer.to(dnTPMaxIndex),
		            Pointer.to(ddblTPMax),
		            Pointer.to(ddblTPMin),
		            Pointer.to(dnTZMaxIndex),
		            Pointer.to(ddblTZMax),
		            Pointer.to(ddblDistanceAbs),
		            Pointer.to(ddblDistanceDelta),
		            Pointer.to(ddblCoreIndex),
		            Pointer.to(ddblSurroundIndex)
		        );
	
	        // Call the kernel function.
	        blockSizeX = 1;
	        gridSizeX = 1;
	        
	        blockSizeY = 1;
	        gridSizeY = 1;
	        nStart = System.nanoTime();
	        cuLaunchKernel(fGetMatrixData,
	        		gridSizeX,  gridSizeY, 1,      // Grid dimension
	                blockSizeX, blockSizeY, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        nEnd = System.nanoTime();
	        
	        int nTPMaxIndex[] = new int[1];
	        cuMemcpyDtoH(Pointer.to(nTPMaxIndex), dnTPMaxIndex, Sizeof.INT);
	        
	        double dblTPMax[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblTPMax), ddblTPMax, Sizeof.DOUBLE);
	        
	        double dblTPMin[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblTPMin), ddblTPMin, Sizeof.DOUBLE);
	        
	        int nTZMaxIndex[] = new int[1];
	        cuMemcpyDtoH(Pointer.to(nTZMaxIndex), dnTZMaxIndex, Sizeof.INT);
	        
	        double dblTZMax[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblTZMax), ddblTZMax, Sizeof.DOUBLE);
	        
	        double dblDistanceAbs[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblDistanceAbs), ddblDistanceAbs, Sizeof.DOUBLE);
	        
	        double dblDistanceDelta[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblDistanceDelta), ddblDistanceDelta, Sizeof.DOUBLE);
	        
	        double dblCoreIndex[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblCoreIndex), ddblCoreIndex, Sizeof.DOUBLE);
	        
	        double dblSurroundIndex[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblSurroundIndex), ddblSurroundIndex, Sizeof.DOUBLE);
	        
	        System.out.println("GetMatrixData: " + (nEnd-nStart)/1E9 + "s");
	        
	        Matrix mBackFromOuterSpace = new Matrix(
	        		nTPMaxIndex[0], 
	        		dblTPMax[0],
	        		dblTPMin[0],
	        		nTZMaxIndex[0],
	        		dblTZMax[0],
	        		dblDistanceAbs[0],
	        		dblDistanceDelta[0],
	        		dblCoreIndex[0],
	        		dblSurroundIndex[0],
	        		0.0);
			// End GetMatrixData
	        
	        // Block10 GetDIRs
	        CUdeviceptr dnTZIndex = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dnTZIndex, Sizeof.INT, bMemoryOnHost);
		   	
		   	CUdeviceptr dnTPIndex = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dnTPIndex, Sizeof.INT, bMemoryOnHost);
		   	
		   	CUdeviceptr dnSteps = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(dnSteps, Sizeof.INT, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblMinTime = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblMinTime, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblMaxTime = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblMaxTime, Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	CUdeviceptr ddblPower = new CUdeviceptr();
		   	MyJCudaDriver.cuMemAlloc(ddblPower, 98*851*nSAMPLES_IR*Sizeof.DOUBLE, bMemoryOnHost);
		   	
		   	nStart = System.nanoTime();
		  
   			cuMemcpyHtoD(dnTZIndex, Pointer.to(new int[]{0}), Sizeof.INT);
   			cuMemcpyHtoD(dnTPIndex, Pointer.to(new int[]{0}), Sizeof.INT);
   			
	        kernelParameters = Pointer.to(
	        		Pointer.to(dpM2),
		            Pointer.to(dnTZIndex),
		            Pointer.to(dnTPIndex),
		            Pointer.to(dnSteps),
		            Pointer.to(ddblMinTime),
		            Pointer.to(ddblMaxTime),
		            Pointer.to(ddblPower)
		        );
	        
	        // Call the kernel function.
	        blockSizeX = 32;
	        blockSizeY = 16;
	        
	        gridSizeX = 32;
	        gridSizeY = 8;
	        
	        cuLaunchKernel(fGetDIRData,
	        		gridSizeX,  gridSizeY, 1,      // Grid dimension
	                blockSizeX, blockSizeY, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();
	        
	        int nSteps[] = new int[1];
	        cuMemcpyDtoH(Pointer.to(nSteps), dnSteps, Sizeof.INT);
	        
	        double dblMinTime[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblMinTime), ddblMinTime, Sizeof.DOUBLE);
	        
	        double dblMaxTime[] = new double[1];
	        cuMemcpyDtoH(Pointer.to(dblMaxTime), ddblMaxTime, Sizeof.DOUBLE);
	        
	        double dblPower[] = new double[98*851*nSAMPLES_IR];
	        cuMemcpyDtoH(Pointer.to(dblPower), ddblPower, Sizeof.DOUBLE*98*851*nSAMPLES_IR);
	        
	        int imax = 851;
	        int jmax = 98;
	        
	        for(int i = 0; i < imax; i++)
	        {
	        	for(int j = 0; j < jmax; j++)
	        	{
	        		int nIndex1 = i*nSAMPLES_IR*98 + j*nSAMPLES_IR;
	        		int nIndex2 = nIndex1 + nSAMPLES_IR-1;
	        		double dblCurrentArray[] = Arrays.copyOfRange(dblPower, nIndex1, nIndex2);
	        		mBackFromOuterSpace.getCells()[i][j] = new MatrixCell();
	 		        mBackFromOuterSpace.getCells()[i][j].setDIR(new DiscreteImpulseResponse(nSteps[0], dblMinTime[0], dblMaxTime[0], dblCurrentArray, false, 0.0));
	        	}
	        }
        
	        nEnd = System.nanoTime();
	        
	        
	        
	        System.out.println("GetDIRData: " + (nEnd-nStart)/1E9 + "s");
	        // End GetDIRs
	        
	        
	        DiscreteImpulseResponse dir = mBackFromOuterSpace.createImpulseResponse(nSAMPLES_IR, true, true, false);
	        dir.createDiscreteFile(strPathToResults,"iacuda"+ bUseLongIR + "_" + nCycles +".txt");
	        dir.createSpikedFile(strPathToResults,"iacudaSpiked.txt");
	        
	        // release allocated memory
			MyJCudaDriver.cuMemFree(dpNTS, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dpAH, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dpM1, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dpM2, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dDistanceAbsolute, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dDistanceDelta, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dnOutside, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dnInside, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dTZMax, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dTPMin, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dTPMax, bMemoryOnHost);
	        
	        
			MyJCudaDriver.cuMemFree(dinTZ, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dinTP, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(diSD, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dTZInside, bMemoryOnHost);
	        
			MyJCudaDriver.cuMemFree(dZeroValue, bMemoryOnHost);
	        
	        
			MyJCudaDriver.cuMemFree(dnTPMaxIndex, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblTPMax, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblTPMin, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dnTZMaxIndex, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblTZMax, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblDistanceAbs, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblDistanceDelta, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblCoreIndex, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblSurroundIndex, bMemoryOnHost);
	        
			MyJCudaDriver.cuMemFree(dnTZIndex, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dnTPIndex, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(dnSteps, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblMinTime, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblMaxTime, bMemoryOnHost);
			MyJCudaDriver.cuMemFree(ddblPower, bMemoryOnHost);
	        
	        
	        System.exit(0);
		}
		catch(Exception exc)
		{
			exc.printStackTrace();
		}
	}
}


