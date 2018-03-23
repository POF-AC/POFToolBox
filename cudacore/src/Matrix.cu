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
#ifndef MATRIX
#define MATRIX
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DISCRETEIMPULSERESPONSE
#include "DiscreteImpulseResponse.cu"
#endif

#ifndef TRANSFERANGLE
#define "TransferAngle.cu"
#endif

#ifndef ANGLEHELPER
#include "AngleHelper.cu"
#endif

#ifndef DEFINES
#include "Defines.cu"
#endif

/**
 * This struct defines a cell of the matrix and consists basically of a scatter cell and an impulse response.
 *  
 * @author Thomas Becker
 *
 */
struct MatrixCell
{
	/**
	 * The impulse response
	 */
	strDiscreteImpulseResponse dir;
	
	/**
	 * The data array holding the scattering distribution
	 */
	double m_dblData[STEPWIDTH_SC]; 
	
	/**
	 * Theta z inside the fiber for which this power distribution is valid
	 */
	double m_dblThetaZInside;
	
	/**
	 * c'tor. This c'tor creates the impulse response and the scatter cell.
	 *  
	 * @param a_bPeak	<code>true</code> if the impulse response is in peak mode
	 */
	__device__ void construct(bool a_bPeak)
	{
		dir.construct(a_bPeak);
		for(int i = 0; i < STEPWIDTH_SC; i++)
		{
			m_dblData[i] = 0.0;
		}
		m_dblThetaZInside = 0.0;
	}
};

/**
 * This struct hold a two-dimensional matrix of MatrixCells and represents the state of the 
 * power distribution over angles and time at a specific point in the fiber.
 * 
 * @author Thomas Becker
 *
 */
struct Matrix
{
	/**
	 * The main Matrix. First index is Theta_z, second Theta_phi.
	 */
	MatrixCell m_matrix[STEPWIDTH_TZ][STEPWIDTH_TP];
	
	/**
	 * The maximum index of the matrix in Theta_phi direction.
	 */
	int m_nTPMaxIndex;
	
	/**
	 * The maximum value for Theta_phi.
	 */
	double m_dblTPMax;
	
	/**
	 * The minimum value for Theta_phi.
	 */
	double m_dblTPMin;
	
	/**
	 * The maximum index of the matrix in Theta_z direction.
	 */
	int m_nTZMaxIndex;
	
	/**
	 * The maximum value for Theta_phi.
	 */
	double m_dblTZMax;
	
	/**
	 * The absolute positon of the matrix in the fiber. This distance includes strained parts.
	 */
	double m_dblDistanceAbsolute;
	
	/**
	 * The relative distance between two matrices. This distance excludes strain.
	 */
	double m_dblDistanceDelta;
	
	/**
	 * The refractive index of the core.
	 */
	double m_dblCoreIndex;
	
	/**
	 * Helper for transferring angles between inside and outside of the fiber.
	 */
	TransferAngle m_ta;
	
	/**
	 * Constructor. This method constructs the matrix cells and their impulse responses.
	 * 
	 * @param a_pDistanceAbsolute	Absolute position of the Matrix in the fiber
	 * @param a_ppDistanceDelta		Distance between two matrices
	 * @param a_pdblnOutside			Refractive index of the surrounding material
	 * @param a_pdblnInside			Refractive index of the core
	 * @param a_pTZMax				Max angle for Theta z
	 * @param a_pdblTPMin			Min angle for Theta Phi
	 * @param a_pdblTPMax			Max angle for Theta Phi
	 * @param a_bPeak				If <code>true</code>, the impulse response is constructed in peak mode
	 */	
	__device__ void construct(double a_dblDistanceAbsolute, double* a_pdblDistanceDelta, double* a_pdblnOutside, double* a_pdblnInside, 
	double* a_pdblTZMax, double* a_pdblTPMin, double* a_pdblTPMax, bool a_bPeak)
	{
		int nIndexx = (blockIdx.x * blockDim.x) + threadIdx.x;
		int nIndexy = (blockIdx.y * blockDim.y) + threadIdx.y;
		
		if(nIndexx < STEPWIDTH_TZ && nIndexy < STEPWIDTH_TP)
		{
			m_matrix[nIndexx][nIndexy].construct(a_bPeak);
		}
		
		if(nIndexx == 0 && nIndexy == 0)
		{
			m_nTPMaxIndex = STEPWIDTH_TP-1;
			
			m_dblTPMax = *a_pdblTPMax;
			m_dblTPMin = *a_pdblTPMin;
			
			m_nTZMaxIndex = STEPWIDTH_TZ-1;
			m_dblTZMax = *a_pdblTZMax;
			
			m_dblDistanceAbsolute = a_dblDistanceAbsolute;
			m_dblDistanceDelta = *a_pdblDistanceDelta;
			
			m_dblCoreIndex = *a_pdblnInside;
			
			m_ta.construct(*a_pdblnOutside, *a_pdblnInside);
		}
	}
};

/**
 * This method constructs a new matrix from the current one.
 * 
 * @param a_pM					The new matrix
 * @param a_pDistanceAbsolute	Absolute position of the Matrix in the fiber
 * @param a_pDistanceDelta		Distance between two matrices
 * @param a_pdblnOutside			Refractive index of the surrounding material
 * @param a_pdblnInside			Refractive index of the core
 * @param a_pTZMax				Max angle for Theta z
 * @param a_pdblTPMin			Min angle for Theta Phi
 * @param a_pdblTPMax			Max angle for Theta Phi
 */
extern "C"
__global__ void initMatrix(Matrix* a_pM, double* a_dblDistanceAbsolute, double* a_dblDistanceDelta, double* a_pdblnOutside, double* a_pdblnInside, 
	double* a_dblTZMax, double* a_dblTPMin, double* a_dblTPMax)
{
	a_pM->construct(*a_dblDistanceAbsolute, a_dblDistanceDelta, a_pdblnOutside, a_pdblnInside, 
	a_dblTZMax, a_dblTPMin, a_dblTPMax, true);
}

/**
 * This function loads the scatter data into the matrix cells.
 * 
 * @param a_pnTZ				Index Theta z 
 * @param a_pnTP				Index Theta phi
 * @param a_dblScatterData	Scatter data
 * @param a_pM 					Matrix
 * @param a_dblThetaZInside	Theta z inside the fiber
 */
extern "C"
__global__ void loadScatterData(int* a_pnTZ, int* a_pnTP, double* a_dblScatterData, Matrix* a_pM, double* a_pdblThetaZInside)
{
	a_pM->m_matrix[*a_pnTZ][*a_pnTP].m_dblData[threadIdx.x] = a_dblScatterData[threadIdx.x];
	a_pM->m_matrix[*a_pnTZ][*a_pnTP].m_dblThetaZInside = *a_pdblThetaZInside;	
}

/**
 * This function copies the launching condition for a specific theta phi into the model.
 * 
 * @param a_pnTP				Index Theta phi
 * @param a_pdblIRZeroValues		Double array that contains the power distribution for a specific theta phi over theta z
 * @param a_pM					The matrix that we initialize
 */
extern "C"
__global__ void transferIRs(int* a_pnTP, double* a_pdblIRZeroValues, Matrix* a_pM)
{
	a_pM->m_matrix[threadIdx.x][*a_pnTP].dir.m_dblPowerPeak[0] = 0.0;
	a_pM->m_matrix[threadIdx.x][*a_pnTP].dir.m_dblPowerPeak[1] = a_pdblIRZeroValues[threadIdx.x];
	a_pM->m_matrix[threadIdx.x][*a_pnTP].dir.m_dblMinTime = 0.0;
	a_pM->m_matrix[threadIdx.x][*a_pnTP].dir.m_dblMaxTime = 0.0;

}

/**
 * This method performs the matrix transition for a specific time and one angle of the scatter distribution during the propagation process.
 * 
 * @param a_pMOld				The old matrix
 * @param a_pMNew				The new matrix
 * @param a_nIndexTZ			Theta z index
 * @param a_nIndexTP			Theta phi index
 * @param a_pNTS				Norm time slicer
 * @param a_pAH					Angle helper
 * @param a_dblTime				The time of the considered power in the old matrix
 * @param a_dblPower			The power that hast to be spread
 * @param a_bPrint				Prints debugging information if <code>true</code>
 * 
 */
__device__ void scatterToNextMatrix(Matrix* a_pMOld, Matrix* a_pMNew, int a_nIndexTZ, int a_nIndexTP, NormTimeSlicer* a_pNTS, AngleHelper* a_pAH, double a_dblTime, double a_dblPower, bool a_bPrint)
{
	// determine the index of the scatter distribution that we handle
	int i = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// do only what is possible
	if(i < 851)
	{
		// calculate the minimum time difference between two matrices
		double mtp = (a_pMNew->m_dblDistanceDelta*REFRACTIVE__GROUP_INDEX_CORE)/m_dblSpeedOfLight;
		
		// To be done for a scattercell
		{	
			double* data = &(a_pMOld->m_matrix[a_nIndexTZ][a_nIndexTP].m_dblData[i]);
		
			double dblFactor = *data*a_pAH->getStepWidthForStep(i);

			if(0.0 != dblFactor)
			{
				double &dblTZ = a_pMOld->m_matrix[a_nIndexTZ][a_nIndexTP].m_dblThetaZInside;
				
				double dblTZ2;
				
				dblTZ2 = a_pAH->getCenterAngleForIndex(i);
				
				double dblMeanAngle = (dblTZ+dblTZ2)/2.0; 
				
				// we calculate the transit-time with the mean angle
				double dblNewTime = a_dblTime +  mtp/cos(dblMeanAngle*M_PI/180.0);
				
				double dblNewPower = dblFactor*a_dblPower;
				a_pMNew->m_matrix[i][a_nIndexTP].dir.addPowerPeak(dblNewTime, dblNewPower, a_pNTS, i, a_nIndexTP, a_bPrint);
			}			
		}
	}
}

/**
 * This method performs the matrix transition for one angle of the scatter distribution during the propagation process.
 * 
 * @param a_pMOld				The old matrix
 * @param a_pMNew				The new matrix
 * @param a_nIndexTZ			Theta z index
 * @param a_nIndexTP			Theta phi index
 * @param a_pNTS				Norm time slicer
 * @param a_pAH					Angle helper
 * @param a_dblStepWidth		This array carries the stepwidths for the impulse response
 * @param a_dblTimes			This array carries the times of the impulse response
 * 
 */
__device__ void spreadPower(Matrix* a_pMOld, Matrix* a_pMNew, int a_nIndexTZ, int a_nIndexTP, NormTimeSlicer* a_pNTS, AngleHelper* a_pAH, double* a_dblStepWidth, double* a_dblTimes)
{
	int nIndexX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int nIndexY = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if(nIndexX < STEPWIDTH_IR && nIndexY < STEPWIDTH_TZ)
	{
		if(a_pMOld->m_matrix[a_nIndexTZ][a_nIndexTP].dir.m_bPeak)
		{
			double &dblTime = a_pMOld->m_matrix[a_nIndexTZ][a_nIndexTP].dir.m_dblPowerPeak[0];
			double &dblPower = a_pMOld->m_matrix[a_nIndexTZ][a_nIndexTP].dir.m_dblPowerPeak[1];
			
			scatterToNextMatrix(a_pMOld, a_pMNew, a_nIndexTZ, a_nIndexTP, a_pNTS, a_pAH, dblTime, dblPower, false);
		}
		else
		{
			double dblPower;
			double *dblPowerRaw = &(a_pMOld->m_matrix[a_nIndexTZ][a_nIndexTP].dir.m_dblPower[0]);
			{
				dblPower = *(dblPowerRaw+nIndexX)*a_dblStepWidth[nIndexX];
	
				if(0 != dblPower)
				{
					scatterToNextMatrix(a_pMOld, a_pMNew, a_nIndexTZ, a_nIndexTP, a_pNTS, a_pAH, a_dblTimes[nIndexX], dblPower, true);
				}
			}
		}
	}
}

/**
 * This function is called from Java and constructs an object of the type NormTimeSlicer in the referenced memory.
 * 
 * @param a_pNTS Pointer to the space for the NormTimeSlicer
 * 
 */
extern "C"
__global__ void constuctNormTimeSlicer(NormTimeSlicer* a_pNTS)
{
	a_pNTS->construct();
	printf("CUDA: NTS constructed\n");
}

/**
 * This function is called from Java and is Step A during a matrix transition.
 * In this function, we prepare the next matrix.
 * 
 * @param a_pMOld	The old matrix
 * @param a_pMNew	The new matrix
 * @param a_pNTS	NormTimeSlicer
 * @param a_pAH		AngleHelper
 * 
 */
extern "C"
__global__ void MatrixReloadedA(Matrix* a_pMOld, Matrix* a_pMNew, NormTimeSlicer* a_pNTS, AngleHelper* a_pAH)
{
	double dblNewDistance;
	dblNewDistance = a_pMOld->m_dblDistanceAbsolute+a_pMOld->m_dblDistanceDelta;
	
		a_pMNew->construct(dblNewDistance, &(a_pMOld->m_dblDistanceDelta),
			&(a_pMOld->m_ta.m_dblIndex1), &(a_pMOld->m_ta.m_dblIndex2), &(a_pMOld->m_dblTZMax),  &(a_pMOld->m_dblTPMin), &(a_pMOld->m_dblTPMax), false);
}

/**
 * This function is called from Java and is Step B during a matrix transition.
 * In this function, we copy the scattering distributions.
 * 
 * @param a_pMOld	The old matrix
 * @param a_pMNew	The new matrix
 * @param a_pNTS	NormTimeSlicer
 * @param a_pAH		AngleHelper
 * 
 */
extern "C"
__global__ void MatrixReloadedB(Matrix* a_pMOld, Matrix* a_pMNew, NormTimeSlicer* a_pNTS, AngleHelper* a_pAH)
{
	int nIndexX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int nIndexY = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	double dblMaxTime;
	double dblMinTime;

	dblMinTime = a_pMNew->m_dblDistanceAbsolute*a_pMNew->m_ta.m_dblIndex2/ m_dblSpeedOfLight;
	dblMaxTime = dblMinTime/cos(a_pMNew->m_dblTZMax*M_PI/180.0);
	if(dblMaxTime == 0.0)
	{
		dblMaxTime = 1.0E-12;
	}
	
	if(nIndexX < STEPWIDTH_TZ && nIndexY < STEPWIDTH_TP)
	{
		for(int j = 0; j <= a_pMOld->m_nTZMaxIndex; j++)
		{
			a_pMNew->m_matrix[nIndexX][nIndexY].m_dblData[j] = a_pMOld->m_matrix[nIndexX][nIndexY].m_dblData[j];
		}

		a_pMNew->m_matrix[nIndexX][nIndexY].m_dblThetaZInside = a_pMOld->m_matrix[nIndexX][nIndexY].m_dblThetaZInside;
		a_pMNew->m_matrix[nIndexX][nIndexY].dir.m_dblMinTime = dblMinTime;
		a_pMNew->m_matrix[nIndexX][nIndexY].dir.m_dblMaxTime = dblMaxTime;		
	}
}


/**
 * This function is called from Java and prepares Step A during a matrix transition.
 * In this function, we set the absolute distance of the new matrix and obtain the minimum and maximum time of the next impulse responses.
 * 
 * @param a_pMNew	The new matrix
 * @param a_pdblMinTime	Minimum time of the impulse response
 * @param a_pdblMaxTime	Maximum time of the impulse response
 * 
 */
extern "C"
__global__ void prepareMatrixForReloadsA(Matrix* spMNew, double* a_pdblMinTime, double* a_pdblMaxTime)
{
	spMNew->m_dblDistanceAbsolute+=2*spMNew->m_dblDistanceDelta;
	
	*a_pdblMinTime = spMNew->m_dblDistanceAbsolute*spMNew->m_ta.m_dblIndex2/ m_dblSpeedOfLight;
	*a_pdblMaxTime = *a_pdblMinTime/cos(spMNew->m_dblTZMax*M_PI/180.0);
	if(*a_pdblMaxTime == 0.0)
	{
		*a_pdblMaxTime = 1.0E-12;
	}
}

/**
 * This function is called from Java and prepares Step B during a matrix transition.
 * In this function, we prepare the minimum and maximum times of the impulse responses of the next matrix
 * 
 * @param a_pMNew	The new matrix
 * @param a_dblMinTime	Minimum time of the impulse response
 * @param a_dblMaxTime	Maximum time of the impulse response
 * 
 */	
extern "C"
__global__ void prepareMatrixForReloadsB(Matrix* spMNew, double* a_dblMinTime, double* a_dblMaxTime)
{
	int nIndexX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int nIndexY = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if(nIndexX < STEPWIDTH_TZ && nIndexY < STEPWIDTH_TP)
	{
		// prepare irs for hyperspace
		spMNew->m_matrix[nIndexX][nIndexY].dir.m_dblMinTime = *a_dblMinTime;
		spMNew->m_matrix[nIndexX][nIndexY].dir.m_dblMaxTime = *a_dblMaxTime;		
		spMNew->m_matrix[nIndexX][nIndexY].dir.m_bPeak = false;
		spMNew->m_matrix[nIndexX][nIndexY].dir.cleanTime();
	}
}

/**
 * This function is called from Java and prepares Step C during a matrix transition.
 * In this function, we prepare the times and step width of the impulse responses.
 * 
 * @param a_pMOld	The old matrix
 * @param a_pNTS	NormTimeSlicer
 * @param a_pdblStepTimes	Minimum time of the impulse response
 * @param a_pdblStepWidth	Step width of the impulse response
 * 
 */
extern "C"
__global__ void prepareStepC(Matrix* a_pMOld, NormTimeSlicer* a_pNTS, double *a_pdblStepTimes, double* a_pdblStepWidth)
{
	int nIndexX = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	double dblMinTime = a_pMOld->m_matrix[0][0].dir.m_dblMinTime;
	
	if(nIndexX < STEPWIDTH_IR)
	{
		a_pdblStepTimes[nIndexX] = dblMinTime*a_pNTS->getNormTime(nIndexX);
		a_pdblStepWidth[nIndexX] = a_pNTS->getStepTime(nIndexX)*dblMinTime;
	}
}

/**
 * This function is called from Java and is the final Step C during a matrix transition.
 * In this function, we actually start the transition.
 * 
 * @param a_pMOld	The old matrix
 * @param a_pMNew	The new matrix
 * @param a_pNTS	NormTimeSlicer
 * @param a_pAH		AngleHelper
 * @param a_pnTZ	Theta z index
 * @param a_pNTP	Theta phi index
 * @param a_pdblStepTimes	Minimum time of the impulse response
 * @param a_pdblStepWidth	Step width of the impulse response
 * 
 */
extern "C"
__global__ void MatrixReloadedC(Matrix* a_pMOld, Matrix* a_pMNew, NormTimeSlicer* a_pNTS, AngleHelper* a_pAH, int *a_pnTZ, int *a_pnTP, double *a_pdblStepTimes, double* a_pdblStepWidth)
{
	spreadPower(a_pMOld, a_pMNew, *a_pnTZ, *a_pnTP, a_pNTS, a_pAH, a_pdblStepWidth, a_pdblStepTimes);
}

/**
 * This function is called from Java and returns some data of the matrix.
 * 
 * @param a_pM	The matrix
 * @param a_pnTPMaxIndex	The maximum theta phi index
 * @param a_pdblTPMax	The maximum theta phi
 * @param a_pdblTPMin	The minimum theta phi
 * @param a_pnTZMaxIndex	The maximum theta z index
 * @param a_pnTZMax			The maximum theta z
 * @param a_pdblDistanceAbsolute	The absolute position of the matrix
 * @param a_pdblDistaneDelta		The distance between two matrices
 * @param a_pnCoreIndex				The refractive index of the core
 * @param a_pnSurroundIndex			The refractive index of the surrounding material
 */
extern "C"
__global__ void GetMatrixData(
	Matrix* a_pM, 
	int* a_pnTPMaxIndex, 
	double* a_pdblTPMax,
	double* a_pdblTPMin,
	int* a_pnTZMaxIndex,
	double* a_pnTZMax,
	double* a_pdblDistanceAbsolute,
	double* a_pdblDistanceDelta,
	double* a_pnCoreIndex,
	double* a_pnSurroundIndex
	)
{
	*a_pnTPMaxIndex = a_pM->m_nTPMaxIndex;
	*a_pdblTPMax = a_pM->m_dblTPMax;
	*a_pdblTPMin = a_pM->m_dblTPMin;
	*a_pnTZMaxIndex = a_pM->m_nTZMaxIndex;
	*a_pnTZMax = a_pM->m_dblTZMax;
	*a_pdblDistanceAbsolute = a_pM->m_dblDistanceAbsolute;
	*a_pdblDistanceDelta = a_pM->m_dblDistanceDelta;
	*a_pnCoreIndex = a_pM->m_ta.m_dblIndex2;
	*a_pnSurroundIndex = a_pM->m_ta.m_dblIndex1;
}

/**
 * This function is accessed from Java and returns the resulting impulse responses of the referenced matrix.
 * The given pointer a_pdblPower points to an array large enough to carry all impulse responses of the matrix.
 * 
 * @param a_pM	Pointer to the matrix
 * @param a_pnTZIndex	Theta z index of the cell which impulse response is transferred in this call
 * @param a_pnTPIndex	Theta phi index of the cell which impulse response is transferred in this call
 * @param a_pnNumberOfSteps	Stepsize of the ir. Only filled in one call
 * @param a_pdblMinTime	Minimum time of the impulse responses. Only filled in one call
 * @param a_pdblMaxTime	Maximum time of the impulse responses. Only filled in one call
 * @param a_pdblPower	The power array that carries all impulse responses
 */
extern "C"
__global__ void GetDIRData(
	Matrix* a_pM, 
	int* a_pnTZIndex,
	int* a_pnTPIndex, 
	int* a_pnNumerOfSteps,
	
	double* a_pdblMinTime,
	double* a_pdblMaxTime,
	double* a_pdblPower)
{	
	int nIndexx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int nIndexy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(nIndexx < STEPWIDTH_TZ && nIndexy< STEPWIDTH_TP)
	{
		strDiscreteImpulseResponse *pDIR = &(a_pM->m_matrix[nIndexx][nIndexy].dir);
		
		if(0 == nIndexx && 0 == nIndexy)
		{
			*a_pnNumerOfSteps = pDIR->m_nSteps;
			*a_pdblMinTime = pDIR->m_dblMinTime;
			*a_pdblMaxTime = pDIR->m_dblMaxTime;
		}

		int nIndex1 = nIndexx*STEPWIDTH_IR*STEPWIDTH_TP + nIndexy*STEPWIDTH_IR;
			
		for(int i = 0; i < STEPWIDTH_IR; i++)
		{
			a_pdblPower[nIndex1+i] = pDIR->m_dblPower[i];
		}
	}
}
/**
 * This function is accessed from Java and initlializes an object of the type AngleHelper in the memory the given pointer points to.
 * 
 * @param a_pAH Pointer to the space for the new AngleHelper
 */
extern "C"
__global__ void constructAngleHelper(AngleHelper* a_pAH)
{
	a_pAH->construct();
}

/**
 * This function is the entry point to the executable and prints the sizes of the structs
 * Matrix, NormTimeslicer and AngleHelper since they have to be known for the access from JCuda.
 * 
 */
int main(void)
{
	printf("sizeofMatrix: %ld\n", sizeof(Matrix));
	printf("sizeofNTS: %ld\n", sizeof(NormTimeSlicer));
	printf("sizeofAH: %ld\n", sizeof(AngleHelper));
};



