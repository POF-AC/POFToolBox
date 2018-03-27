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
    
    This software contains source code provided by NVIDIA Corporation.
	
    In detail the implementation of the function atomicAdd2 is provided by NVIDIA.

    For details see:
    http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz5AU713DUR 
*/

#ifndef DISCRETEIMPULSERESPONSE
#define DISCRETEIMPULSERESPONSE
#endif

#ifndef NORMTIMESLICER
#include "NormTimeSlicer.cu"
#endif 

#ifndef DEFINES
#include "Defines.cu"
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
 
/**
 * 	This flag defines if our own implementation of atomicAdd (atomicAdd2) should be used
 */
#define OWN_DOUBLE_ATOMICADD

/**
 * This function is a workaround implementation of atomicAdd since for double-precision floating-point numbers atomicAdd is not available on devices with compute capability lower than 6.0 .
 *
 * The implementation is copied from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz5AU713DUR 
 * 
 */
__device__ double atomicAdd2(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do 
	{
        assumed = old;
		old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
 * This is a minimum amount of time by which the time span of our ir is increased in order not to miss the rays with the maximum time.
 */
__device__ static double sDBL_SAFETY_TIME = 1.0E-12;

/**
 * This struct represents an impulse response. It can be of two types:
 * 
 * 	- Peak: A single Dirac impulse
 * 	- Hyperspace: A discrete impulse resonse which step widths are adjusted to the angles of the model
 * 
 * @author Thomas Becker
 *
 */
struct strDiscreteImpulseResponse
{
	/**
	 * Number of steps
	 */
	int	m_nSteps;
	
	/**
	 * This array holds the power distribution over time if we are in normal or Hyperspace mode
	 */
	double m_dblPower[STEPWIDTH_IR];
	
	/**
	 * This array holds the Dirac impulse for the excitation of the first matrix
	 */
	double m_dblPowerPeak[2];
	
	/**
	 * The minimum time of the ir (only used in normal or Hyperspace mode)
	 */
	double m_dblMinTime;
	
	/**
	 * The maximum time of the ir (only used in normal or Hyperspace mode)
	 */
	double m_dblMaxTime;

	/**
	 * If this is <code>true</code>, the ir is based on Dirac impulses
	 */
	bool m_bPeak;
	
	/**
	 * This function resets the power density of each interval.
	 */
	__device__ void cleanTime()
	{
		for(int i = 0; i < STEPWIDTH_IR; i++)
		{
			m_dblPower[i] = 0.0;
		}
	}
	
	/**
	 * This c'tor creates a Hyperspace or Peak ir.
	 * 
	 * @param a_bUseHyperSpace	if <code>true</code> the ir is in Hyperspace mode, otherwise in peak mode
	 */
	__device__ void construct(bool a_bPeak)
	{
		m_nSteps = STEPWIDTH_IR;
		cleanTime();
		m_dblPowerPeak[0] = 0.0;
		m_dblPowerPeak[1] = 0.0;
		m_dblMinTime = 0.0;
		m_dblMaxTime = 0.0;
		m_bPeak = a_bPeak;	
	}

	/**
	 * This method adds a Dirac impulse.
	 * 
	 * @param a_dblTime	Time of the impulse
	 * @param a_dblPower	Power of the impulse
	 * @param a_pNTS	Pointer to the NormTimeSlicer
	 * @param a_nTZIndex	Theta z index
	 * @param a_nTPIndex	Theta phi index
	 * @param a_pPrintAdd	Print debug infos
	 */
	__device__ void addPowerPeak(double a_dblTime, double a_dblPower, NormTimeSlicer* a_pNTS, int a_nTZIndex, int a_nTPIndex, bool a_bPrintAdd)
	{
		if(a_dblPower > 0.0)
		{
			int nIndex = -1;
			
			nIndex = a_pNTS->getIndexForNormTime(a_dblTime/m_dblMinTime);
			
			if(nIndex >= m_nSteps)
			{
				printf("Index Overflow  %d %d %.20e %.20e \n", nIndex, m_nSteps, a_dblTime, m_dblMinTime);
			}
			else
			{
				double* ptr = m_dblPower;
				double dblNewPower = ( a_dblPower/(a_pNTS->getStepTime(nIndex)*m_dblMinTime));	

				#ifdef  OWN_DOUBLE_ATOMICADD
					atomicAdd2((ptr+nIndex), dblNewPower);
				#else
					atomicAdd((float*)(ptr+nIndex),(float)( a_dblPower/(a_pNTS->getStepTime(nIndex)*m_dblMinTime)));
				#endif
			}
		}
	}	
};



