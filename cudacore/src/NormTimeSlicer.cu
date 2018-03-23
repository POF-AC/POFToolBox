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

#ifndef NORMTIMESLICER
#define NORMTIMESLICER
#endif


#ifndef TRANSFERANGLE
#include "TransferAngle.cu"
#endif

#ifndef DEFINES
#include "Defines.cu"
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <ctime>

/**
 * This struct offers functions regarding the non equal time steps that are applied in Hyperspace.
 * Two factors cause the steps to be non equidistant:
 * 	1. Due to Snell's law equidistant steps outside the fiber means non equidistant steps inside the fiber
 * 	2. The transition time is proportional to 1/cos(theta z)
 * 
 * @author Thomas Becker
 *
 */
struct NormTimeSlicer
{
	/**
	 * Minimum time, maximum time and width of each time step
	 */
	double dblTimeSlices[STEPWIDTH_IR][3]; 
	
	/**
	 * This array holds the normalized temporary center times for each step.
	 */
	double dblTempTimes[STEPWIDTH_IR];

	/**
	 * <code>true</code> if the instance of the struct has already been initialized
	 */
	bool m_initialized = false;

	/**
	 * c'tor that fills all data arrays.
	 * 
	 * We consider a maximum angle of 85° outside the fiber and a refractive index of 1.49.
	 */
	__host__ __device__ void construct()
	{
		m_initialized = true;
		TransferAngle ta;
		ta.construct(1.0, 1.49);

		double dblMaxAngle = 85.0;	

		for(int i = 0; i < STEPWIDTH_IR; i++)
		{
			double dblCurrentAngle = ta.getAngle2(((M_PI/180.0)*(double)i*dblMaxAngle/(double)(STEPWIDTH_IR-1)));
			dblTempTimes[i] = 1/cos(dblCurrentAngle);
		}

		for(int i = 0; i < STEPWIDTH_IR; i++)
		{
			double dblTimeSlice = 0.0;
			
			if(i == 0)
			{
				double dblTime1 = dblTempTimes[i];
				double dblTime2 = dblTempTimes[i+1];
				
				dblTimeSlice = (dblTime1+dblTime2)/2.0 - dblTime1;
				dblTimeSlices[i][0] = dblTime1;
				dblTimeSlices[i][1] = (dblTime1+dblTime2)/2.0;
			}
			else if(i == STEPWIDTH_IR-1)
			{
				double dblTime1 = dblTempTimes[i-1];
				double dblTime2 = dblTempTimes[i];
				
				dblTimeSlice = dblTime2 - (dblTime1+dblTime2)/2.0;
				dblTimeSlice*=2.0;
				dblTimeSlices[i][0] = (dblTime1+dblTime2)/2.0;
				dblTimeSlices[i][1] = dblTime2;
				dblTimeSlices[i][1] = dblTimeSlices[i][0]+dblTimeSlice;
			}
			else
			{
				double dblTime1 = dblTempTimes[i-1];
				double dblTime2 = dblTempTimes[i];
				double dblTime3 = dblTempTimes[i+1];
				
				dblTimeSlice = (dblTime2+dblTime3)/2.0 - (dblTime1+dblTime2)/2.0;
				dblTimeSlices[i][0] = (dblTime1+dblTime2)/2.0;
				dblTimeSlices[i][1] = (dblTime2+dblTime3)/2.0;
			}
			
			dblTimeSlices[i][2] = dblTimeSlice;
		}
	}

	/**
	 * c'tor that fills the data arrays at the given index.
	 * 
	 * We consider a maximum angle of 85° outside the fiber and a refractive index of 1.49.
	 * 
	 * @param i	Index of the arrays.
	 */
	__host__ __device__ void construct1D(int i)
	{
		m_initialized = true;
		TransferAngle ta;
		ta.construct(1.0, 1.49);

		double dblMaxAngle = 85.0;	

		double dblCurrentAngle = ta.getAngle2(((M_PI/180.0)*(double)i*dblMaxAngle/(double)(STEPWIDTH_IR-1)));
		dblTempTimes[i] = 1/cos(dblCurrentAngle);
		

		double dblTimeSlice = 0.0;
		
		if(i == 0)
		{
			double dblTime1 = dblTempTimes[i];
			double dblTime2 = dblTempTimes[i+1];
			
			dblTimeSlice = (dblTime1+dblTime2)/2.0 - dblTime1;
			dblTimeSlices[i][0] = dblTime1;
			dblTimeSlices[i][1] = (dblTime1+dblTime2)/2.0;
		}
		else if(i == STEPWIDTH_IR-1)
		{
			double dblTime1 = dblTempTimes[i-1];
			double dblTime2 = dblTempTimes[i];
			
			dblTimeSlice = dblTime2 - (dblTime1+dblTime2)/2.0;
			dblTimeSlice*=2.0;
			dblTimeSlices[i][0] = (dblTime1+dblTime2)/2.0;
			dblTimeSlices[i][1] = dblTime2;
			dblTimeSlices[i][1] = dblTimeSlices[i][0]+dblTimeSlice;
		}
		else
		{
			double dblTime1 = dblTempTimes[i-1];
			double dblTime2 = dblTempTimes[i];
			double dblTime3 = dblTempTimes[i+1];
			
			dblTimeSlice = (dblTime2+dblTime3)/2.0 - (dblTime1+dblTime2)/2.0;
			dblTimeSlices[i][0] = (dblTime1+dblTime2)/2.0;
			dblTimeSlices[i][1] = (dblTime2+dblTime3)/2.0;
		}
		
		dblTimeSlices[i][2] = dblTimeSlice;
		
	}

	/**
	 * Returns the index for the given normalized time.
	 * If the requested time is larger than the maximum time, we return the maximum index+1.
	 * 
	 * @param a_dblTime	Normalized time
	 * @return			Index
	 */
	__host__ __device__ int getIndexForNormTime(double a_dblTime)
	{
		int nReturn = 0;
		
		for(int i = 0; i < STEPWIDTH_IR; i++)
		{
		
			if(a_dblTime >= dblTimeSlices[i][0] && a_dblTime <= dblTimeSlices[i][1])
			{		
				break;
			}
			nReturn++;
		}
		
		if(nReturn >= STEPWIDTH_IR)
			nReturn = STEPWIDTH_IR;
		
		return nReturn;
	}
	
	/**
	 * This function returns the width for the given index.
	 * 
	 * @param a_nIndex	Index
	 * @return			Step width
	 */
	__device__ double getStepTime(int a_nIndex) 
	{
		return dblTimeSlices[a_nIndex][2];
	}
	
	/**
	 * This function returns the normalized time for the given index.
	 * 
	 * @param a_nIndex	Index
	 * @return			Normalized time
	 */
	__device__ double getNormTime(int a_nIndex)
	{
		return (dblTimeSlices[a_nIndex][0] + dblTimeSlices[a_nIndex][1]) / 2.0;
	}

	/**
	 * This method prints the minimum time, maximum time and width for each step.
	 */
	__host__ __device__ void printSlices()
	{
		for(int i = 0; i < STEPWIDTH_IR; i++)
		{
			printf("%d %.16e %.16e %.16e\n", i, dblTimeSlices[i][0], dblTimeSlices[i][1], dblTimeSlices[i][2]);
		}
	}
};


