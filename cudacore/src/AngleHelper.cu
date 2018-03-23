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

#ifndef ANGLEHELPER
#define ANGLEHELPER
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
#include <math.h>

/**
 * This struct provides functions regarding the angular range that is covered by a matrix cell.
 * 
 * @author Thomas Becker
 *
 */
struct AngleHelper 
{
	/**
	 * The amount of steps in which the whole angular range is divided
	 */
	int m_nSteps;
	
	/**
	 * The refractive index of the surrounding material
	 */
	double m_dblIndex0;
	
	/**
	 * The refractive index of the core of the fiber
	 */
	double m_dblIndex1;
	
	/**
	 * The maximum angle that the model handels inside the fiber
	 */
	double m_dblMaxAngleInside;
		
	/**
	 * This array hold the minimum, center and maximum angle outside the fiber for each step
	 */
	double m_dblaAngleRangesOutside[STEPWIDTH_TZ][4];
	
	/**
	 * This array hold the minimum, center and maximum angle inside the fiber for each step
	 */
	double m_dblaAngleRangesInside[STEPWIDTH_TZ][4];
	
	/**
	 * Transfer angle
	 */
	TransferAngle m_ta;
	
	/**
	 * c'tor.
	 * 
	 * This function initializes the angle helper.
	 */
	__device__ void construct()
	{
		m_nSteps = STEPWIDTH_TZ;
		m_dblIndex0 = REFRACTIVE_INDEX_SURROUND;
		m_dblIndex1 = REFRACTIVE_INDEX_CORE;
		double dblMaxAngleOutside = MAX_ANGLE_OUTSIDE;
		m_dblMaxAngleInside = asin(sin(dblMaxAngleOutside*M_PI/180.0)*m_dblIndex0/m_dblIndex1)*180.0/M_PI;

		m_ta.construct(m_dblIndex0, m_dblIndex1);
		
		// fill both data arrays
		// since the angular steps are equidistant outside the fiber, they can't be inside
		for(int i = 0; i < m_nSteps; i++)
		{
			m_dblaAngleRangesOutside[i][0] = dblMaxAngleOutside*(double)i/(double)m_nSteps;
			m_dblaAngleRangesOutside[i][2] = dblMaxAngleOutside*(double)(i+1)/(double)m_nSteps;
			m_dblaAngleRangesOutside[i][1] = (m_dblaAngleRangesOutside[i][0] + m_dblaAngleRangesOutside[i][2])/2.0;
			m_dblaAngleRangesOutside[i][3] = m_dblaAngleRangesOutside[i][2]-m_dblaAngleRangesOutside[i][0];
		}
		
		for(int i = 0; i < m_nSteps; i++)
		{
			m_dblaAngleRangesInside[i][0] = m_ta.getAngle2(m_dblaAngleRangesOutside[i][0]*M_PI/180.0)*180.0/M_PI;
			m_dblaAngleRangesInside[i][1] = m_ta.getAngle2(m_dblaAngleRangesOutside[i][1]*M_PI/180.0)*180.0/M_PI;
			m_dblaAngleRangesInside[i][2] = m_ta.getAngle2(m_dblaAngleRangesOutside[i][2]*M_PI/180.0)*180.0/M_PI;
			m_dblaAngleRangesInside[i][3] = m_dblaAngleRangesInside[i][2]-m_dblaAngleRangesInside[i][0];
		}

		printf("CUDA: AH constructed\n");
	}
	
	/**
	 * This methode returns the maximum angle inside the fiber.
	 * 
	 * @return Maximum angle inside the fiber
	 */
	double getMaxAngleInside()
	{
		return m_dblMaxAngleInside;
	}
	
	/**
	 * This function returns the angular step width for a specific step.
	 * 
	 * @param a_nStep The index of the requested step
	 * 
	 * @return Step width
	 */
	__device__ double getStepWidthForStep(int a_nStep)
	{
		return m_dblaAngleRangesInside[a_nStep][3];
	}

	/**
	 * This function plots the the angular ranges outside the fiber for each cell.
	 */
	void plotRangeOutside() 
	{
		for(int i = 0; i < m_nSteps; i++)
		{
			printf("%f %f %f", m_dblaAngleRangesOutside[i][0], m_dblaAngleRangesOutside[i][1], m_dblaAngleRangesOutside[i][2]);
		}		
	}
	
	/**
	 * This function plots the the angular ranges inside the fiber for each cell.
	 */
	void plotRangeInside() 
	{
		for(int i = 0; i < m_nSteps; i++)
		{
			printf("%f %f %f", m_dblaAngleRangesInside[i][0], m_dblaAngleRangesInside[i][1], m_dblaAngleRangesInside[i][2]);
		}
	}
	
	/**
	 * This function returns the cell index for the given inner angle.
	 * If the requested angle is not coverd but does not exceed the maximum angle by more than 1%, the return value is 
	 * <code>-1</code>, otherwise it is <code>-2</code>.
	 * 
	 * @param a_dblAngle	Inner angle
	 * @return				Cell index.
	 */
	int getIndexForInnerAngle(double  a_dblAngle)
	{
		int nIndex =-2;
	
		for(int i = 0; i < m_nSteps; i++)
		{
			if(a_dblAngle >= m_dblaAngleRangesInside[i][0] && a_dblAngle <= m_dblaAngleRangesInside[i][2])
			{
				nIndex = i;
				break;
			}
		}
		
		if(nIndex == -2)
		{
			// we exceeded the possible range...
			if(0.0 <= a_dblAngle && a_dblAngle <= 1.01*m_dblMaxAngleInside)
			{
				// within 1%
				nIndex = -1;
			}
		}
		
		return nIndex;
	}

	/**
	 * This function returns the center angle for the cell addressed by the given index.
	 * 
	 * @param i	Cell index
	 * @return	Center angle
	 */
	__device__ double getCenterAngleForIndex(int i) 
	{
		return m_dblaAngleRangesInside[i][1];
	}
};

	


