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
package model;

/**
 * This class provides functions regarding the angular range that is covered by a matrix cell.
 * 
 * @author Thomas Becker
 *
 */
public class AngleHelper 
{
	/**
	 * The amount of steps in which the whole angular range is divided
	 */
	private int m_nSteps;
	
	/**
	 * The refractive index of the surrounding material
	 */
	private double m_dblIndex0;
	
	/**
	 * The refractive index of the core of the fiber
	 */
	private double m_dblIndexCore;
	
	/**
	 * The maximum angle that the model handels inside the fiber
	 */
	private double m_dblMaxAngleInside;
		
	/**
	 * This array hold the minimum, center and maximum angle outside the fiber for each step
	 */
	private double[][] m_dblaAngleRangesOutside;
	
	/**
	 * This array hold the minimum, center and maximum angle inside the fiber for each step
	 */
	private double[][] m_dblaAngleRangesInside;
	
	/**
	 * Transfer angle
	 */
	private TransferAngle m_ta;
	
	/**
	 * This class is a singleton and this member is the only instance
	 */
	private static AngleHelper m_Instance = null;
	
	/**
	 * This methode returns the only instance of AngleHelper
	 * 
	 * @return Instance of AngleHelper
	 */
	public static AngleHelper getInstance()
	{
		if(null == m_Instance)
		{
			// as of now, the maximum angle of 85° and the number of cells is fixed
			m_Instance = new AngleHelper(1.0, GlobalModelSettings.getInstance().getCoreIndex(), 851, 85.0);
		}
		
		return m_Instance;
	}
	
	/**
	 * Private c'tor.
	 * 
	 * @param a_dblIndex0	Refractive index of the surrounding material
	 * @param a_dblIndex1	Refractive index of the core of the fiber
	 * @param a_nSteps		Number of steps
	 * @param a_dblMaxAngleOutside	Maximum angle outside the fiber
	 */
	private AngleHelper(double a_dblIndex0, double a_dblIndexCore, int a_nSteps, double a_dblMaxAngleOutside)
	{
		m_nSteps = a_nSteps;
		m_dblIndex0 = a_dblIndex0;
		m_dblIndexCore = a_dblIndexCore;
		m_dblMaxAngleInside = Math.asin(Math.sin(a_dblMaxAngleOutside*Math.PI/180.0)*a_dblIndex0/a_dblIndexCore)*180.0/Math.PI;
		double dblMaxAngleOutside = a_dblMaxAngleOutside;
		
		m_ta = new TransferAngle(m_dblIndex0, m_dblIndexCore);
		
		m_dblaAngleRangesOutside = new double [a_nSteps][3];
		m_dblaAngleRangesInside = new double [a_nSteps][3];
		
		// fill both data arrays
		// since the angular steps are equidistant outside the fiber, they can't be inside
		for(int i = 0; i < m_nSteps; i++)
		{
			m_dblaAngleRangesOutside[i][0] = dblMaxAngleOutside*(double)i/(double)m_nSteps;
			m_dblaAngleRangesOutside[i][2] = dblMaxAngleOutside*(double)(i+1)/(double)m_nSteps;
			m_dblaAngleRangesOutside[i][1] = (m_dblaAngleRangesOutside[i][0] + m_dblaAngleRangesOutside[i][2])/2.0;
		}
		
		for(int i = 0; i < m_nSteps; i++)
		{
			m_dblaAngleRangesInside[i][0] = m_ta.getAngle2(m_dblaAngleRangesOutside[i][0]*Math.PI/180.0)*180.0/Math.PI;
			m_dblaAngleRangesInside[i][1] = m_ta.getAngle2(m_dblaAngleRangesOutside[i][1]*Math.PI/180.0)*180.0/Math.PI;
			m_dblaAngleRangesInside[i][2] = m_ta.getAngle2(m_dblaAngleRangesOutside[i][2]*Math.PI/180.0)*180.0/Math.PI;
		}
	}
	
	/**
	 * This function is for testing purposes only.
	 * 
	 * @param a_strArgs	Commandline arguments
	 */
	public static void main (String[] a_strArgs)
	{
		GlobalModelSettings.getInstance().setCoreIndex(1.49);
		AngleHelper ah = new AngleHelper(1.0, GlobalModelSettings.getInstance().getCoreIndex(), 851, 85.0);
		ah.plotRangeInside();
	}
	
	/**
	 * This methode returns the maximum angle inside the fiber.
	 * 
	 * @return Maximum angle inside the fiber
	 */
	public double getMaxAngleInside()
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
	public double getStepWidthForStep(int a_nStep)
	{
		return m_dblaAngleRangesInside[a_nStep][2]-m_dblaAngleRangesInside[a_nStep][0];
	}

	/**
	 * This function plots the the angular ranges outside the fiber for each cell.
	 */
	private void plotRangeOutside() 
	{
		for(int i = 0; i < m_nSteps; i++)
		{
			System.out.println("" + m_dblaAngleRangesOutside[i][0] + " " + m_dblaAngleRangesOutside[i][1] + " " + m_dblaAngleRangesOutside[i][2]);
		}		
	}
	
	/**
	 * This function plots the the angular ranges inside the fiber for each cell.
	 */
	private void plotRangeInside() 
	{
		for(int i = 0; i < m_nSteps; i++)
		{
			System.out.println(i + " " + m_dblaAngleRangesInside[i][0] + " " + m_dblaAngleRangesInside[i][1] + " " + m_dblaAngleRangesInside[i][2]);
		}
	}
	
	/**
	 * This function returns the cell index for the given inner angle.
	 * If the requested angle is not coverd but does not exceed the maximum angle by more than 1%, the return value is 
	 * <code>-1</code>, otherwise it is <code>Integer.MIN_VALUE</code>.
	 * 
	 * @param a_dblAngle	Inner angle
	 * @return				Cell index.
	 */
	public int getIndexForInnerAngle(double  a_dblAngle)
	{
		int nIndex = Integer.MIN_VALUE;
	
		// lets see if the angle is in one of our cells
		for(int i = 0; i < m_nSteps; i++)
		{
			if(a_dblAngle >= m_dblaAngleRangesInside[i][0] && a_dblAngle <= m_dblaAngleRangesInside[i][2])
			{
				nIndex = i;
				break;
			}
		}
		
		// nope...
		if(nIndex == Integer.MIN_VALUE)
		{
			// ... we exceeded the possible range...
			if(0.0 <= a_dblAngle && a_dblAngle <= 1.01*this.m_dblMaxAngleInside)
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
	public double getCenterAngleForIndex(int a_nIndex) 
	{
		return this.m_dblaAngleRangesInside[a_nIndex][1];
	}

	
}

