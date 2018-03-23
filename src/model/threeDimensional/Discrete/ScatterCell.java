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
package model.threeDimensional.Discrete;

import model.AngleHelper;
import model.FiberProps;

/**
 * This class implements all functionality that is related to the scattering process between one matrix cell and the next matrix.
 * 
 * @author Thomas Becker
 *
 */
public class ScatterCell extends RadialPower
{
	/**
	 * Theta z index of the matrix cell
	 */
	private int m_nIndexTZ;
	
	/**
	 * Theta phi index of the matrix cell
	 */
	private int m_nIndexTP;
	
	/**
	 * c'tor. 
	 * 
	 * @param a_strScatterFile	Full qualified path to the scatter file for this cell.
	 * @param a_nSteps			Number of steps in the theta z direction
	 * @param a_dblThetaZ		Theta z of this cell
	 * @param a_nTZ				Theta z index of this cell
	 * @param a_nTP				Theta phi index of this cell
	 */
	public ScatterCell( String a_strScatterFile, int a_nSteps, double a_dblThetaZ, int a_nTZ, int a_nTP)
	{
		// call the c'tor of the super class
		super(a_strScatterFile, a_nSteps, true, a_dblThetaZ);
		
		m_nIndexTZ = a_nTZ;
		m_nIndexTP = a_nTP;
	}

	/**
	 * This function normalizes all steps of the scatter cell so that the total sum matches the given power.
	 * 
	 * @param a_dblPower
	 */
	public void normalizeToPower(double a_dblPower) 
	{
		double dblPower = this.getTotalPower();
		if(dblPower != 0)
		{
			double dblFactorCorrection = a_dblPower/dblPower;
			normalizeRest(dblFactorCorrection);
		}
	}

	/**
	 * Scatters a Power Peak to the following matrix.
	 * 
	 * @param a_newMatrix	The next matrix
	 * @param a_dblTime		The time of the considered power in this matrix
	 * @param a_dblPower	The considered power
	 * @param a_bSuppressSA	If <code>true</code> scattering and attenuation are neglected
	 */
	public void scatterToNextMatrix(Matrix a_newMatrix, double a_dblTime, double a_dblPower, boolean a_bSuppressSA, boolean a_bPrint) 
	{
		// only apply scattering and attenuation if requested
		if(!a_bSuppressSA)
		{
			// iterate the scatter data for this cell
			int i = 0;
			for(double data : m_dblData)
			{
				// the total power transferred to the target cell is the power density from the scatter matrix 
				// multiplied by the angular step width of the target cell
				double dblFactor = data*AngleHelper.getInstance().getStepWidthForStep(i);

				// skip empty cells
				if(0.0 == dblFactor)
				{
					i++;
					continue;
				}
				else
				{
					double dblTZ = m_dblThetaZInside;
					double dblTZ2 = AngleHelper.getInstance().getCenterAngleForIndex(i);
					
					// mean angle between start and arrival
					double dblMeanAngle = (dblTZ+dblTZ2)/2.0; 
					
					/*
					 *	we calculate the transit-time with the mean angle and consider...
					 *	- the relative distance
					 *	- strain
					 * 	- refractive index depending on the angle
					 */
					double dblNewTime = a_dblTime +  (a_newMatrix.getRelativeDistance()*(1.0+a_newMatrix.getStrain())*a_newMatrix.getCoreIndex(dblMeanAngle))/(Math.cos(dblMeanAngle*Math.PI/180.0)*FiberProps.m_dblSpeedOfLight);
					
					// new power is the total power to scatter multiplied with the factor for the destination cell
					double dblNewPower = dblFactor*a_dblPower;
					
					// Due to the lack of better scatter data we maintain theta phi
					if(0.0 != dblNewPower)
						a_newMatrix.addPowerPeak(i, this.m_nIndexTP,dblNewTime, dblNewPower);
					
					i++;
				}
				
			}
		}
		else
		{
			// no scattering and attenuation, no cry - just calculate the new time depending on the angle to consider modal dispersion
			int i = AngleHelper.getInstance().getIndexForInnerAngle(m_dblThetaZInside);
			int i2 = this.m_nIndexTZ;
			
			if(i != i2)
			{
				System.out.println("Warning: unexpected angle index mismatch: " + i + " vs. " + i2);
			}
						
			// maintain theta phi
			double dblNewTime = a_dblTime +  (a_newMatrix.getRelativeDistance()*(1.0+a_newMatrix.getStrain())*a_newMatrix.getCoreIndex(m_dblThetaZInside))/(Math.cos(m_dblThetaZInside*Math.PI/180.0)*FiberProps.m_dblSpeedOfLight);
			a_newMatrix.addPowerPeak(i, this.m_nIndexTP,dblNewTime, a_dblPower);
		}
		
	}

	/**
	 * This function smoothes the scatter distribution with the given width.
	 * 
	 * @param a_nSteps	The amount of steps over which the smoothing process should be applied
	 */
	public void fixDistribution(int a_nSteps) 
	{
		// clone the original data
		double dblValuesOriginal[] = m_dblData.clone();
		double dblHistory[] = new double[a_nSteps];
		
		// initialize the history
		for(int i = 0; i<dblHistory.length; i++){dblHistory[i] = 0.0;}
		
		// iterate over the whole data array
		for(int nIndexAngle = 0; nIndexAngle < m_dblData.length; nIndexAngle++)
		{
			//fillmode average
			for(int i = 0; i<dblHistory.length; i++)
			{
				int index = nIndexAngle+i-a_nSteps/2;
				
				// if the requested index is outside the valid range, we set the history entry to zero
				if(index < 0)
				{
					dblHistory[i] = 0.0;
				}
				else if(index >= dblValuesOriginal.length)
				{
					dblHistory[i] = 0.0;
				}
				else
					dblHistory[i] = dblValuesOriginal[index];
			}

			// sum up values
			m_dblData[nIndexAngle] = 0.0;
			for(int i = 0; i<dblHistory.length; i++)
			{
				m_dblData[nIndexAngle] += dblHistory[i];
			}
			
			// at the edges of the data array we are not able to consider the expected amount of values
			if(nIndexAngle < a_nSteps/2)
			{
				m_dblData[nIndexAngle]/= (nIndexAngle+a_nSteps/2);
			}
			else if(nIndexAngle > (m_dblData.length-1)-a_nSteps/2)
			{
				m_dblData[nIndexAngle]/= (m_dblData.length-1)-nIndexAngle+a_nSteps/2+1;
			}
			else
				m_dblData[nIndexAngle]/= a_nSteps;
		}	
	}
}
