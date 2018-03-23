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

import model.FiberProps;

/**
 * This class defines a cell of the matrix and consists basically of a scatter cell and an impulse response.
 *  
 * @author Thomas Becker
 *
 */
public class MatrixCell 
{
	/**
	 * The scatter cell
	 */
	private ScatterCell m_sc;
	
	/**
	 * The impulse response
	 */
	private DiscreteImpulseResponse m_dir;
		
	/**
	 * c'tor. This c'tor creates the impulse response and the scatter cell.
	 *  
	 * @param a_strScatterFile	Full qualified path to the file containing the scatter data
	 * @param a_nScatterSteps	Number of Steps
	 * @param a_dblThetaZ		Theta z of this cell
	 * @param a_matrix			Reference to the outer matrix
	 * @param a_nTZ				Index of theta z
	 * @param a_nTP				Index of theta phi
	 */
	public MatrixCell(String a_strScatterFile, int a_nScatterSteps, double a_dblThetaZ, Matrix a_matrix, int a_nTZ, int a_nTP)
	{
		m_sc = new ScatterCell(a_strScatterFile, a_nScatterSteps, a_dblThetaZ, a_nTZ, a_nTP);
		m_dir = new DiscreteImpulseResponse();
	}
	
	/**
	 * default c'tor.
	 */
	public MatrixCell()
	{
	}
	
	/**
	 * This function creates an identical copy of the cell.
	 * 
	 * @return Copy of the MatrixCell
	 */
	public MatrixCell clone()
	{
		MatrixCell mc = new MatrixCell();
		// The scatter data can be referenced
		mc.m_sc = m_sc;
		// The impulse response has to be copied
		mc.m_dir = m_dir.clone();
		
		return mc;
	}
	
	/**
	 * This function creates a copy of the cell and initializes the impulse response if requested.
	 * 
	 * @param a_newMatrix				The next matrix
	 * @param a_bCreateImpulseResponse	If <code>true</code> the impulse response of the new matrix is initialized
	 * @param a_nNumberOfTimeSlices		Number of steps for the new impulse response
	 * @param a_bUseHyperSpace			Use Hyperspace for the new impulse response if <code>true</code>
	 * @param a_dblOldMinTime			The minimum time of the old impulse response
	 * @param a_dblOldMaxTime			The maximum tiem of the old impulse repsonse
	 * @return							The new MatrixCell
	 */
	public MatrixCell cloneScatter(Matrix a_newMatrix, boolean a_bCreateImpulseResponse, int a_nNumberOfTimeSlices, boolean a_bUseHyperSpace, 
			double a_dblOldMinTime, double a_dblOldMaxTime)
	{
		MatrixCell mc = new MatrixCell();
		mc.m_sc = m_sc;
		mc.m_dir = new DiscreteImpulseResponse();
		
		if(a_bCreateImpulseResponse)
		{
			double dblMinTime = a_newMatrix.getRelativeDistance()*(1.0+a_newMatrix.getStrain())*a_newMatrix.getCoreIndex(0.0)/ FiberProps.m_dblSpeedOfLight + a_dblOldMinTime;
		
			// since the consideration of strain, the refractive index depends on the angle
			double dblMaxTime = a_newMatrix.getRelativeDistance()*(1.0+a_newMatrix.getStrain())*a_newMatrix.getCoreIndex(a_newMatrix.getTZMax())/ (FiberProps.m_dblSpeedOfLight*Math.cos(a_newMatrix.getTZMax()*Math.PI/180.0))
								+ a_dblOldMaxTime;
			mc.m_dir.prepareImpulseResponse(dblMinTime, dblMaxTime, a_nNumberOfTimeSlices, a_bUseHyperSpace, a_newMatrix.getStrain());
		}
		
		return mc;
	}
	
	/**
	 * This function returns the ScatterCell of this MatrixCell.
	 * 
	 * @return	ScatterCell
	 */
	public ScatterCell getScatterCell()
	{
		return m_sc;
	}
	
	public DiscreteImpulseResponse getDIR()
	{
		return m_dir;
	}
	
	/**
	 * This function sets the impulse response of the cell
	 * 
	 * @param a_dir	Impulse response
	 */
	public void setDIR(DiscreteImpulseResponse a_dir)
	{
		m_dir = a_dir;
	}

	/**
	 * This function spreads the power of this cell to the next cell according to the scatter matrix.
	 * 
	 * @param newMatrix		The next matrix
	 * @param a_bSuppressSA	<code>true</code> if scattering and attenuation should be suppressed
	 */
	public void spreadPower(Matrix newMatrix, boolean a_bSuppressSA) 
	{
		// Power spikes or steps?
		if(m_dir.isPeak())
		{
			for(double[] dblTP : m_dir.getPeaks())
			{
				double dblTime = dblTP[0];
				double dblPower = dblTP[1];
				
				m_sc.scatterToNextMatrix(newMatrix, dblTime, dblPower, a_bSuppressSA, false);
			}
		}
		else
		{
			double dblStepWidth;
			double dblTime;
			double dblPower;
			
			int i = 0;
			NormTimeSlicer nts = NormTimeSlicer.getInstance(m_dir.getSteps(), m_dir.m_dblStrain);
			for(double dblPowerRaw : m_dir.getPower())
			{
				if(m_dir.m_bUseHyperSpace)
				{
					
					dblTime = m_dir.dblgetMinTime()*nts.getNormTime(i);
					dblStepWidth = nts.getStepTime(i)*m_dir.dblgetMinTime();
				}
				else
				{
					dblStepWidth = (m_dir.dblgetMaxTime()-m_dir.dblgetMinTime())/(double)m_dir.getSteps();
					dblTime = m_dir.dblgetMinTime() + ((double)i + 0.5) * dblStepWidth;
				}
				dblPower = dblPowerRaw*dblStepWidth;
				
				if(0 != dblPower)
					m_sc.scatterToNextMatrix(newMatrix, dblTime, dblPower, a_bSuppressSA, true);
				i++;
			}
		}
	}

	/**
	 * This function deletes the reference to the scatter cell and can be called to support the garbage collector.
	 */
	public void dropScatterFile() 
	{
		m_sc = null;
	}

	/**
	 * This function exports the scatter data to a text file.
	 * 
	 * @param string		Full qualified path to the scatter file
	 * @throws Exception	Exception that can occur while writing the file
	 */
	public void exportScatterFile(String string) throws Exception 
	{
		m_sc.writeToFile(string, false);
	}
}
