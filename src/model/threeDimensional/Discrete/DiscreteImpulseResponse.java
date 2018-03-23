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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Vector;

/**
 * This class represents an impulse response. It can be of three types:
 * 
 * 	- Peak: A set of Dirac impulses
 * 	- Normal: A discrete impulse response with equidistant steps
 * 	- Hyperspace: A discrete impulse resonse which step widths are adjusted to the angles of the model
 * 
 * @author Thomas Becker
 *
 */
public class DiscreteImpulseResponse 
{
	/**
	 * Number of steps
	 */
	private int m_nSteps;
	
	/**
	 * Applied strain
	 */
	public double m_dblStrain;
	
	/**
	 * This array holds the power distribution over time if we are in normal or Hyperspace mode
	 */
	private double m_dblPower[];
	
	/**
	 * This vector holds the Dirac impulses if this mode is chosen
	 */
	private Vector<double[]> m_vecPowerPeaks;
	
	/**
	 * The minimum time of the ir (only used in normal or Hyperspace mode)
	 */
	private double m_dblMinTime;
	
	/**
	 * The maximum time of the ir (only used in normal or Hyperspace mode)
	 */
	private double m_dblMaxTime;
	
	/**
	 * This is a minimum amount of time by which the time span of our ir is increased in order not to miss the rays with the maximum time.
	 */
	static public double sDBL_SAFETY_TIME = 1.0E-12;
	
	/**
	 * If this is <code>true</code>, the ir is based on Dirac impulses
	 */
	boolean m_bPeak = true;
	
	/**
	 * If this is <code>true</code>, we are in non equidistant Hyperspace mode
	 */
	boolean m_bUseHyperSpace = false;
	
	/**
	 * This c'tor creates a normal or Hyperspace ir.
	 * 
	 * @param a_bUseHyperSpace	if <code>true</code> the ir is in Hyperspace mode, otherwise in normal mode
	 */
	public DiscreteImpulseResponse(boolean a_bUseHyperSpace)
	{
		m_vecPowerPeaks = null;
		m_bUseHyperSpace = a_bUseHyperSpace;
		m_bPeak = false;
	}
	
	/**
	 * c'tor that constructs a peak impulse response.
	 */
	public DiscreteImpulseResponse()
	{
		m_vecPowerPeaks = new Vector<double[]>();
		m_bUseHyperSpace = false;
		m_bPeak = true;
	}
	
	/**
	 * This c'tor creates a new impulse response in normal or peak mode according to the parameters.
	 * 
	 * @param a_nSteps		Number of steps if in normal mode
	 * @param a_dblMinTime	Minimum time if in normal mode
	 * @param a_dblMaxTime	Maximum time if in normal mode
	 * @param a_dblpArray	Power array if in normal mode
	 * @param a_bPeak		if<code>true</code> the new ir is created in peak mode
	 * @param a_dblStrain	Strain that has to be applied
	 */
	public DiscreteImpulseResponse(int a_nSteps, double a_dblMinTime, double a_dblMaxTime, double a_dblpArray[], boolean a_bPeak, double a_dblStrain)
	{
		this(false);
		m_nSteps = a_nSteps;
		m_dblStrain = a_dblStrain;
		m_dblMinTime = a_dblMinTime;
		m_dblMaxTime = a_dblMaxTime;
		if(a_dblpArray != null)
			m_dblPower = a_dblpArray.clone();
		m_bPeak = a_bPeak;
	}
	
	/**
	 * This method clones the current ir.
	 * 
	 * @return The new impulse response
	 */
	public DiscreteImpulseResponse clone()
	{
		DiscreteImpulseResponse dir = new DiscreteImpulseResponse(m_nSteps, m_dblMinTime, m_dblMaxTime, m_dblPower, m_bPeak, m_dblStrain);
		
		// set the remaining parameters that were not covered by the c'tor
		if(null != m_vecPowerPeaks)
			dir.m_vecPowerPeaks = (Vector<double[]>) m_vecPowerPeaks.clone();
		dir.m_bUseHyperSpace = m_bUseHyperSpace;
		
		return dir;
		
	}
	
	/**
	 * This methode returns the array of Dirac impulses when in peak mode.
	 * 
	 * @return	Vector with Dirac impulses
	 */
	public Vector<double[]> getPeaks()
	{
		return m_vecPowerPeaks;
	}
	
	/**
	 * This method adds a Dirac impulse.
	 * 
	 * @param a_dblTime	Time of the impulse
	 * @param a_dblPower	Power of the impulse
	 */
	public synchronized void addPowerPeak(double a_dblTime, double a_dblPower)
	{
		// negative power does not make any sense
		if(a_dblPower < 0.0)
		{
			a_dblPower = 0.0;
		}

		if(this.m_bPeak)
		{
			m_vecPowerPeaks.add(new double[]{a_dblTime, a_dblPower});
		}
		else
		{
			double dblStepTime = (m_dblMaxTime+sDBL_SAFETY_TIME-m_dblMinTime)/(double)m_nSteps;
			
			int nIndex = -1;
			
			if(m_bUseHyperSpace)
			{
				nIndex = NormTimeSlicer.getInstance(m_nSteps, m_dblStrain).getIndexForNormTime(a_dblTime/m_dblMinTime);
			}
			else
			{
				nIndex = (int)((a_dblTime-m_dblMinTime)/dblStepTime);
			}
			
				
			if(nIndex >= m_nSteps)
			{
				// notify the user but do not throw an exception since the result of the simulation may still be interesting
				System.out.println("Index Overflow");
			}
			else
			{
				if(m_bUseHyperSpace)
				{
					m_dblPower[nIndex] += a_dblPower/(NormTimeSlicer.getInstance(m_nSteps, m_dblStrain).getStepTime(nIndex)*m_dblMinTime);
				}
				else
				{
					m_dblPower[nIndex] += a_dblPower/dblStepTime;
				}
			}
		}
	}
	
	/**
	 * This method creates an impulse reponse from the given matrix.
	 * 
	 * @param a_nSteps	Number of steps
	 * @param a_matrix	The matrix holding all the cell impulse responses	
	 * @param a_dblMin	Minimum time
	 * @param a_dblMax	Maximum time
	 * @param a_bAllEqual	if <code>true</code> all cell impulse responses have the same spacing
	 * @throws Exception
	 */
	public void createImpulseResponse(int a_nSteps, MatrixCell[][] a_matrix, double a_dblMin, double a_dblMax, boolean a_bAllEqual) throws Exception 
	{
		if(a_dblMax == 0.0)
		{
			a_dblMax = 1.0E-12;
		}
	
		
		m_nSteps = a_nSteps;
		m_dblMinTime = a_dblMin;
		
	
		m_dblMaxTime = a_dblMax;
		
		m_dblPower = new double[a_nSteps];
		m_bPeak = false;
		// if we are in Hyperspace mode we will fix the step time when necessary
		double dblStepTime = (m_dblMaxTime+sDBL_SAFETY_TIME-m_dblMinTime)/(double)m_nSteps;
		
		int nTZMax = a_matrix.length-1;
		int nTPMax = a_matrix[0].length-1;
		int nTZStart = 0;
		int nTPStart = 0;
		
		// fill power array
		for(int nTZ = nTZStart; nTZ <= nTZMax; nTZ++)
		{
			for(int nTP = nTPStart; nTP <= nTPMax; nTP++)
			{
				MatrixCell mc = a_matrix[nTZ][nTP];
				if(null != mc && mc.getDIR().isPeak())
				{
					for(double[] da : mc.getDIR().getPeaks())
					{
						double dblTime = da[0];
						
						int nIndex;
						
						if(m_bUseHyperSpace)
						{
							nIndex = NormTimeSlicer.getInstance(a_nSteps, m_dblStrain).getIndexForNormTime(dblTime/m_dblMinTime);
							dblStepTime = NormTimeSlicer.getInstance(a_nSteps, m_dblStrain).getStepTime(nIndex)*m_dblMinTime;
						}
						else
						{
							nIndex = (int)((dblTime-m_dblMinTime)/dblStepTime);
						}
						
						if(nIndex >= m_nSteps)
						{
							System.out.println(nIndex);
						}
						else
						{
							double dblPower = da[1]/dblStepTime;
	
							m_dblPower[nIndex] += dblPower;
						}
					}
				}
				else if(null != mc)
				{
					if(a_bAllEqual)
					{
						if(null != mc.getDIR().getPower())
						{
							for(int i = 0; i < mc.getDIR().getPower().length; i++)
							{
								double dblPower = mc.getDIR().getPower()[i];
								
								m_dblPower[i] += dblPower;
							}
						}
					}
					else
					{
						throw new Exception("Not implemented!");
					}
				}
			}
		}
	}
	
	/**
	 * This method returns <code>true</code> if we are in Dirac impulse mode
	 * 
	 * @return <code>true</code> if we are in Dirac impulse mode
	 */
	public boolean isPeak()
	{
		return m_bPeak;
	}
	
	/**
	 * This method returns the amount of steps.
	 * 
	 * @return Amount of steps
	 */
	public int getSteps()
	{
		return m_nSteps;
	}
	
	/**
	 * This function returns the minimum time of the ir.
	 * 
	 * @return Minimum time
	 */
	public double dblgetMinTime()
	{
		return m_dblMinTime;
	}
	
	/**
	 * This function returns the maximum time of the ir.
	 * 
	 * @return maximum time
	 */
	public double dblgetMaxTime()
	{
		return m_dblMaxTime;
	}
	
	/**
	 * This method returns the power array if in normal or Hyperspace mode
	 * 
	 * @return Power array
	 */
	public double[] getPower()
	{
		return m_dblPower;
	}

	/**
	 * This method calculates the total power that is held by the ir.
	 * 
	 * @return Total power
	 */
	public double getTotalPower() 
	{
		double dblTotalPower = 0.0;
		if(m_bPeak)
		{
			for(double[] da : m_vecPowerPeaks)
			{
				dblTotalPower+=da[1];
			}
		}
		else
		{
			if(m_nSteps > 0)
			{
				double dblTimePerCell = (m_dblMaxTime-m_dblMinTime)/(double)m_nSteps;
				for(int i = 0; i < m_dblPower.length; i++)
				{
					if(m_bUseHyperSpace)
					{
						dblTimePerCell = NormTimeSlicer.getInstance(m_nSteps, m_dblStrain).getStepTime(i)*m_dblMinTime;
					}
					dblTotalPower += dblTimePerCell*m_dblPower[i];
				}
			}
		}
		return dblTotalPower;
	}

	/**
	 * This method prepares a normal or Hyperspace impulse response.
	 * 
	 * @param a_dblMinTime		Minimum time of the ir
	 * @param a_dblMaxTime		Maximum time of the ir
	 * @param a_nNumberOfTimeSlices	Number of steps
	 * @param a_bUseHyperSpace	If <code>true</code> the new ir will be in Hyperspace mode
	 * @param a_dblStrain		Applied strain
	 */
	public void prepareImpulseResponse(double a_dblMinTime, double a_dblMaxTime, int a_nNumberOfTimeSlices, boolean a_bUseHyperSpace, double a_dblStrain) 
	{
		// Adjust the maximum time if set to zero
		if(a_dblMaxTime == 0.0)
		{
			a_dblMaxTime = 1.0E-12;
		}
		
		m_dblStrain = a_dblStrain;
		
		m_dblMinTime = a_dblMinTime;
		m_dblMaxTime = a_dblMaxTime;
		
		m_nSteps = a_nNumberOfTimeSlices;
		m_dblPower = new double[m_nSteps];
		
		m_bUseHyperSpace = a_bUseHyperSpace;
		
		m_bPeak = false;
	}
	
	/**
	 * This function writes the impulse response to a discrete file.
	 * 
	 * @param a_strFolder	Full qualified path to the destination folder
	 * @param a_strFileName	Name of the output file
	 */
	public void createDiscreteFile(String a_strFolder, String a_strFileName)
    {
    	try
		{
			try 
			{
				File foutD = new File(a_strFolder);
				foutD.mkdirs();

				FileWriter fwd = new FileWriter(a_strFolder + a_strFileName);
				BufferedWriter bwd = new BufferedWriter(fwd);
												
				for(int i = 0; i < m_nSteps; i++)
				{
					double dblTime = m_dblMinTime + i * (m_dblMaxTime-m_dblMinTime)/m_nSteps;
					
					if(m_bUseHyperSpace)
					{
						dblTime = m_dblMinTime*NormTimeSlicer.getInstance(m_nSteps, m_dblStrain).getNormTime(i);
					}
					
					double dblValue = this.m_dblPower[i];
					bwd.write(dblTime + " " + dblValue + "\n");
				}
				
				bwd.close();
				fwd.close();
			} 
			catch (IOException e) 
			{
				e.printStackTrace();
			}
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
    }
	
	/**
	 * This function creates a spike file of this impulse response. The resulting file does not contain power densities but
	 * Dirac power spikes at discrete positions.
	 * 
	 * @param a_strFolder	Full qualified path to the destination folder
	 * @param a_strFileName	Name of the output file
	 */
	public void createSpikedFile(String a_strFolder, String a_strFileName)
    {
    	try
		{		
			try 
			{
				File foutD = new File(a_strFolder);
				foutD.mkdirs();

				FileWriter fwd = new FileWriter(a_strFolder + a_strFileName);
				BufferedWriter bwd = new BufferedWriter(fwd);
				
				bwd.write("spiked\n");
				
				double dblTime;
												
				for(int i = 0; i < m_nSteps; i++)
				{
					if(m_bUseHyperSpace)
					{
						dblTime = m_dblMinTime*NormTimeSlicer.getInstance(m_nSteps, m_dblStrain).getNormTime(i);
					}
					else
					{
						throw new Exception("Funktioniert nur im HyperSpace!");
					}
					
					double dblValue = this.m_dblPower[i];
					// recreate Power
					dblValue *= NormTimeSlicer.getInstance(m_nSteps, m_dblStrain).getStepTime(i)*m_dblMinTime;
					
					bwd.write(dblTime + " " + dblValue + "\n");
				}
				
				bwd.close();
				fwd.close();
			} 
			catch (IOException e) 
			{
				e.printStackTrace();
			}
		}
		catch(Exception e)
		{
			System.out.println(e);	
		}
    }
}



