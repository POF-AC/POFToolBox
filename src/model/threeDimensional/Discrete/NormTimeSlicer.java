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

import model.GlobalModelSettings;
import model.StrainIndex;
import model.TransferAngle;

/**
 * This class offers functions regarding the non equal time steps that are applied in Hyperspace.
 * Two factors cause the steps to be non equidistant:
 * 	1. Due to Snell's law equidistant steps outside the fiber means non equidistant steps inside the fiber
 * 	2. The transition time is proportional to 1/cos(theta z)
 * 
 * @author Thomas Becker
 *
 */
public class NormTimeSlicer 
{
	/**
	 * Since we consider up to 20 different strains, we can hold 20 NormTimeSlicer maximum
	 */
	private static NormTimeSlicer[] sTS = new NormTimeSlicer[20];
		
	/**
	 * This method returns the only instance of NormTimeSlicer for a specific strain.
	 * 
	 * @param a_nNumberOfSteps	Number of time steps
	 * @param a_dblStrain		Strain that is applied to the matrix
	 * @return					Only Object of NormTimeSlicer for the requested combination
	 */
	public static NormTimeSlicer getInstance(int a_nNumberOfSteps, double a_dblStrain)
	{
		int i = 0;
		for(NormTimeSlicer nts : sTS)
		{
			if(null == nts)
				break;
			
			i++;
			
			if(nts.getSteps() == a_nNumberOfSteps && nts.getStrain() == a_dblStrain)
			{
				return nts;
			}
		}
		
		// synchronized since multiple threads may call this function
		synchronized(sTS)
		{
			sTS[i] = new NormTimeSlicer(a_nNumberOfSteps, a_dblStrain);
		}
			
		return sTS[i];
	}
	
	/**
	 * Minimum time, maximum time and width of each time step
	 */
	private double dblTimeSlices[][]; 
	
	/**
	 * This array holds the normalized temporary center times for each step.
	 */
	private double dblTempTimes[];
	
	/**
	 * Amount of steps
	 */
	private int m_nSteps = 0;
	
	/**
	 * Strain that has to be considered
	 */
	private double m_dblStrain = 0.0;
			
	/**
	 * c'tor that fills all data arrays.
	 * 
	 * @param a_nSteps		Amount of steps
	 * @param a_dblStrain	Strain that has to be considered
	 */
	private NormTimeSlicer(int a_nSteps, double a_dblStrain)
	{
		dblTimeSlices = new double[a_nSteps][3];
		dblTempTimes = new double[a_nSteps];
		
		m_nSteps = a_nSteps;
		m_dblStrain = a_dblStrain;
		
		TransferAngle ta = new TransferAngle(1.0, GlobalModelSettings.getInstance().getCoreIndex());
		
		double dblMaxAngle = 85.0;
		
		// If the fiber is strained we have to adjust the first factor so the normalized interval still starts at 1
		double dblTempFactor = StrainIndex.getStrainIndex(a_dblStrain).getnForAngle(0.0*180.0/Math.PI)/GlobalModelSettings.getInstance().getCoreIndex();
		
		// fill the temp center times first
		for(int i = 0; i < a_nSteps; i++)
		{
			double dblCurrentAngle = ta.getAngle2(((Math.PI/180.0)*(double)i*dblMaxAngle/(double)(a_nSteps-1)));
			// consdier the strain
			dblTempTimes[i] = (StrainIndex.getStrainIndex(a_dblStrain).getnForAngle(dblCurrentAngle*180.0/Math.PI)/GlobalModelSettings.getInstance().getCoreIndex())/(Math.cos(dblCurrentAngle));
			dblTempTimes[i] /= dblTempFactor;
		}
		
		// now fill the real norm times
		for(int i = 0; i < a_nSteps; i++)
		{
			double dblTimeSlice = 0.0;
			
			// special treatment for the first and the last cell
			if(i == 0)
			{
				double dblTime1 = dblTempTimes[i];
				double dblTime2 = dblTempTimes[i+1];
				
				dblTimeSlice = (dblTime1+dblTime2)/2.0 - dblTime1;
				dblTimeSlices[i][0] = dblTime1;
				dblTimeSlices[i][1] = (dblTime1+dblTime2)/2.0;
			}
			else if(i == a_nSteps-1)
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
	 * Returns the index for the given normalized time.
	 * If the requested time is larger than the maximum time, we return the maximum index+1.
	 * 
	 * @param a_dblTime	Normalized time
	 * @return			Index
	 */
	public int getIndexForNormTime(double a_dblTime)
	{
		int nReturn = 0;
		
		for(double da[] : dblTimeSlices)
		{
			if(a_dblTime >= da[0] && a_dblTime <= da[1])
			{		
				break;
			}
			nReturn++;
		}
		
		if(nReturn >= dblTimeSlices.length)
			nReturn = dblTimeSlices.length;
		
		return nReturn;
	}
	
	/**
	 * This method prints the minimum time, maximum time and width for each step.
	 */
	public void printSlices()
	{
		int i = 0;
		for(double[] da : this.dblTimeSlices)
		{
			System.out.println(i + " " + da[0] + " " + da[1] + " " + da[2]);
			i++;
		}
	}
	
	/**
	 * This method prints the temporary center times.
	 */
	public void printTempTimes()
	{
		int i = 0;
		for(double da : this.dblTempTimes)
		{
			System.out.println(i + " " + da);
			i++;
		}
	}
	
	/**
	 * This function is for testing purposes only.
	 * 
	 * @param a_args	Commandline arguments
	 */
	public static void main(String[] a_args)
	{
		GlobalModelSettings.getInstance().setCoreGroupIndex(1.51);
		GlobalModelSettings.getInstance().setCoreIndex(1.49);
		long nStart = System.nanoTime();
		NormTimeSlicer sc = NormTimeSlicer.getInstance(851, 0.0);
		long nEnd = System.nanoTime();
		
		
		System.out.println(sc.getNormTime(10));
		System.out.println(sc.getStepTime(10));
	}

	/**
	 * This function returns the width for the given index.
	 * 
	 * @param a_nIndex	Index
	 * @return			Step width
	 */
	public double getStepTime(int a_nIndex) 
	{
		return dblTimeSlices[a_nIndex][2];
	}
	
	/**
	 * This function returns the normalized time for the given index.
	 * 
	 * @param a_nIndex	Index
	 * @return			Normalized time
	 */
	public double getNormTime(int a_nIndex)
	{
		return (dblTimeSlices[a_nIndex][0] + dblTimeSlices[a_nIndex][1]) / 2.0;
	}
	
	/**
	 * This function returns the amount of steps.
	 * 
	 * @return Steps
	 */
	public int getSteps()
	{
		return m_nSteps;	
	}
	
	/**
	 * This function returns the applied strain.
	 * 
	 * @return Strain
	 */
	public double getStrain()
	{
		return m_dblStrain;
	}
	
}
