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
import java.util.Vector;


/**
 * This function is a wrapper that can warp multiple impulse responses and is mainly used to consider reflections.
 * 
 * @author Thomas Becker
 *
 */
public class DiscreteImpulseResponseWrapper 
{
	/**
	 * This vector holds all wrapped impulse responses
	 */
	private Vector<DiscreteImpulseResponse> m_vecDIRs;
	
	/**
	 * c'tor that initializes the vector
	 */
	public DiscreteImpulseResponseWrapper()
	{
		m_vecDIRs = new Vector<DiscreteImpulseResponse>();
	}
	
	/**
	 * This function adds the given impulse response to the vector
	 * 
	 * @param a_DIR	The impulse reponse to be added
	 */
	public void addDIR(DiscreteImpulseResponse a_DIR)
	{
		m_vecDIRs.add(a_DIR);
	}
		
	/**
	 * This function writes the resulting impulse response to a file.
	 * 
	 * @param a_strFolder	Full qualified path to the output folder
	 * @param a_strFileName	Name of the output file
	 * @param a_nSteps		Number of steps of the impulse response
	 * @throws Exception	Exception that might occur while creating the file
	 */
	public void createDiscreteFile(String a_strFolder, String a_strFileName, int a_nSteps) throws Exception
    {
		// Step 1 create Array
		Double dblMinTime = null;
		Double dblMaxTime = null;
		
		double dblPower[] = new double[a_nSteps];
		
		for(DiscreteImpulseResponse DIR : m_vecDIRs)
		{
			if(null == dblMinTime || dblMinTime > DIR.dblgetMinTime())
			{
				dblMinTime = DIR.dblgetMinTime();
			}
			
			if(null == dblMaxTime || dblMaxTime < DIR.dblgetMaxTime())
			{
				dblMaxTime = DIR.dblgetMaxTime();
			}
		}
		
		// Step 2 fill the array
		double dblStepTime = (dblMaxTime+DiscreteImpulseResponse.sDBL_SAFETY_TIME-dblMinTime)/a_nSteps;
		
		for(DiscreteImpulseResponse dir : m_vecDIRs)
		{
			for(int i = 0; i < dir.getSteps(); i++)
			{
				double dblTime;
				
				if(dir.m_bUseHyperSpace)
				{
					dblTime = dir.dblgetMinTime()*NormTimeSlicer.getInstance(dir.getSteps(), dir.m_dblStrain).getNormTime(i);
				}
				else
				{
					throw new Exception("Funktioniert nur im HyperSpace!");
				}
				
				double dblValue = dir.getPower()[i];
				// recreate Power
				dblValue *= NormTimeSlicer.getInstance(dir.getSteps(), dir.m_dblStrain).getStepTime(i)*dir.dblgetMinTime();
		
				int nIndex = -1;
				
				nIndex = (int)((dblTime-dblMinTime)/dblStepTime);
				dblPower[nIndex] += dblValue;
			}
		}
		
		File foutD = new File(a_strFolder);
		foutD.mkdirs();
		
		FileWriter fwd = new FileWriter(a_strFolder + a_strFileName);
		BufferedWriter bwd = new BufferedWriter(fwd);
										
		for(int i = 0; i < a_nSteps; i++)
		{
			double dblTime = dblMinTime + i * (dblMaxTime-dblMinTime)/a_nSteps;
								
			double dblValue = dblPower[i];
			bwd.write(dblTime + " " + dblValue + "\n");
		}
		
		bwd.close();
    }
	
	/**
	 * This function creates a file with power peaks of the resulting impulse response.
	 * This kind of output is usually preferred when transferring the impulse response to the frequency domain.
	 * 
	 * @param a_strFolder Full qualified path to the output folder
	 * @param a_strFileName	Filename
	 */
	public void createSpikedFile(String a_strFolder, String a_strFileName)
    {
		try 
		{
			File foutD = new File(a_strFolder);
			foutD.mkdirs();
			
			FileWriter fwd = new FileWriter(a_strFolder + a_strFileName);
			BufferedWriter bwd = new BufferedWriter(fwd);
			
			bwd.write("spiked\n");
			
			double dblTime;
			
			for(DiscreteImpulseResponse dir : m_vecDIRs)
			{
				for(int i = 0; i < dir.getSteps(); i++)
				{
					if(dir.m_bUseHyperSpace)
					{
						dblTime = dir.dblgetMinTime()*NormTimeSlicer.getInstance(dir.getSteps(), dir.m_dblStrain).getNormTime(i);
					}
					else
					{
						throw new Exception("Funktioniert nur im HyperSpace!");
					}
					
					double dblValue = dir.getPower()[i];
					// recreate Power
					dblValue *= NormTimeSlicer.getInstance(dir.getSteps(), dir.m_dblStrain).getStepTime(i)*dir.dblgetMinTime();
					
					bwd.write(dblTime + " " + dblValue + "\n");
				}
			}
			
			bwd.close();
	
		}
		catch(Exception e)
		{
			System.out.println(e);	
		}
    }
}



