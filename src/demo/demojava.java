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
package demo;

import model.FiberProps;
import model.FresnelTransmission;
import model.GlobalModelSettings;
import model.RefractiveIndexHelper;
import model.RefractiveIndexHelper.Material;
import model.threeDimensional.Angle.*;
import model.threeDimensional.Angle.LightSourceAngularDistribution.LightSourceAngularEnum;
import model.threeDimensional.Discrete.DiscreteImpulseResponse;
import model.threeDimensional.Discrete.DiscreteImpulseResponseWrapper;
import model.threeDimensional.Discrete.Matrix;

/**
 * This class demonstrates how to use the model to simulate a fiber with reflections.
 * 
 * @author Thomas Becker
 *
 */
public class demojava 
{
	/**
	 * This method demonstrates how to use the model to simulate a fiber with reflections.
	 * 
	 * @param a_strPathToScatterFiles	Full qualified path to the scatter files
	 */
	static public void runPlainFiberWithReflections(String a_strPathToScatterFiles, String a_strPathForResults)
	{
		// set the refractive indices of the core
		double dblGroupIndex = RefractiveIndexHelper.getGroupIndexForWavelength(Material.PMMA, 650.0E-9);
		GlobalModelSettings.getInstance().setCoreGroupIndex(dblGroupIndex);
		
		double dblIndex = RefractiveIndexHelper.getGroupIndexForWavelength(Material.PMMA, 650.0E-9);
		GlobalModelSettings.getInstance().setCoreIndex(dblIndex);

		// Fiber length is 16 cm according to the scatter data
		double dblFiberLength = 0.16;
		FiberProps fp = new FiberProps(GlobalModelSettings.getInstance().getCoreIndex(), 1.42, 1.0, dblFiberLength, 0.49E-3);

		// area light source with all mode categories
		ModalDistribution2DArea mps = new ModalDistribution2DArea(fp, 0.0, (Math.PI/2.0), 1.0, false, LightSourceAngularEnum.Y);

		// 12m of fiber need 75 cycles
		int nCycles = 75;
		
		// init Matrix
		Matrix m1 = new Matrix(851, 98, 0.0, dblFiberLength, 1.0, GlobalModelSettings.getInstance().getCoreIndex(), 85.0, 0.0000);
		try
		{
			// load the scatter files
			m1.loadScatterFiles(a_strPathToScatterFiles);
	
			// load the initial power distribution
			m1.loadInitialPowerDistribution(mps, false, 0, false);
			
			// we need a second matrix to swap	
			Matrix m2 = null;
			
			// steps of the impulse responsese
			int nSteps = 85;
			
			// we preinitialize the impulse responses of each matrix to limit the RAM usage
			boolean bDirectManipulation = true;
					
			// Fresnel losses at the end surface of the fiber
			FresnelTransmission f2 = new FresnelTransmission(fp.m_dblIndexCore, fp.m_dblIndexSurround, false);
			
			// no strain
			double dblStrain = 0.0000;
			
			// backup matrix to save the state for the forward propagation 
			// so we can calculate the reflections
			Matrix m3 = null;
				
			// there...
			for(int i = 1; i <= nCycles; i++)
			{	
				System.out.println("Starting Run " + i);
				
				// start the timer
				long nStart = System.nanoTime();
				
				// restore the matrix for the forward propagation if a backup exists (not the case in the first run)
				if(null != m3)
				{
					m1 = m3;
				}
	
				// use 8 threads
				m2 = m1.MatrixReloaded(false, bDirectManipulation, nSteps,8, true, dblStrain);
								
				// swap the matrices, new is the new old...
				m1 = m2;
				
				// save the matrix for the future forward propagation
				m3 = m1.clone();
				
				// create the forward ir for the s5052
				m2.applyFresnelTransmission(f2, false);
				m2.applyAngularFilter(new LightSourceAngularDistribution(LightSourceAngularEnum.S5052_PURE, fp, 1.0));
				DiscreteImpulseResponse dir1h = m2.createImpulseResponse(nSteps, true, true, false);
				
				// create the forward ir for the BPW34
				m2 = m3.clone();
				m2.applyFresnelTransmission(f2, false);
				m2.applyAngularFilter(new LightSourceAngularDistribution(LightSourceAngularEnum.BPW34_PURE, fp, 1.0));
				DiscreteImpulseResponse dir1b = m2.createImpulseResponse(nSteps, true, true, false);
				
				// ... and back again
				// restore m1 because we applied Fresnel losses and receiver characteristics
				m1 = m3.clone();
				// apply reflectivity
				m1.applyFresnelReflection(f2);
				for(int j = 1; j <= i; j++)
				{
					// backward propagation
					m2 = m1.MatrixReloaded(false, bDirectManipulation, nSteps,8, true, dblStrain);
					System.out.println(i + " backward " + j);
					// swap matrices
					m1 = m2;
				}
				
				// apply reflection(s)
				m1.applyConnectorFresnelReflection(f2);
				for(int j = 1; j <= i; j++)
				{	
					// forward propagation
					m2 = m1.MatrixReloaded(false, bDirectManipulation, nSteps,8, true, dblStrain);
					System.out.println(i + " forward again " + j);
					// swap matrices
					m1 = m2;
				}
				
				// create the rebound ir
				m2.applyFresnelTransmission(f2, false);
				
				// backup the matrix so we can export both receivers
				Matrix mb = m2.clone();
				
				// create the ir for the S5052
				m2.applyAngularFilter(new LightSourceAngularDistribution(LightSourceAngularEnum.S5052_PURE, fp, 1.0));
				DiscreteImpulseResponse dir2 = m2.createImpulseResponse(nSteps, true, true, false);
				
				// create a wrapper that can hold both impulse responses
				DiscreteImpulseResponseWrapper dirw = new DiscreteImpulseResponseWrapper();
				dirw.addDIR(dir1h);
				dirw.addDIR(dir2);
				
				// export the irs in discrete and spike form
				dirw.createDiscreteFile(a_strPathForResults,   "S5052_ird_"+ i*0.16 + ".stxt", nSteps*3);
				dirw.createSpikedFile(a_strPathForResults, "S5052_irs_"+ i*0.16 + ".stxt");
				
				// restore matrix
				m2 = mb;
				
				// create the ir for the BPW34
				m2.applyAngularFilter(new LightSourceAngularDistribution(LightSourceAngularEnum.BPW34_PURE, fp, 1.0));
				dir2 = m2.createImpulseResponse(nSteps, true, true, false);
				
				// create a wrapper that can hold both impulse responses
				dirw = new DiscreteImpulseResponseWrapper();
				dirw.addDIR(dir1b);
				dirw.addDIR(dir2);
				
				// export the irs in discrete and spike form
				dirw.createDiscreteFile(a_strPathForResults, "BPW34_ird_"+ i*0.16 + ".stxt", nSteps*3);
				dirw.createSpikedFile(a_strPathForResults, "BPW34_irs_"+ i*0.16 + ".stxt");
				
				long nEnd = System.nanoTime();
				System.out.println(i + " " + (nEnd-nStart) / 1.0E9 +"s");
			}
		}
		catch(Exception exc)
		{
			exc.printStackTrace();
		}
	}
	
	/**
	 * This is the main function that and starts the simulation.
	 * 
	 * @param args	Command line arguments
	 */
	public static void main(String[] args)
	{
		demojava.runPlainFiberWithReflections("/path/to/scatterfiles", "/path/for/results");
	}
}

