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
 * This class calculates the refractive index and the group index for a specific wavelength at a given wavelength.
 * As of now, only PMMA is supported and the coefficients are taken from
 *  
 * G. Beadie, M. Brindza, R. A. Flynn, A. Rosenberg, and J. S. Shirk. 
 * Refractive index measurements of poly(methyl methacrylate) (PMMA) from 0.4-1.6 μm, Appl. Opt. 54, F139-F143 (2015).
 * 
 * @author Thomas Becker
 *
 */
public class RefractiveIndexHelper 
{
	/**
	 * This function calculates the refractive index for a given wavelength and material.
	 * 
	 * @param a_mat				The given material
	 * @param a_dblWaveLength	The given wavelength
	 * 
	 * @return					The calculated refractive index
	 */
	static public double getIndexForWavelength(Material a_mat, double a_dblWaveLength)
	{
		// formula is based on micrometers, but the parameter is meter
		double dblWLMM = a_dblWaveLength*1E6;
		double dbln2 = Double.NaN;
		
		switch(a_mat)
		{
		case PMMA:
			dbln2 = PMMACoefficients.sm_dblC0+
					PMMACoefficients.sm_dblC1*Math.pow(dblWLMM, 2) + 
					PMMACoefficients.sm_dblC2*Math.pow(dblWLMM, 4) +
					PMMACoefficients.sm_dblC3*Math.pow(dblWLMM, -2) +
					PMMACoefficients.sm_dblC4*Math.pow(dblWLMM, -4) +
					PMMACoefficients.sm_dblC5*Math.pow(dblWLMM, -6) +
					PMMACoefficients.sm_dblC6*Math.pow(dblWLMM, -8);
			break;
		default:
			break;
		}
		
		// formula calculates the second power
		return Math.sqrt(dbln2);
	}
	
	/**
	 * This function calculates the refractive group index for a given wavelength and material.
	 * 
	 * @param a_mat				The given material
	 * @param a_dblWaveLength	The given wavelength
	 * 
	 * @return					The calculated refractive index
	 */
	static public double getGroupIndexForWavelength(Material a_mat, double a_dblWaveLength)
	{
		// we need the normal refractive index
		double dbln = getIndexForWavelength(a_mat, a_dblWaveLength);
		
		// formula is based on micrometers, but the parameter is meter
		double dblWLMM = a_dblWaveLength*1E6;
		double dbldndl2 = Double.NaN; 
		
		/*
		 * We need the derivation of the refractive index over lambda.
		 * Since the formula of the refractive index calculates its square, we need to do some math...
		 * f(x) = g(h(x))
		 * f'(x) = g'(h(x))*h'(x)
		 * => n = sqrt(n^2)
		 * => dn/dlambda = 1/(2*sqrt(n^2))*d(n^2)/dlambda 
		 * 
		 */
		
		// calculate d(n^2)/dlambda
		switch(a_mat)
		{
		case PMMA:
			dbldndl2 = 
					2 * PMMACoefficients.sm_dblC1*Math.pow(dblWLMM, 1) + 
					4 * PMMACoefficients.sm_dblC2*Math.pow(dblWLMM, 3) +
					-2 * PMMACoefficients.sm_dblC3*Math.pow(dblWLMM, -3) +
					-4 * PMMACoefficients.sm_dblC4*Math.pow(dblWLMM, -5) +
					-6 * PMMACoefficients.sm_dblC5*Math.pow(dblWLMM, -7) +
					-8 * PMMACoefficients.sm_dblC6*Math.pow(dblWLMM, -9);
			break;
		default:
			break;
		}
		
		// calculate dn/dlambda
		double dbldndl = 1/(2*Math.sqrt(dbln*dbln))*dbldndl2;
		
		// apply the formula for the refractive group index
		double dblng = dbln/(1+(dblWLMM/dbln)*(dbldndl));
		
		return dblng;
	}
	
	/**
	 * This enum lists all supported materials.
	 * 
	 * @author Thomas Becker
	 *
	 */
	public enum Material
	{
		PMMA
	}
	
	/**
	 * This class holds all necessary indices for PMMA.
	 * 
	 * @author Thomas Becker
	 *
	 */
	static private class PMMACoefficients
	{
		static double sm_dblC0 = 2.1778;
		static double sm_dblC1 = 6.1209E-3;
		static double sm_dblC2 = -1.5004E-3;
		static double sm_dblC3 = 2.3678E-2;
		static double sm_dblC4 = -4.2137E-3;
		static double sm_dblC5 = 7.3417E-4;
		static double sm_dblC6 = -4.5042E-5;
	}
	
	/**
	 * This is a test function for the class RefractiveIndexHelper.
	 * 
	 * @param args Command line arguments (not used)
	 */
	static public void main(String[] args)
	{
		double dblwl = 650E-9;
		
		System.out.println(RefractiveIndexHelper.getGroupIndexForWavelength(Material.PMMA, dblwl)/1.49);
	}
}
