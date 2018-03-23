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
package model.threeDimensional.Angle;

import model.FiberProps;
import model.FresnelTransmission;
import model.TransferAngle;
import model.threeDimensional.Function3D;

/**
 * This class implements the angular power distribution of a points source over thata z and theta phi.
 * 
 * @author Thomas Becker
 *
 */
public class ModalDistribution2DPS implements Function3D 
{
	/**
	 * Fiber properties are necessary to consider the angular transformation and Fresnel losses at the entry
	 */
	protected FiberProps fp;
	
	/**
	 * Helper object to compute the Fresnel losses
	 */
	private FresnelTransmission fEntry;
	
	/**
	 * This Object is responsible for the angular transformation of the power distribution
	 */
	private AngularTransformation2D m_at2D;
	
	/**
	 * This Object is responsible for the angular relation between beta and theta phi 
	 */
	private AngularTransformationBeta2TP m_atB2TP;
	
	/**
	 * The maximum angle that we consider
	 */
	double m_dblMaxAngle;
	
	/**
	 * The minimum angle that we consider
	 */
	double m_dblMinAngle;
	
	/**
	 * Distance of the pointsource from the center of the fiber
	 */
	double m_dblDistanceFromCenter;
	
	/**
	 * Power of the light source in the half space 2 Pi
	 */
	double m_dblPT;
	
	/**
	 * If set to <code>true</code> we neglect refracted modes
	 */
	boolean m_bNeglectRefractedModes;
	
	/**
	 * c'tor. This constructor creates a power distribution according to the parameters.
	 * 
	 * @param a_dblDistanceFromCenter	Distance of the point source from the center of the fiber
	 * @param a_fp						Fiberproperties for the refractive indices and the fiber geometry
	 * @param a_dblMinAngle				Minimum angle that we consider
	 * @param a_dblMaxAngle				Maximum angle that we consider
	 * @param a_dblPT					Power of the light source in the half space 2 Pi
	 * @param a_bNeglectRefractedModes	If set to <code>true</code> we neglect refracted modes
	 * @param a_bUseIntegral			If <code>true</code>, the function is integrated over theta phi
	 */
	public ModalDistribution2DPS(
			double a_dblDistanceFromCenter, 
			FiberProps a_fp, 
			double a_dblMinAngle,
			double a_dblMaxAngle,
			double a_dblPT,
			boolean a_bNeglectRefractedModes,
			boolean a_bUseIntegral
	)
	{
		fEntry = new FresnelTransmission(a_fp.m_dblIndexSurround, a_fp.m_dblIndexCore, true);
		fp = a_fp;
		m_dblDistanceFromCenter = a_dblDistanceFromCenter;
		TransferAngle ta = new TransferAngle(a_fp.m_dblIndexSurround, a_fp.m_dblIndexCore);
		m_dblMaxAngle = ta.getAngle2(a_dblMaxAngle);
		m_dblMinAngle = ta.getAngle2(a_dblMinAngle);
		m_dblPT = a_dblPT;
		m_at2D = new AngularTransformation2D(fp);
		m_atB2TP = new AngularTransformationBeta2TP(fp, m_dblDistanceFromCenter,a_bUseIntegral);
		m_bNeglectRefractedModes = a_bNeglectRefractedModes;
	}
	
	/**
	 * This function returns the power density at the requested angles
	 * 
	 * @param a_dblTZ	Theta z
	 * @param a_dblTP	Theta phi
	 * @return 			Power density
	 */
	public double getValue(double a_dblTZ, double a_dblTP) 
	{
		if(a_dblTZ <= m_dblMaxAngle && a_dblTZ >= m_dblMinAngle)
    	{
    		if(m_bNeglectRefractedModes)
    		{
    			double dblAlpha = Math.acos(Math.sin(a_dblTZ)*Math.sin(a_dblTP));
    			double dblMaxGuidedAngle = Math.PI/2 - fp.getMaximumGuidedAngle(); 
    			
    			if(dblAlpha < dblMaxGuidedAngle)
    			{
    				return 0.0;
    			}
    		}
    		
	        double dblErg =  
	        		m_dblPT*(2.0/Math.PI)*Math.sin(a_dblTZ)
	        		*
	        		m_at2D.getValue(a_dblTZ, a_dblTP)
	        		*
	        		m_atB2TP.getValue(a_dblTZ, a_dblTP);

	        // consider Fresnel losses at the entry
	        dblErg *= fEntry.getTransmissionForAngle(a_dblTZ);
	       
	        if(Double.isNaN(dblErg) || Double.isInfinite(dblErg))
	        {
	        	dblErg = 0.0;
	        }
	        
	    	return dblErg;
    	}
    	else
    	{
    		return 0.0;
    	}
    }
}

