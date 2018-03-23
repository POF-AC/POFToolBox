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
import model.threeDimensional.Function3D;

/**
 * This class describes the relation between the angle beta and theta phi of a point light source.
 * 
 * @author Thomas Becker
 *
 */
public class AngularTransformationBeta2TP implements Function3D
{
	/**
	 * Fiber properties
	 */
	protected FiberProps fp;
	
	/**
	 * Distance of the light source from the center fo the fiber
	 */
	protected double m_dblDistanceFromCenter;
	
	/**
	 * If <code>true</code>, the function is integrated over theta phi
	 */
	protected boolean m_bUseIntegral;
	

	/**
	 * c'tor.
	 * 
	 * @param a_fp						Fiber properties
	 * @param a_dblDistanceFromCenter	Distance of the light source from the center fo the fiber
	 * @param a_bUseIntegral			If <code>true</code>, the function is integrated over theta phi
	 */
	public AngularTransformationBeta2TP(FiberProps a_fp, double a_dblDistanceFromCenter, boolean a_bUseIntegral)
	{
		fp = a_fp;
		m_dblDistanceFromCenter = a_dblDistanceFromCenter;
		m_bUseIntegral = a_bUseIntegral;
	}
	
	/**
	 * This method evaluates the function at the given angles.
	 * 
	 * @param a_dblTZ	Theta z
	 * @param a_dblTP	Theta phi
	 */
	public double getValue(double a_dblTZ, double a_dblTP) 
    {
    	double dblResult =  0.0;
    	
    	if(!m_bUseIntegral)
    	{
    		dblResult = (Math.sin(a_dblTP))*fp.m_dblRadiusCore
	    	/
	    	(
	    			m_dblDistanceFromCenter
	    			*
	    			Math.sqrt(
	    					1
	    					-
	    					Math.pow(Math.cos(a_dblTP)*fp.m_dblRadiusCore/m_dblDistanceFromCenter, 2)
	    			 )
	    			
	    	);
    	}
    	else
    	{
    		dblResult = 
				Math.PI/2.0
				-
				Math.asin
				(
    				(
    					fp.m_dblRadiusCore/m_dblDistanceFromCenter
    				)
    				*
    				Math.sin(Math.PI/2.0-a_dblTP)
    			);	
    	}
    	
        return dblResult;
    }
}
