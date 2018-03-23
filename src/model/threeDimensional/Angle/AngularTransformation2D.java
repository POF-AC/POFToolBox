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
 * This class is responsible for the angular transformation of the power distribution due to the fiber entry.
 * 
 * @author Thomas Becker
 */
public class AngularTransformation2D implements Function3D
{
	/**
	 * Fiber properties
	 */
	protected FiberProps fp;
	
	/**
	 * c'tor.
	 * 
	 * @param a_fp	Fiber properties
	 */
	public AngularTransformation2D(FiberProps a_fp)
	{
		fp = a_fp;
	}
	
	/**
	 * This method evaluates the function at the given angles.
	 * 
	 * @param a_dblTZ	Theta z
	 * @param a_dblTP	Theta phi
	 */
	public double getValue(double a_dblTZ, double a_dblTP)  
    {
    	double dblResult =  
    	(Math.cos(a_dblTZ)*fp.m_dblIndexCore)
    	/
    	(
    		Math.sqrt(Math.pow(fp.m_dblIndexSurround, 2)-Math.pow(Math.sin(a_dblTZ)*(fp.m_dblIndexCore), 2))
    	);
    	
        return dblResult;
    }
}
