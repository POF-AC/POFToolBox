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
 * This class manages data that are related to a fiber.
 * 
 * @author Thomas Becker
 *
 */
public class FiberProps 
{
	/**
	 * The speed of light in vacuum
	 */
	public static double m_dblSpeedOfLight = 299792458.0;
	
	/**
	 * The refractive index of the core
	 */
	public double m_dblIndexCore;
	
	/**
	 * The refractive index of the cladding
	 */
	public double m_dblIndexClad;
	
	/**
	 * The refractive index of the surrounding material
	 */
	public double m_dblIndexSurround;
	
	/**
	 * The radius of the core
	 */
	public double m_dblRadiusCore;
	
	/**
	 * The length of the fiber
	 */
	public double m_dblFiberLength;
	
	/**
	 * c'tor. This c'tor constructs an object of the class Fiberprops by the given parameters.
	 * 
	 * @param a_dblIndexCore		The refractive index of the core
	 * @param a_dblIndexClad		The refractive index of the cladding
	 * @param a_dblIndexSurround	The refractive index of the surrounding material
	 * @param a_dblFiberLength		The length of the fiber
	 * @param a_dblRadiusCore		The radius of the core
	 */
	public FiberProps(double a_dblIndexCore, double a_dblIndexClad, double a_dblIndexSurround,
			double a_dblFiberLength, double a_dblRadiusCore)
	{
		m_dblIndexCore = a_dblIndexCore;
		m_dblIndexClad = a_dblIndexClad;
		m_dblIndexSurround = a_dblIndexSurround;
		m_dblFiberLength = a_dblFiberLength;
		m_dblRadiusCore = a_dblRadiusCore;
	}
	
	/**
	 * This method returns the transit time of the fastest ray, which is a ray with theta z = 0.
	 * 
	 * @return Minimum transit time
	 */
	public double getFastestRayTime()
	{
		return m_dblIndexCore*m_dblFiberLength/m_dblSpeedOfLight;
	}
	
	/**
	 * This method returns the transit time of the slowest guided ray.
	 * 
	 * @return Maximum transit time
	 */
	public double getSlowestRayTime()
	{
		return getFastestRayTime()/Math.cos(getMaximumAngle());
	}
	
	/**
	 * This method returns the maximum angle that can be excited inside the fiber.
	 * 
	 * @return	Maximum excitable angle
	 */
	public double getMaximumAngle()
	{
		return Math.asin(m_dblIndexSurround/m_dblIndexCore);
	}
	
	/**
	 * This method returns the maximum guided angle inside the fiber.
	 * 
	 * @return	Maximum guided angle
	 */
	public double getMaximumGuidedAngle()
	{
		return Math.acos(m_dblIndexClad/m_dblIndexCore);
	}
}


