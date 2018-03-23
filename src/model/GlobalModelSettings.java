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
 * This class holds variables that are shared between different parts of the model and are therefore held in a static object.
 * 
 * @author Thomas Becker
 *
 */
public class GlobalModelSettings 
{
	/**
	 * The only instance of the GlobalModelSettings
	 */
	static private GlobalModelSettings m_Instance;
	
	/**
	 * This method returns the static instance of the GlobalModelSettings.
	 * 
	 * @return	Static instance of GlobalModelSettings
	 */
	static public GlobalModelSettings getInstance()
	{
		if(null == m_Instance)
		{
			m_Instance = new GlobalModelSettings();
		}
		
		return m_Instance;
	}
	
	/**
	 * Refractive index of the core of the fiber
	 */
	private double m_dblIndexCore;
	
	/**
	 * Refractive group index of the core of the fiber
	 */
	private double m_dblGroupIndexCore;
	
	/**
	 * This method returns the refractive index of the core of the fiber.
	 * 
	 * @return	 Refractive index of the core of the fiber
	 */
	public double getCoreIndex()
	{
		return m_dblIndexCore;
	}
	
	/**
	 * This method sets the refractive index of the core of the fiber.
	 * 
	 * @param a_dblCoreIndex	 Refractive index of the core of the fiber
	 */
	public void setCoreIndex(double a_dblCoreIndex)
	{
		m_dblIndexCore = a_dblCoreIndex;
	}
	
	/**
	 * This method returns the refractive group index of the core of the fiber.
	 * 
	 * @return	 Refractive group index of the core of the fiber
	 */
	public double getCoreGroupIndex()
	{
		return m_dblGroupIndexCore;
	}
	
	/**
	 * This method sets the refractive group index of the core of the fiber.
	 * 
	 * @param a_dblCoreIndex	 Refractive group index of the core of the fiber
	 */
	public void setCoreGroupIndex(double a_dblCoreGroupIndex)
	{
		m_dblGroupIndexCore = a_dblCoreGroupIndex;
	}
}
