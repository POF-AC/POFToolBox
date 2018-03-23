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

import model.RefractiveIndexHelper.Material;

/**
 * If a SI-POF is strained its refractive index is affected. The change in the refractive index depends on the direction inside the POF 
 * and on the lateral deformation of the POF which can be described by the Poisson's ratio. This class provides functions to calculate
 * the refractive index that a ray of a certain angle is facing.
 * 
 * @author Thomas Becker
 *
 */
public class StrainIndex 
{
	/**
	 * We can handle up to 20 strain states which are held in a static array of the type StrainIndex
	 */
	private static StrainIndex[] m_sSI = new StrainIndex[20];
	
	/**
	 * Elasto optic constant p11
	 */
	private double m_dblp11 = 0.3;
	
	/**
	 * Elasto optic constant p12
	 */
	private double m_dblp12 = 0.297;
		
	/**
	 * The refractive index of the core in the instrained state
	 */
	private double m_dblncore0 = GlobalModelSettings.getInstance().getCoreGroupIndex();
	
	/**
	 * The Poisson's ratio
	 */
	static private double m_dblPoison = 0.43;
	
	/**
	 * The refractive index in the forward direction
	 */
	private double m_dblnx = 0.0;
	
	/**
	 * The refractive index in the lateral direction
	 */
	private double m_dblny = 0.0;
	
	/**
	 * The strain
	 */
	private double m_dblStrain = 0.0;
	
	/**
	 * Private c'tor since the access to the strain objects is designed as a singleton.
	 * 
	 * @param a_dblStrain1	The strain for this object
	 */
	private StrainIndex(double a_dblStrain1)
	{
		// strains for the three directions
		double dble1 = 0.0;
		double dble2 = 0.0;
		double dble3 = 0.0;
		
		m_dblStrain = a_dblStrain1;
		
		// calculate the strains in the lateral directions via Poisson's ratio
		dble1 = a_dblStrain1;
		dble2 = dble3 = -dble1*m_dblPoison;
		
		// refractive index in the forward direction
		
		// Delta B2 + Delta B3
		double dbl2p3 = 2*m_dblp12*dble1+(m_dblp11+m_dblp12)*(dble2+dble3);
		
		// Delta B2 - Delta B3
		double dbl2m3 = (m_dblp11-m_dblp12)*(dble2-dble3);
		
		// note: dbl2m3 is not used here because it is applied to distinguish between both polarizations
		// since we consider equally polarized light, the term can be neglected
		m_dblnx = 1/Math.sqrt(1/Math.pow(m_dblncore0, 2) + dbl2p3/2);
		
		
		// for the lateral index, we simply switch the strains
		dble2 = a_dblStrain1;
		dble1 = dble3 = -dble2*m_dblPoison;
		dbl2p3 = 2*m_dblp12*dble1+(m_dblp11+m_dblp12)*(dble2+dble3);
		dbl2m3 = (m_dblp11-m_dblp12)*(dble2-dble3);
		
		m_dblny = 1/Math.sqrt(1/Math.pow(m_dblncore0, 2) + dbl2p3/2);
	}
	
	/**
	 * The default Poisson ratio is 0.43. It can however be adjusted with this method.
	 * 
	 * @param a_dblPoison	The value for Poisson's ratio
	 */
	public static void setPoison(double a_dblPoison)
	{
		m_dblPoison = a_dblPoison;
	}
	
	/**
	 * Get the internal index for the given Strain value. 
	 * If we don't handle this strain yet, we create a new entry for it.
	 * 
	 * @param a_dblStrain	The given strain value
	 * @return				The corresponding index
	 */
	public static StrainIndex getStrainIndex(double a_dblStrain)
	{
		int i = 0;
		for(StrainIndex si : m_sSI)
		{
			if(null == si)
				break;
			
			i++;
			
			if(si.getStrain() == a_dblStrain)
			{
				return si;
			}
		}
	
		// multiple threads may call this function, so be careful when adding new objects
		synchronized(m_sSI)
		{
			m_sSI[i] = new StrainIndex(a_dblStrain);
		}
		
		return m_sSI[i];
	}
		
	/**
	 * This method calculates the refractive index for a specific angle.
	 * 
	 * @param a_dblAngle	The angle theta z
	 * @return				The refractive index
	 */
	public double getnForAngle(double a_dblAngle)
	{
		double dblAnglePi = a_dblAngle*Math.PI/180.0;
		
		double dbln = Math.sqrt(Math.pow(Math.cos(dblAnglePi)*m_dblnx, 2) + Math.pow(Math.sin(dblAnglePi)*m_dblny, 2));
		
		return dbln;
	}
	
	/**
	 * This function plots the development of the refractive index over the angle theta z.
	 */
	public void printn()
	{
		for(int i = 0; i <= 90; i++)
		{
			System.out.println(i + ": " + getnForAngle((double)i));
		}
	}
	
	/**
	 * This method returns the strain of the current object.
	 * 
	 * @return	The strain
	 */
	public double getStrain()
	{
		return m_dblStrain;
	}
	
	/**
	 * This function is for testing purposes only.
	 * 
	 * @param args
	 */
	public static void main(String[] args)
	{
		double dblnGroupIndex = RefractiveIndexHelper.getGroupIndexForWavelength(Material.PMMA, 650E-9);
		GlobalModelSettings.getInstance().setCoreGroupIndex(dblnGroupIndex);
		StrainIndex si = getStrainIndex(0.01);
		si.printn();
	}
}
