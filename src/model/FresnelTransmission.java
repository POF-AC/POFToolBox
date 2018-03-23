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
 * This class provides functions to calculate Fresnel losses and Fresnel reflection between two different materials.
 * 
 * @author Thomas Becker
 *
 */
public class FresnelTransmission 
{
	/**
	 * TransferAngle to calculate the angle conversion
	 */
	protected TransferAngle ta;
	
	/**
	 * If set to <code>true</code>, the provided angles are treated as angles inside the second material, otherwise as angles inside the first
	 */
	boolean m_bUseAngle2 = false;
	
	/**
	 * This function is for testing purposes only
	 * 
	 * @param args	Commandline arguments
	 */
	public static void main(String[] args)
	{
		FresnelTransmission FT = new FresnelTransmission(1.0, 3.8, false);
		
		for( int i = 0; i < 90; i++)
		{
			double dblAngleResult = FT.getTransmissionForAngle(((double)i)*Math.PI/180.0);
			System.out.println(""+i+": " + dblAngleResult);
		}
	}
	
	/**
	 * c'tor.
	 * 
	 * @param a_dblIndex1	Refractive index of the first material
	 * @param a_dblIndex2	Refractive index of the second material
	 * @param a_bUseAngle2	If set to <code>true</code>, the provided angles are treated as angles inside the second material, otherwise as angles inside the first
	 */
	public FresnelTransmission(double a_dblIndex1, double a_dblIndex2, boolean a_bUseAngle2)
	{
		ta = new TransferAngle(a_dblIndex1, a_dblIndex2);
		m_bUseAngle2 = a_bUseAngle2;
	}
	
	/**
	 * inner transfers inner angle to outer angle
	 */
    public double getTransmissionForAngle(double a_dblThetaz) 
    {
    	double dblThetaz = a_dblThetaz;
    	if(m_bUseAngle2)
    	{
    		dblThetaz = this.ta.getAngle1(a_dblThetaz);
    	}
    	
    	double dblTrans = 
    	(
    		(
    			(
    				1
    				-
    				Math.pow((
    					(
    						(
    							ta.m_dblIndex1
    							*
    							Math.cos
    							(
    								dblThetaz
    							)
    							-
    							ta.m_dblIndex2
    							*
    							Math.sqrt
    							(
    								1
    								- 
    								Math.pow((
    									(
    										ta.m_dblIndex1
    										/
    										ta.m_dblIndex2
    									)
    									*
    									Math.sin
    									(
    										dblThetaz
    									)
    								),2)
    							)
    						)
    						/
    						(
    							ta.m_dblIndex1
    							*
    							Math.cos
    							(
    								dblThetaz
								)
								+
								ta.m_dblIndex2
								*
								Math.sqrt
								(
									1
									-
									Math.pow((
										(
											ta.m_dblIndex1
											/
											ta.m_dblIndex2
										)
										*
										Math.sin
										(
											dblThetaz
										)
									),2)
								)
							)
						)
					),2)
				) 
				+
				(
					1
					-
					Math.pow((
						(
							(
								ta.m_dblIndex2
								*
								Math.cos
								(
									dblThetaz
								)
								-
								ta.m_dblIndex1
								*
								Math.sqrt
								(
									1 
									- 
									Math.pow((
										(
											ta.m_dblIndex1
											/
											ta.m_dblIndex2
										)
										*
										Math.sin
										(
											dblThetaz
										)
									),2)
								)
							) 
							/
							(
								ta.m_dblIndex2
								*
								Math.cos
								(
									dblThetaz
								)
								+
								ta.m_dblIndex1
								*
								Math.sqrt
								(
									1 
									-
									Math.pow((
										(
											ta.m_dblIndex1
											/
											ta.m_dblIndex2
										)
										*
										Math.sin
										(
											dblThetaz
										)
									),2)
								)
							)
						)
					),2)
				)
			)
			/
			2
		);
    	
    	if(Double.isNaN(dblTrans))
    		dblTrans = 0.0;
    	
        return dblTrans;
    }
}
