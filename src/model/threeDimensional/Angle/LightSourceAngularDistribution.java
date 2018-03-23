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
import model.GlobalModelSettings;

/**
 * This class describes different angular power distributions over theta z to model different light sources and receiver characteristics.
 * 
 * @author Thomas Becker
 *
 */
public class LightSourceAngularDistribution 
{
	/**
	 * This enum defines different light sources and receiver characteristics
	 * 
	 * @author Thomas Becker
	 *
	 */
	public enum LightSourceAngularEnum
	{
		/**
		 * Constant over the solid angle, relation between regular and solid angle considered
		 */
		EQUAL_OVER_SOLID_ANGLE,
		
		/**
		 * Power distribution of the Y coupler, relation between regular and solid angle considered
		 */
		Y,
		
		/**
		 * Power distribution of the Y coupler, relation between regular and solid angle neglected
		 */
		Y_PURE,
		
		/**
		 * Angular sensitivity of the Hamamatsu S5052
		 */
		S5052_PURE,
		
		/**
		 * Angular sensitivity of the Vishay BPW34
		 */
		BPW34_PURE
	}
	
	/**
	 * This type of the current instance
	 */
	private LightSourceAngularEnum m_mode;
	
	/**
	 * Fiber properties to obtain the refractive indices
	 */
	private FiberProps m_props;
	
	/**
	 * Power of the light source in the half space 2 Pi
	 */
	private double m_dblPT;
	
	/**
	 * This method is for debugging purposes only.
	 * 
	 * @param cargs Command line arguments
	 */
	public static void main(String[] cargs)
	{
		GlobalModelSettings.getInstance().setCoreIndex(1.49);
		FiberProps fp = new FiberProps(GlobalModelSettings.getInstance().getCoreIndex(), 1.42, 1.0, 10, 0.5E-3);
		LightSourceAngularDistribution lsad = new LightSourceAngularDistribution(LightSourceAngularEnum.Y_PURE, fp,1.0);

		for(double i = 0; i < 90; i+=1)
		{
			System.out.println(i + " " + lsad.getValue(i*Math.PI/180));
		}
	}
	
	/**
	 * c'tor. This constructor instantiates an object of the class according to the parameters.
	 * 	
	 * @param a_mode	Type of the light source (see <code>LightSourceAngularEnum</code>)
	 * @param a_props	Fiber properties
	 * @param a_dblPT	Power of the light source over the half space 2 Pi
	 */
	public LightSourceAngularDistribution(LightSourceAngularEnum a_mode, FiberProps a_props, double a_dblPT)
	{
		m_mode = a_mode;
		m_props = a_props;
		m_dblPT = a_dblPT;
	}
	
	/**
	 * This method returns the power density / sensitivity at the given angle
	 * 
	 * @param a_dblTZ	Theta z
	 * @return			Power density / sensitivity
	 */
	public double getValue(double a_dblTZ)
	{
		double dblReturn = 0.0;
		
		switch(m_mode)
		{
			case EQUAL_OVER_SOLID_ANGLE:
			{
				// sin outside transferred to inner angle
				double dblTempAngle = Math.asin((m_props.m_dblIndexCore/m_props.m_dblIndexSurround)*Math.sin(a_dblTZ));
				dblReturn = Math.sin(dblTempAngle)*m_dblPT;
				break;
			}
			case S5052_PURE:
			{
				double dblTempAngle = Math.asin((m_props.m_dblIndexCore/m_props.m_dblIndexSurround)*Math.sin(a_dblTZ));
				dblReturn = S5052Coefficients.getValue(dblTempAngle, m_dblPT, m_props);
				break;
			}
			case BPW34_PURE:
			{
				double dblTempAngle = Math.asin((m_props.m_dblIndexCore/m_props.m_dblIndexSurround)*Math.sin(a_dblTZ));
				dblReturn = BPW34Coefficients.getValue(dblTempAngle, m_dblPT, m_props);
				break;
			}
			case Y:
			{
				double dblTempAngle = Math.asin((m_props.m_dblIndexCore/m_props.m_dblIndexSurround)*Math.sin(a_dblTZ));
				dblReturn = Math.sin(dblTempAngle)*m_dblPT;
	
				dblReturn *= YCoefficients.getValue(dblTempAngle, m_dblPT, m_props);
				break;
			}
			case Y_PURE:
			{
				double dblTempAngle = Math.asin((m_props.m_dblIndexCore/m_props.m_dblIndexSurround)*Math.sin(a_dblTZ));
				dblReturn = YCoefficients.getValue(dblTempAngle, m_dblPT, m_props);
				break;
			}
		}

		return dblReturn;
	}
	
	/**
	 * This class models the S5052.
	 * 
	 * @author Thomas Becker
	 *
	 */
	static private class S5052Coefficients
	{
		/**
		 * Coefficient of the third order
		 */
		static double m_dblap = -1.66667e-05*Math.pow((180.0/Math.PI),3);
		
		/**
		 * Coefficient of the second order
		 */
		static double m_dblbp = -0.0006*Math.pow((180.0/Math.PI),2);
		
		/**
		 * Coefficient of the first order
		 */
		static double m_dblcp = -0.000333333*Math.pow((180.0/Math.PI),1);
		
		/**
		 * Coefficient of the zeroth order
		 */
		static double m_dbldp = 1;
		
		/**
		 * This function evaluates the function at the given outer angle
		 * 
		 * @param a_dblTempAngle	Outer angle
		 * @param a_dblPT			Power of the light source over the half space 2 Pi
		 * @param a_props			Fiber properties
		 * @return					The result of the function
		 */
		static public double getValue(double a_dblTempAngle, double a_dblPT, FiberProps a_props)
		{
			double dblReturn;
			if(a_dblTempAngle >= 30.0*Math.PI/180.0 || Double.isNaN(a_dblTempAngle))
			{
				dblReturn = 0.0;
			}
			else
			{
				dblReturn = (m_dblap*Math.pow(a_dblTempAngle,3)+m_dblbp*Math.pow(a_dblTempAngle,2)+m_dblcp*a_dblTempAngle+m_dbldp);
			}
			
			return dblReturn;
		}
	}
	
	/**
	 * This class models the BPW34.
	 * 
	 * @author Thomas Becker
	 *
	 */
	static private class BPW34Coefficients
	{
		/**
		 * Coefficient of the second order
		 */
		static double m_dblap = -0.000152036*Math.pow((180.0/Math.PI),2);
		
		/**
		 * Coefficient of the first order
		 */
		static double m_dblbp = 0.00243934*(180.0/Math.PI);
		
		/**
		 * Coefficient of the zeroth order
		 */
		static double m_dblcp = 0.9997;	
		
		/**
		 * This function evaluates the function at the given outer angle
		 * 
		 * @param a_dblTempAngle	Outer angle
		 * @param a_dblPT			Power of the light source over the half space 2 Pi
		 * @param a_props			Fiber properties
		 * @return					The result of the function
		 */
		static public double getValue(double dblTempAngle, double a_dblPT, FiberProps a_props)
		{
			double dblReturn;
			if(dblTempAngle >= 90.0*Math.PI/180.0 || Double.isNaN(dblTempAngle))
			{
				dblReturn = 0.0;
			}
			else
			{
				dblReturn = (m_dblap*Math.pow(dblTempAngle,2)+m_dblbp*Math.pow(dblTempAngle,1)+m_dblcp);
			}
			
			return dblReturn;
		}
	}
	
	/**
	 * This class models the power distribution at the end of the Y-coupler.
	 * 
	 * @author Thomas Becker
	 *
	 */
	static private class YCoefficients
	{
		/**
		 * Coefficient mu
		 */
		static double m_dblmu = -0.6;
		
		/**
		 * Coefficient si
		 */
		static double m_dblsi = 130;
		
		/**
		 * This function evaluates the function at the given outer angle
		 * 
		 * @param a_dblTempAngle	Outer angle
		 * @param a_dblPT			Power of the light source over the half space 2 Pi
		 * @param a_props			Fiber properties
		 * @return					The result of the function
		 */
		static public double getValue(double dblTempAngle, double a_dblPT, FiberProps a_props)
		{
			double dblAngleInternal = 180.0*dblTempAngle/Math.PI;
			
			double dblReturn;
			if(dblTempAngle > 35.0*Math.PI/180.0 || Double.isNaN(dblTempAngle))
			{
				dblReturn = 0.0;
			}
			else
			{
				dblReturn = Math.exp(m_dblmu*Math.pow(dblAngleInternal, 2)/m_dblsi); 
			}
			
			return dblReturn;
		}
	}
}
