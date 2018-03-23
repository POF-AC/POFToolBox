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
package model.threeDimensional.Discrete;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import model.AngleHelper;
import model.GlobalModelSettings;
import model.TransferAngle;

/**
 * This class represents a two dimensional power distribution over an angle and is mainly used to model 
 * the power distribution that is caused by scattering and attenuation for a discrete angle.
 * 
 * @author Thomas Becker
 *
 */
public class RadialPower 
{
	/**
	 * The number of the angular steps
	 */
	protected int m_nSteps;
	
	/**
	 * This variable states if the power distribution has been normalized
	 */
	protected boolean m_bNormalized = false;
	
	/**
	 * The data array holding the power distribution
	 */
	protected double[] m_dblData;
		
	/**
	 * This string contains the theta phi range for which this power distribution is valid
	 */
	protected String m_strThetaPhiRange = null;
	
	/**
	 * The lower theta phi boundary
	 */
	protected double m_dblThetaPhiMin;
	
	/**
	 * The maximum theta phi boudary
	 */
	protected double m_dblThetaPhiMax;
	
	/**
	 * Theta z outside the fiber for which this power distribution is valid
	 */
	protected double m_dblThetaZOutside;
	
	/**
	 * Theta z inside the fiber for which this power distribution is valid
	 */
	protected double m_dblThetaZInside;
	
	/**
	 * This function sets intervals with negative powers to 0
	 */
	public void deleteNegIntervals() 
	{
		for(int i = 0; i < this.m_nSteps; i++)
		{
			if(0 > m_dblData[i])
				m_dblData[i] = 0.0;
		}
	}

	/**
	 * This method reduces noise by suppressing powers that are further away from the exciting angle than 6° (inner angle).
	 */
	public void reduceNoise()
	{
		double dblAllowedDeviation = 6.0;
		
		// determine the allowed min and max indices
		int nMin = AngleHelper.getInstance().getIndexForInnerAngle(m_dblThetaZInside - 6.0);
		
		if(Integer.MIN_VALUE == nMin)
		{
			nMin = 0;
		}
		
		int nMax = AngleHelper.getInstance().getIndexForInnerAngle(m_dblThetaZInside + 6.0);
		
		if(Integer.MIN_VALUE == nMax || -1 == nMax)
		{
			nMax = m_nSteps-1;
		}
		
		for(int i = 0; i < m_nSteps; i++)
		{
			if(i < nMin || i > nMax)
			{
				m_dblData[i] = 0.0;
			}
		}
	}
	
	/**
	 * This function smoothes the scattering distribution represented by an object of this class with the given number of steps.
	 * 
	 * @param a_nSteps	Number of smoothing steps.
	 */
	public void fixDistribution(int a_nSteps) 
	{
		double dblValuesOriginal[] = m_dblData.clone();
		double dblHistory[] = new double[a_nSteps];
		
		for(int i = 0; i<dblHistory.length; i++){dblHistory[i] = 0.0;}
		
		for(int nIndexAngle = 0; nIndexAngle < m_dblData.length; nIndexAngle++)
		{
			//fillmode average
			for(int i = 0; i<dblHistory.length; i++)
			{
				int index = nIndexAngle+i-a_nSteps/2;
				
				if(index < 0)
				{
					dblHistory[i] = 0.0;
				}
				else if(index >= dblValuesOriginal.length)
				{
					dblHistory[i] = 0.0;
				}
				else
					dblHistory[i] = dblValuesOriginal[index];
			}

			// sum up values
			m_dblData[nIndexAngle] = 0.0;
			for(int i = 0; i<dblHistory.length; i++)
			{
				m_dblData[nIndexAngle] += dblHistory[i];
			}
			
			if(nIndexAngle < a_nSteps/2)
			{
				m_dblData[nIndexAngle]/= (nIndexAngle+a_nSteps/2);
			}
			else if(nIndexAngle > (m_dblData.length-1)-a_nSteps/2)
			{
				m_dblData[nIndexAngle]/= (m_dblData.length-1)-nIndexAngle+a_nSteps/2+1;
			}
			else
				m_dblData[nIndexAngle]/= a_nSteps;
		}
	}
	
	/**
	 * This function calls the smoothing function multiple times.
	 * 
	 * @param a_nSteps	Number of smoothing steps
	 * @param a_nCycles	Number of smoothing cycles
	 */
	public void fixDistribution(int a_nSteps, int a_nCycles) 
	{
		for(int i = 0; i < a_nCycles; i++)
		{
			fixDistribution(a_nSteps);
		}
	}
	
	/**
	 * This function applies triangular smoothing to the data.
	 * 
	 * @param a_nPoints	Number of smoothing steps
	 * @param nPower	Power of the smoothing function
	 */
	public void applyTriangularSmoothing(int a_nPoints, int nPower)
	{
		int nTemp = a_nPoints/2;
		if(0 == a_nPoints-2*nTemp)
		{
			System.out.println("This smoothing filter works only for an odd amount of steps");
			return;
		}
		
		double dblValuesOriginal[] = m_dblData.clone();
		
		int nWeigh = 0;
		for(int i = 1; i <= nTemp; i++)
		{
			nWeigh += Math.pow(i,nPower);
		}
		nWeigh *= 2;
		nWeigh += Math.pow(nTemp+1,nPower);
		
		for(int j = 0; j < m_dblData.length; j++)
		{
			int nFactor = 1;
			
			// determine the minum and maximum possible indices
			int nStart = j-nTemp;
			if(nStart < 0) 
			{
				nStart = 0;
			}
			
			int nEnd = j-nTemp+a_nPoints;
			if(nEnd >= m_dblData.length)
			{
				nEnd = m_dblData.length;
			}
			
			nTemp = (nEnd-nStart)/2;
			nWeigh = 0;
			for(int i = 1; i <= nTemp; i++)
			{
				nWeigh += Math.pow(i,nPower);
			}
			nWeigh *= 2;
			nWeigh += Math.pow(nTemp+1,nPower);
			
			m_dblData[j] = 0.0;
						
			for(int np = nStart; np < nEnd; np++)
			{
				
				m_dblData[j] += dblValuesOriginal[np]*Math.pow(nFactor,nPower);
					
				if((np-nStart) < nTemp)
				{
					nFactor++;
				}
				else
				{
					nFactor--;
				}
			}
			
			m_dblData[j] /= nWeigh;
			
			if(m_dblData[j] < 0)
			{
				int a = 5+3;
			}
		}
	}
	
	/**
	 * c'tor. 
	 * 		
	 * @param a_nSteps				Number of angular steps
	 * @param a_dblThetaZOutside	Theta z for which this power distribution
	 */
	public RadialPower(int a_nSteps, double a_dblThetaZOutside)
	{
		m_nSteps = a_nSteps;
		
		m_dblData = new double[m_nSteps];
		
		m_dblThetaZOutside = a_dblThetaZOutside;
		m_dblThetaZInside = new TransferAngle(1.0, GlobalModelSettings.getInstance().getCoreIndex()).getAngle2(m_dblThetaZOutside*Math.PI/180.0)*180.0/Math.PI;
	}
	
	/**
	 * c'tor.
	 * 
	 * This constructor creates a new RadialPower by interpolating two existing ones with the given radio.
	 * 
	 * @param r1			The first RadialPower
	 * @param r2			The second RadialPower
	 * @param a_dblRatio12	Ratio between both RadialPowers
	 */
	public RadialPower(RadialPower r1, RadialPower r2, double a_dblRatio12)
	{
		m_nSteps = r1.getSteps();
		m_dblData = new double[m_nSteps];
		
		double dblThetaZOutside1 = r1.getThetaZOutside();
		double dblThetaZInside1 = new TransferAngle(1.0, GlobalModelSettings.getInstance().getCoreIndex()).getAngle2(dblThetaZOutside1*Math.PI/180.0)*180.0/Math.PI;
		
		double dblThetaZOutside2 = r2.getThetaZOutside();
		double dblThetaZInside2 = new TransferAngle(1.0, GlobalModelSettings.getInstance().getCoreIndex()).getAngle2(dblThetaZOutside2*Math.PI/180.0)*180.0/Math.PI;

		m_dblThetaZOutside = dblThetaZOutside1*a_dblRatio12 + dblThetaZOutside2*(1-a_dblRatio12);
		m_dblThetaZInside = new TransferAngle(1.0, GlobalModelSettings.getInstance().getCoreIndex()).getAngle2(m_dblThetaZOutside*Math.PI/180.0)*180.0/Math.PI;
			
		double dblOffSet1 = m_dblThetaZInside - dblThetaZInside1;
		double dblOffSet2 = m_dblThetaZInside - dblThetaZInside2;
		
		for(int i = 0; i < m_nSteps; i++)
		{
			double dblPower1 = r1.PowerForStep(i)*a_dblRatio12;
			double dblAngleTemp = AngleHelper.getInstance().getCenterAngleForIndex(i);
			dblAngleTemp+=dblOffSet1;
			int index1 = AngleHelper.getInstance().getIndexForInnerAngle(dblAngleTemp);
			if(0 <= index1)
				m_dblData[index1] += dblPower1; 
			
			
			double dblPower2 = r2.PowerForStep(i)*(1-a_dblRatio12);
			dblAngleTemp = AngleHelper.getInstance().getCenterAngleForIndex(i);
			dblAngleTemp+=dblOffSet2;
			int index2 = AngleHelper.getInstance().getIndexForInnerAngle(dblAngleTemp);
			if(0 <= index2)
				m_dblData[index2] += dblPower2; 
		}
	}
	
	/**
	 * This function returns theta z outside the fiber.
	 * 
	 * @return Theta z outside
	 */
	public double getThetaZOutside()
	{
		return m_dblThetaZOutside;
	}
	
	/**
	 * This function returns theta z inside the fiber.
	 * 
	 * @return Theta z inside
	 */
	public double getThetaZInside()
	{
		return m_dblThetaZInside;
	}
	
	/**
	 * This function returns the spanned theta phi range as string.
	 * 
	 * @return Theta phi range as string
	 */
	public String getThetaPhiRange()
	{
		return m_strThetaPhiRange;
	}
	
	/**
	 * This method returns the lower theta phi boundary.
	 * 
	 * @return Lower theta phi boundary
	 */
	public double getThetaPhiMin()
	{
		return m_dblThetaPhiMin;
	}
	
	/**
	 * This method returns the upper theta phi boundary.
	 * 
	 * @return Upper theta phi boundary
	 */
	public double getThetaPhiMax()
	{
		return m_dblThetaPhiMax;
	}
	
	/**
	 * This method returns the power density for a specific step.
	 * 
	 * @param Index of the requested step
	 * 
	 * @return Power density of the step
	 */
	private double PowerForStep(int i) 
	{
		return m_dblData[i];
	}
	
	/**
	 * This method returns the full data array.
	 * 
	 * @return Data array of the power distribution
	 */
	public double[] getData()
	{
		return m_dblData;
	}

	/**
	 * This method returns the amount of steps.
	 * 
	 * @return Amount of steps
	 */
	public int getSteps() 
	{
		return m_nSteps;
	}

	/**
	 * This c'tor constructs an instance of this class by reading the necessary data from a file.
	 * 
	 * @param a_strFileIn	Full qualified path to the file
	 * @param a_nSteps		Number of steps
	 * @param a_dblThetaZ	Theta z for this power distribution
	 */
	public RadialPower(String a_strFileIn, int a_nSteps, double a_dblThetaZ)
	{
		this(a_strFileIn, a_nSteps, false, a_dblThetaZ);
	}

	/**
	 * This c'tor constructs an instance of this class by reading the necessary data from a file.
	 * 
	 * @param a_strFileIn	Full qualified path to the file
	 * @param a_nSteps		Number of steps
	 * @param a_bSetThetaPhiRange	If set to <code>true</code> the theta phi range is read from the file
	 * @param a_dblThetaZ	Theta z for this power distribution
	 */
	public RadialPower(String a_strFileIn, int a_nSteps, boolean a_bSetThetaPhiRange, double a_dblThetaZOutside) 
	{
		m_bNormalized = true;
		m_nSteps = a_nSteps;
		m_dblData = new double[m_nSteps];
		
		m_dblThetaZOutside = a_dblThetaZOutside;
		m_dblThetaZInside = new TransferAngle(1.0, GlobalModelSettings.getInstance().getCoreIndex()).getAngle2(m_dblThetaZOutside*Math.PI/180.0)*180.0/Math.PI;
		
		FileReader fr;
		try 
		{
			fr = new FileReader(a_strFileIn);
			BufferedReader br = new BufferedReader(fr);
			
			String strLine = null;
			int nCounter = 0;
			while(null != (strLine = br.readLine()))
			{
				nCounter++;
				
				switch(nCounter)
				{
				case 1:
					if(a_bSetThetaPhiRange)
					{
						m_strThetaPhiRange = strLine;
						String[] stra = m_strThetaPhiRange.substring(1).split(" - ");
						m_dblThetaPhiMax = Double.parseDouble(stra[0]);
						m_dblThetaPhiMin = Double.parseDouble(stra[1]);
						m_strThetaPhiRange = Math.round(100.0*m_dblThetaPhiMin)/100.0 + "-" + Math.round(100.0*m_dblThetaPhiMax)/100.0; 
					}
					continue;
				default:
					break;
				}
		
				String[] strA = strLine.split(" ");
				m_dblData[nCounter-4] = Double.parseDouble(strA[1]);
			}
			
			br.close();
			fr.close();

		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}

	/**
	 * This method adds the given power to the power density of the given angle.
	 * 
	 * @param a_dblAngle	Angle for which the power has to be added
	 * @param a_dblPower	Power that has to be added
	 */
	public void addPowerForAngle(double a_dblAngle, double a_dblPower)
	{
		int nIndex = AngleHelper.getInstance().getIndexForInnerAngle(a_dblAngle);
		
		if(-1 == nIndex)
		{
			System.out.println("Invalid index: " + nIndex);
		}
		else if(nIndex == Integer.MIN_VALUE)
		{
			if(a_dblPower != 0)
			{
				System.err.println("Index out of Bounds: " + nIndex + " " + a_dblPower);
			}
		}
		else
		{
			m_dblData[nIndex] += a_dblPower/AngleHelper.getInstance().getStepWidthForStep(nIndex);	
		}
	}
	
	/**
	 * This method prints the Number of steps and the total power of this object to the command line.
	 */
	public void printData()
	{
		System.out.println("Steps:" + m_nSteps);
		System.out.println("TP:" + getTotalPower()+ "mW");
		
	}
	
	/**
	 * This function calculates and returns the total power held by this power distribution.
	 * 
	 * @return Total power
	 */
	public double getTotalPower()
	{
		double dblPower = 0.0;
		
		for(int i = 0; i < m_nSteps; i++)
		{
			dblPower += m_dblData[i]*AngleHelper.getInstance().getStepWidthForStep(i);
		}

		return dblPower;
	}
	
	/**
	 * This function writes the power distribution to an output file.
	 * 
	 * @param a_strFileOut	Full qualified path to the output file
	 * @param a_bUsePowerInsteadOfThetaPhi	If <code>true</code> we write the total power to the file, otherwise the theta phi range
	 * 
	 * @throws Exception	Exception that might occur while writing to the output file
	 */
	public void writeToFile(String a_strFileOut, boolean a_bUsePowerInsteadOfThetaPhi) throws Exception
	{
		FileWriter fw = new FileWriter(a_strFileOut);
		BufferedWriter bw = new BufferedWriter(fw);
		
		String strExtension = m_bNormalized ? " normalized\n" : "mW\n";
		
		if(a_bUsePowerInsteadOfThetaPhi)
		{
			bw.write("#TP: " + this.getTotalPower() + strExtension);
		}
		else
		{
			bw.write("#" + m_dblThetaPhiMax + " - " + m_dblThetaPhiMin+ "\n");
		}
		bw.write("#TZ Outside: " + m_dblThetaZOutside + "\n");
		bw.write("#TZ Inside: " + m_dblThetaZInside + "\n");
		
		for(int i = 0; i < m_nSteps; i++)
		{
			bw.write(AngleHelper.getInstance().getCenterAngleForIndex(i) + " " + m_dblData[i] +"\n");
		}
		
		bw.close();
		fw.close();
	}

	/**
	 * This method normalizes the power distribution so that the total power is 1.
	 * 
	 * @return The factor that had to be applied to correct the intervals.
	 */
	public double normalize() 
	{
		double dblPower = this.getTotalPower();
		double dblFactorCorrection = 1/dblPower;
		m_bNormalized = true;
		
		for(int i = 0; i < m_nSteps; i++)
		{
			m_dblData[i] *= dblFactorCorrection;
		}
		
		return dblFactorCorrection;
	}
	
	/**
	 *	This function normalizes the data array by the given factor.
	 *
	 * @param a_dblFactorCorrection	Factor by which the data should be normalized
	 */
	public void normalizeRest(double a_dblFactorCorrection) 
	{
		m_bNormalized = true;
		
		for(int i = 0; i < m_nSteps; i++)
		{
			m_dblData[i] *= a_dblFactorCorrection;
		}
	}
}



