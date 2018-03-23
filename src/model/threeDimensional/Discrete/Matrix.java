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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Vector;

import model.AngleHelper;
import model.FresnelTransmission;
import model.StrainIndex;
import model.TransferAngle;
import model.threeDimensional.Function3D;
import model.threeDimensional.Angle.LightSourceAngularDistribution;

/**
 * This class hold a two-dimensional matrix of MatrixCells and represents the state of the 
 * power distribution over angles and time at a specific point in the fiber.
 * 
 * @author Thomas Becker
 *
 */
public class Matrix 
{
	/**
	 * The main Matrix. First index is Theta_z, second Theta_phi.
	 */
	private MatrixCell[][] m_matrix;
	
	/**
	 * Strain, if applied to fiber.
	 */
	private double m_dblStrain;
	
	/**
	 * The maximum index of the matrix in Theta_phi direction.
	 */
	private int m_nTPMaxIndex;
	
	/**
	 * The maximum value for Theta_phi.
	 */
	private double m_dblTPMax;
	
	/**
	 * The minimum value for Theta_phi.
	 */
	private double m_dblTPMin;
	
	/**
	 * The maximum index of the matrix in Theta_z direction.
	 */
	private int m_nTZMaxIndex;
	
	/**
	 * The maximum value for Theta_z (initialized during the construction of the matrix).
	 */
	private double m_dblTZMax = Double.NaN;
	
	/**
	 * The absolute positon of the matrix in the fiber. This distance includes strained parts.
	 */
	private double m_dblDistanceAbsolute;
	
	/**
	 * The relative distance between two matrices. This distance excludes strain.
	 */
	private double m_dblDistanceDelta;
	
	/**
	 * The refractive index of the core.
	 */
	private double m_dblCoreIndex;
	
	/**
	 * Helper for transferring angles between inside and outside of the fiber.
	 */
	private TransferAngle m_ta;
	
	
	/**
	 * Constructor. This Constructor is used for the following matrices when the Theta Phi range is already known.
	 * 
	 * @param a_pTPMaxIndex			Max index for Theta Phi
	 * @param a_pdblTPMax			Max angle for Theta Phi
	 * @param a_pdblTPMin			Min angle for Theta Phi
	 * @param a_pnTZMaxIndex		Max index for Theta z
	 * @param a_pTZMax				Max angle for Theta z
	 * @param a_pDistanceAbsolute	Absolute position of the Matrix in the fiber
	 * @param a_pDistanceDelta		Distance between two matrices
	 * @param a_pCoreIndex			Refractive index of the core
	 * @param a_pSurroundIndex		Refractive index of the surrounding material
	 * @param a_dblStrain			Applied strain (= 1.0 if unstrained)
	 */
	public Matrix(
			int a_nTPMaxIndex, 
			double a_dblTPMax,
			double a_dblTPMin,
			int a_nTZMaxIndex,
			double a_dblTZMax,
			double a_dblDistanceAbsolute,
			double a_dblDistanceDelta,
			double a_nCoreIndex,
			double a_nSurroundIndex,
			double a_dblStrain)
	{
		this(a_nTZMaxIndex+1, a_nTPMaxIndex+1, a_dblDistanceAbsolute, a_dblDistanceDelta, a_nSurroundIndex, a_nCoreIndex, a_dblTZMax, a_dblStrain);

		m_dblTPMax = a_dblTPMax;
		m_dblTPMin = a_dblTPMin;
		
	}
	
	/**
	 * Constructor. This is Constructor is usually used for the first matrix, where the Theta Phi range is being read from the scatter files.
	 * 
	 * @param a_nTZCells			Number of Cells for Theta z
	 * @param a_nTPCells			Number of Cells for Theta Phi
	 * @param a_dblDistanceAbsolute	Absolute position of the Matrix in the fiber
	 * @param a_dblDistanceDelta	Distance between two matrices
	 * @param a_dblnOutside			Refractive index of the surrounding material
	 * @param a_dblnInside			Refractive index of the core
	 * @param a_dblTZMax			Max angle for Theta z
	 * @param a_dblStrain			Applied strain (= 1.0 if unstrained)
	 * */
	public Matrix(int a_nTZCells, int a_nTPCells, double a_dblDistanceAbsolute, double a_dblDistanceDelta, double a_dblnOutside, double a_dblnInside, 
			double a_dblTZMax, double a_dblStrain)
	{
		
		m_dblDistanceAbsolute = a_dblDistanceAbsolute;
		m_dblDistanceDelta = a_dblDistanceDelta;
		
		m_nTZMaxIndex = a_nTZCells - 1;
		m_nTPMaxIndex = a_nTPCells - 1;
			
		m_dblTZMax = a_dblTZMax;
		
		m_matrix = new MatrixCell[a_nTZCells][a_nTPCells];
		
		m_ta = new TransferAngle(a_dblnOutside, a_dblnInside);
		m_dblCoreIndex = a_dblnInside;
		m_dblStrain = a_dblStrain;
	}
	
	/**
	 * Clones the existing matrix.
	 *  
	 * This method is mainly used for Cutback simulations, where receiver characteristics or Fresnel losses have to be applied to a matrix
	 * that does not represent the fiber's end and therefore needs to be saved for further simulations. 
	 */
	public Matrix clone()
	{
		Matrix mn = new Matrix(m_nTPMaxIndex, m_dblTPMax, m_dblTPMin, m_nTZMaxIndex,
				m_dblTZMax, m_dblDistanceAbsolute, m_dblDistanceDelta, m_dblCoreIndex, m_ta.getIndex1(), m_dblStrain);
		
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				mn.m_matrix[nTZ][nTP] = m_matrix[nTZ][nTP].clone();
			}
		}
		
		return mn;
	}
	
	/**
	 * Sets the relative distance between two matrices.
	 * 
	 * @param a_dblDistance		Distance to set
	 */
	public void setRelativeDistance(double a_dblDistance)
	{
		m_dblDistanceDelta = a_dblDistance;
	}
	
	/**
	 * Returns the the refractive index of the core.
	 * 
	 * This method considers the influence of strain conditions and is therefore angle-dependent.
	 *  
	 * @param a_dblAngle desired angle Theta z 
	 * 
	 * @return Refractive index
	 */
	public double getCoreIndex(double a_dblAngle)
	{
		return StrainIndex.getStrainIndex(m_dblStrain).getnForAngle(a_dblAngle);
	}
	
	/**
	 * Returns the full data matrix.
	 * 
	 * @return Matrix
	 */
	public MatrixCell[][] getCells()
	{
		return m_matrix;
	}
	
	/**
	 * This method is used to load scatter data from a specified folder.
	 * 
	 * It is optimized for 4 Threads.
	 * At the end of the function the maximum value for Theta z and the minimum and maximum value for Theta phi is set.
	 * 
	 * @param a_strFolder Folder where the files are located
	 * @throws InterruptedException
	 */
	public void loadScatterFiles(String a_strFolder) throws InterruptedException
	{
		Thread t1 = new Thread(new ScatterLoadRunnable(0, m_nTZMaxIndex/4, a_strFolder, this));
		Thread t2 = new Thread(new ScatterLoadRunnable(m_nTZMaxIndex/4+1, m_nTZMaxIndex/2, a_strFolder, this));
		Thread t3 = new Thread(new ScatterLoadRunnable(m_nTZMaxIndex/2+1, 3*m_nTZMaxIndex/4, a_strFolder, this));
		Thread t4 = new Thread(new ScatterLoadRunnable(3*m_nTZMaxIndex/4+1, m_nTZMaxIndex, a_strFolder, this));
		
		t1.start();
		t2.start();
		t3.start();
		t4.start();
		
		t1.join();
		t2.join();
		t3.join();
		t4.join();
				
		this.m_dblTZMax = m_matrix[m_nTZMaxIndex][m_nTPMaxIndex].getScatterCell().getThetaZInside();
		this.m_dblTPMax = m_matrix[0][0].getScatterCell().getThetaPhiMax();
		this.m_dblTPMin = m_matrix[0][m_nTPMaxIndex].getScatterCell().getThetaPhiMin();
	}
	
	/**
	 * Subclass for Multithreading purposes.
	 * 
	 * This class loads the scatter files into a single matrix.
	 * 
	 * @author Thomas Becker
	 *
	 */
	public class ScatterLoadRunnable implements Runnable
	{
		/**
		 * The Theta z index range that this instance processes.
		 */
		private int nIndexStart, nIndexEnd;
		
		/**
		 * The folder where the scatter files are stored.
		 */
		private String strFolder = null;
		
		/**
		 * Reference to the corresponding matrix so we can insert the ScatterCell instance.
		 */
		private Matrix matrix;
		
		/**
		 * 
		 * @param a_nIndexStart		Start index for Theta z range to pe processed
		 * @param a_nIndexEnd		End index for Theta z range to pe processed
		 * @param a_strFolder		Folder where the scatter files are stored
		 * @param a_matrix			Reference to the matrix that has to be filled
		 */
		public ScatterLoadRunnable(int a_nIndexStart, int a_nIndexEnd, String a_strFolder, Matrix a_matrix)
		{
			nIndexStart = a_nIndexStart;
			nIndexEnd = a_nIndexEnd;
			strFolder = a_strFolder;
			matrix = a_matrix;
		}

		/**
		 * Threadable function that loads scatter cells for the given range.
		 */
		@Override
		public void run() 
		{
			for(int nTZ = nIndexStart; nTZ <= nIndexEnd; nTZ++)
			{
				for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
				{
					m_matrix[nTZ][nTP] = new MatrixCell(strFolder + "/" + getFilenameForAngles(nTZ, nTP), 851, getTZ0ForTZIndex(nTZ), matrix, nTZ, nTP);			
				}
			}
		}
	}
	
	/**
	 * Gets the maximum angle for theta z.
	 *  
	 * @return		Theta z max
	 */
	public double getTZMax()
	{
		return m_dblTZMax;
	}
	
	/**
	 * Gets the minimum angle for theta phi.
	 *  
	 * @return		Theta phi min
	 */
	public double getTPMin()
	{
		return m_dblTPMin;
	}
	
	/**
	 * Gets the maximum angle for theta phi.
	 *  
	 * @return		Theta phi max
	 */
	public double getTPMax()
	{
		return m_dblTPMax;
	}
	
	/**
	 * Gets the number of steps for theta z. 
	 * 
	 * @return Number of steps
	 */
	public int getTZSteps()
	{
		return this.m_nTZMaxIndex+1;
	}
	
	/**
	 * Gets the number of steps for theta phi. 
	 * 
	 * @return Number of steps
	 */	
	public int getTPSteps()
	{
		return this.m_nTPMaxIndex+1;
	}
	
	/**
	 * This function initializes the cell impulse responses of each Matrix Cell of the first matrix with the power distribution 
	 * described by the supplied function.
	 * 
	 * @param a_function		Supplied power distribution over theta z and theta phi
	 * @param a_bForceTP		For debugging purposes. Allows to reduce the theta phi range to a_nTPIndex
	 * @param a_nTPIndex		The theta phi index that the range is limited to if a_bForceTP is set
	 * @param a_bUseIntegral	if <code>true</code>, the provided function must be integrated over theta phi
	 */
	public void loadInitialPowerDistribution(Function3D a_function, boolean a_bForceTP, int a_nTPIndex, boolean a_bUseIntegral)
	{
		// init the power distribution of the matrix from the 2dimensional powerdistribution
		int nPointsX = m_matrix.length;
		int nPointsY = m_matrix[0].length;
   
        double[][] dblMatrix = new double[nPointsX][nPointsY];
        
        int jmin = 0;
        int jmax = nPointsY;
        
        int imin = 0;
        int imax = nPointsX;
        
        // limit theta phi range for debugging purposes
        if(a_bForceTP)
        {
        	jmin = a_nTPIndex;
        	jmax = a_nTPIndex + 1;
        }
           
        for(int i = imin; i < imax; i++)
        {
        	for(int j = jmin; j < jmax; j++)
        	{
           		double dblTZ = this.m_matrix[i][j].getScatterCell().getThetaZInside();
        		double dblTPMin = this.m_matrix[i][j].getScatterCell().getThetaPhiMin();
        		double dblTPMax = this.m_matrix[i][j].getScatterCell().getThetaPhiMax();
        		
        		if(a_bUseIntegral)
        		{
        			// function is integrated over theta phi
	        		dblMatrix[i][j] = 
					(180.0/Math.PI)
					*
					(
							(a_function.getValue(dblTZ*Math.PI/180.0, (dblTPMax)*Math.PI/180.0))
							-
							(a_function.getValue(dblTZ*Math.PI/180.0, (dblTPMin)*Math.PI/180.0))
					)
					/
					(dblTPMax-dblTPMin);
        		}
        		else
        		{
        			// provided function is a regular power distribution
        			dblMatrix[i][j] = a_function.getValue(dblTZ*Math.PI/180.0, (dblTPMax+dblTPMin)*0.5*Math.PI/180.0);
        		}

        		
        		// Power peak is power and not a power density
        		// => multiply it with both angle-ranges
        		// Important: Theta Phi Slices are not aquidistant
        		dblMatrix[i][j] *= (dblTPMax-dblTPMin)*Math.PI/180.0;
        		dblMatrix[i][j] *= (AngleHelper.getInstance().getStepWidthForStep(i))*Math.PI/180.0;
        	}
        }

        // add a power peak for each cell to generate the initial power distribution 
        for(int i = 0; i < nPointsX; i++)
        {
        	for(int j = 0; j < nPointsY; j++)
        	{
        		m_matrix[i][j].getDIR().addPowerPeak(0.0, dblMatrix[i][j]);
        	}
        }
	}
	
	/**
	 * This function derives the filename of a scatterfile that holds the data for a given angle combination.
	 * 
	 * @param a_nTZIndex	Index for theta z
	 * @param a_nTPIndex	Index for theta phi
	 * 
	 * @return	Filename
	 */
	public String getFilenameForAngles(int a_nTZIndex, int a_nTPIndex)
	{
		String strFilename = "";
		
		// the maximum angle for theta z is 85 degree and we have m_nTZMaxIndex steps
		double dblR = (double)a_nTZIndex*(85.0)/(m_nTZMaxIndex);
		DecimalFormat dr = new DecimalFormat("##.#");
		String strdr = dr.format(dblR);
		
		strFilename = "R=" + strdr + ",";
		
		// the maximum blade position is 0.485 
		double dblX1 = a_nTPIndex*0.49/(m_nTPMaxIndex+1);
		DecimalFormat dX1 = new DecimalFormat("#.###");
		String strX1 = dX1.format(dblX1);
		
		double dblX2 = (a_nTPIndex+1)*0.49/(m_nTPMaxIndex+1);
		DecimalFormat dX2 = new DecimalFormat("#.###");
		String strX2 = dX2.format(dblX2);
		strFilename +="X="+strX1+"-"+strX2 + ".txt" ;
		
		return strFilename;
	}
	
	/**
	 * This function returns the angle inside the fiber corresponding to the given theta z index.
	 * 
	 * @param a_nTZIndex	Index for theta z
	 * @return	Internal angle for theta z
	 */
	public double getTZForTZIndex(int a_nTZIndex)
	{
		double dblTZOutside = (double)a_nTZIndex*(85.0)/(m_nTZMaxIndex);
		double dblTZInside = this.m_ta.getAngle2(dblTZOutside*Math.PI/180.0)*180.0/Math.PI;
		return dblTZInside;
	}
	
	/**
	 * This function returns the angle outside the fiber corresponding to the given theta z index.
	 * 
	 * @param a_nTZIndex	Index for theta z
	 * @return	External angle for theta z
	 */
	public double getTZ0ForTZIndex(int a_nTZIndex)
	{
		double dblTZOutside = (double)a_nTZIndex*(85.0)/(m_nTZMaxIndex);
		return dblTZOutside;
	}
	
	/**
	 * This function returns the index for a given angle theta z inside the fiber.
	 * 
	 * @param a_dblTZ	Internal angle for theta z
	 * @return	Index for theta z
	 */
	public int getIndexForTZ(double a_dblTZ)
	{
		int nIndex = -1;
		
		int nTempIndex1 = -1;
		int nTempIndex2 = -1;
		
		// look for the first cell with a greate angle than a_dblTZ
		for(int i = 0; i <= this.m_nTZMaxIndex; i++)
		{
			if(m_matrix[i][0].getScatterCell().getThetaZInside() > a_dblTZ)
			{
				nTempIndex2 = i;
				break;
			}
		}
		
		// if we have not found such a cell, the requested angle is invalid
		if(nTempIndex2 != -1)
		{
			
			if(nTempIndex2 > 0)
				nTempIndex1 = nTempIndex2-1;
			else
				nTempIndex1 = nTempIndex2;
			
			// return the cell with the angle closest to the requested one
			if(Math.abs(m_matrix[nTempIndex1][0].getScatterCell().getThetaZInside() - a_dblTZ) < Math.abs(m_matrix[nTempIndex2][0].getScatterCell().getThetaZInside() - a_dblTZ))
			{
				nIndex = nTempIndex1;
			}
			else
			{
				nIndex = nTempIndex2;
			}
		}
		else 
			nIndex = -1;
		
		return nIndex;
	}
	
		
	/**
	 * Spread power from the first matrix to a new one.
	 * 
	 * This is the function that initiates the full process of transforming the time and angular power distribution from one matrix to the next.
	 *
	 * @param a_bSuppressSA				If <code>true</code> scattering and attenuation are neglected and only modal dispersion is considered
	 * @param a_bCreateImpulseResponse	If <code>true</code> each cell of the new matrix is initialized with a prepared impulse response. Otherwise the new matrix works with power peaks
	 * @param a_nNumberOfTimeSlices		If a_bCreateImpulsResponse is set to <code>true</code>, the number of time slices for the new impulse response can be given here 
	 * @param a_nThreads				Number of threads that should do re work 
	 * @param a_bUseHyperSpace			Use Hyperspace for the impulse responses of the new matrix			
	 * @param a_dblNewStrain			Strain factor that is to be applied to the new matrix 
	 * 
	 * @return The new Matrix
	 * @throws InterruptedException
	 */
	public Matrix MatrixReloaded(
			boolean a_bSuppressSA, 
			boolean a_bCreateImpulseResponse, 
			int a_nNumberOfTimeSlices, 
			int a_nThreads, 
			boolean a_bUseHyperSpace, 
			double a_dblNewStrain) throws InterruptedException
	{
		
		double dblDistanceRelative = m_dblDistanceDelta;
		double dblDistanceAbsolute = m_dblDistanceAbsolute + m_dblDistanceDelta*(1.0+a_dblNewStrain);
		
		// prepare the new matrix
		Matrix newMatrix = new Matrix(m_nTZMaxIndex+1, m_nTPMaxIndex+1, dblDistanceAbsolute, dblDistanceRelative, 
				m_ta.getIndex1(), m_ta.getIndex2(), m_dblTZMax, a_dblNewStrain);
		newMatrix.initFromMatrix(this, a_bCreateImpulseResponse, a_nNumberOfTimeSlices, a_bUseHyperSpace);
		
		int nFields = m_nTZMaxIndex+1;
		Thread[] ta = new Thread[a_nThreads];
		
		// do the real work
		for(int i = 0; i < a_nThreads;i++)
		{
			ta[i] = new Thread(new ReloadRunnable(
					i*nFields/(a_nThreads)
					,(i+1)*nFields/a_nThreads-1
					, newMatrix, a_bSuppressSA)
					);
			
			ta[i].start();
		}
		
		// wait for all threads to finish
		for(int i = 0; i < a_nThreads;i++)
		{
			ta[i].join();
		}
		
		return newMatrix;
	}
	
	/**
	 * This class supports the Runnable Interface and is designed to do the scatter process for a given cell range.
	 * 
	 * @author Thomas Becker
	 *
	 */
	public class ReloadRunnable implements Runnable
	{
		/**
		 * The first theta z index of our range
		 */
		int m_nStartIndex;
		
		/**
		 * The last theta z index of our range
		 */
		int m_nEndIndex;
		
		/**
		 * Set to <code>true</code> if scattering and attenuation should be neglected
		 */
		boolean m_bSuppressSA;
		
		/**
		 * Reference to the next matrix
		 */
		Matrix m_newMatrix;
		
		/**
		 * c'tor.
		 * 
		 * @param a_nStartIndex	The first theta z index of our range
		 * @param a_nEndIndex	The last theta z index of our range
		 * @param a_newMatrix	Reference to the next matrix
		 * @param a_bSuppressSA	Set to <code>true</code> if scattering and attenuation should be neglected
		 */
		public ReloadRunnable(int a_nStartIndex, int a_nEndIndex, Matrix a_newMatrix, boolean a_bSuppressSA)
		{
			m_nStartIndex = a_nStartIndex;
			m_nEndIndex = a_nEndIndex;
			m_bSuppressSA = a_bSuppressSA;
			m_newMatrix = a_newMatrix;
		}
		
		/**
		 * This function is called for each thread to do the scattering process.
		 */
		@Override
		public void run() 
		{
			for(int nTZ = m_nStartIndex; nTZ <= m_nEndIndex; nTZ++)
			{
				for(MatrixCell mc : m_matrix[nTZ])
				{
					mc.spreadPower(m_newMatrix, m_bSuppressSA);
				}
			}
		}
	}
	
	/**
	 * This function initializes this Matrix object from another.
	 * 
	 * @param a_matrix					The matrix from which this object should be initialized
	 * @param a_bCreateImpulseResponse	<code>true</code> if the impulse response should be prepared
	 * @param a_nNumberOfTimeSlices		The number of time slices for the impulse response
	 * @param a_bUseHyperSpace			<code>true</code> if the impulse response should use Hyperspace
	 */
	private void initFromMatrix(Matrix a_matrix, boolean a_bCreateImpulseResponse, int a_nNumberOfTimeSlices, boolean a_bUseHyperSpace) 
	{
		double dblOldMinTime = a_matrix.getCells()[0][0].getDIR().dblgetMinTime();
		double dblOldMaxTime = a_matrix.getCells()[0][0].getDIR().dblgetMaxTime();

		for(int nTZIndex = 0; nTZIndex <= m_nTZMaxIndex; nTZIndex++)
		{
			for(int nTPIndex = 0; nTPIndex <= m_nTPMaxIndex; nTPIndex++)
			{
				// clone the scatter cell for each matrix cell
				m_matrix[nTZIndex][nTPIndex] = a_matrix.getCells()[nTZIndex][nTPIndex].cloneScatter(this, a_bCreateImpulseResponse, a_nNumberOfTimeSlices, a_bUseHyperSpace, dblOldMinTime, dblOldMaxTime);
				
			}
		}
		
		// adapt the angles
		this.m_dblTZMax = a_matrix.getTZMax();
		this.m_dblTPMin = a_matrix.getTPMin();
		this.m_dblTPMax = a_matrix.getTPMax();
		
	}

	/**
	 * Write the total power of each scatter cell to a file.
	 * 
	 * @param a_strFile	Full qualified path to the output file.
	 */
	public void writeTP(String a_strFile)
	{
		try
		{
			FileWriter fw = new FileWriter(a_strFile);
			BufferedWriter bw = new BufferedWriter(fw);
			
			for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
			{
				for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
				{
					bw.write(this.m_matrix[nTZ][nTP].getScatterCell().getTotalPower() + " ");
				}
				bw.write("\n");
			}
			
			bw.close();
		}
		catch(Exception exc)
		{
			System.err.println(exc);
		}
	}
	
	/**
	 * This function smoothes the scatter distribution of each cell by the given amount of steps.
	 * 
	 * @param a_nSteps	Steps for the smoothing process
	 */
	public void fixScatterDistributions(int a_nSteps) 
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				m_matrix[nTZ][nTP].getScatterCell().fixDistribution(a_nSteps);
			}
		}
	}
	
	/**
	 * This function smoothes the total power of the scatter cells.
	 * 
	 * @param a_nSRTZ			Number of smoothing steps in the theta z direction
	 * @param a_nSRTP			Number of smoothing steps in the theta phi direction
	 * @param a_bForceRange		Forces the relative power to be between 0 and 1
	 * @param a_bSuppressHighTPs	Values for high theta phi steps can be problematic and therefore be suppressed
	 */
	public void fixMatrix(int a_nSRTZ, int a_nSRTP, boolean a_bForceRange, boolean a_bSuppressHighTPs)
	{
		double dblValuesTZ[] = new double[a_nSRTZ];
		double dblValuesTP[] = new double[a_nSRTP];
		
		// create Matrix with corrected power values for each cell
		double dblMatrixA[][] = new double[m_nTZMaxIndex+1][m_nTPMaxIndex+1];
		double dblMatrixB[][] = new double[m_nTZMaxIndex+1][m_nTPMaxIndex+1];
		
		// remove invalid data
		for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
		{
			for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
			{
				m_matrix[nTZ][nTP].getScatterCell().reduceNoise();
				m_matrix[nTZ][nTP].getScatterCell().deleteNegIntervals();
			}
		}
		
		// cut hight tps before smoothing
		if(a_bSuppressHighTPs)
		{
			for(int nTP = m_nTPMaxIndex - 2; nTP <= m_nTPMaxIndex; nTP++)
			{
				for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
				{
					m_matrix[nTZ][nTP].getScatterCell().normalizeToPower(0.0);
				}
			}
		}
		
		// smooth in TP Direction
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			// init old values dblValues
			for(int i = 0; i<dblValuesTP.length; i++){dblValuesTP[i] = 0.0;}
			
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				//fillmode average
				for(int i = 0; i<dblValuesTP.length; i++)
				{
					int index = nTP+i-a_nSRTP/2;
					
					if(index < 0)
					{
						dblValuesTP[i] = 0.0;
					}
					else if(index > m_nTPMaxIndex)
					{
						dblValuesTP[i] = 0.0;
					}
					else
						dblValuesTP[i] = this.m_matrix[nTZ][index].getScatterCell().getTotalPower();
				}

				// sum up values
				for(int i = 0; i<dblValuesTP.length; i++)
				{
					dblMatrixA[nTZ][nTP] += dblValuesTP[i];
				}
				
				if(nTP < a_nSRTP/2)
				{
					dblMatrixA[nTZ][nTP]/= (nTP+a_nSRTP/2);
				}
				else if(nTP > m_nTPMaxIndex-a_nSRTP/2)
				{
					dblMatrixA[nTZ][nTP]/= m_nTPMaxIndex-nTP+a_nSRTP/2+1;
				}
				else
					dblMatrixA[nTZ][nTP]/= a_nSRTP;
			}
		}
		
		// smooth in TZ Direction
		for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
		{
			// init old values dblValues
			for(int i = 0; i<dblValuesTZ.length; i++){dblValuesTZ[i] = 0.0;}
						
			for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
			{
				//fillmode average
				for(int i = 0; i<dblValuesTZ.length; i++)
				{
					int index = nTZ+i-a_nSRTZ/2;
					
					if(index < 0)
					{
						dblValuesTZ[i] = 0.0;
					}
					else if(index > m_nTZMaxIndex)
					{
						dblValuesTZ[i] = 0.0;
					}
					else
						dblValuesTZ[i] = dblMatrixA[index][nTP];
				}

				// sum up values
				for(int i = 0; i<dblValuesTZ.length; i++)
				{
					dblMatrixB[nTZ][nTP] += dblValuesTZ[i];
				}
				
				if(nTZ < a_nSRTZ/2)
				{
					dblMatrixB[nTZ][nTP]/= (nTZ+a_nSRTZ/2);
				}
				else if(nTZ > m_nTZMaxIndex-a_nSRTZ/2)
				{
					dblMatrixB[nTZ][nTP]/= m_nTZMaxIndex-nTZ+a_nSRTZ/2+1;
				}
				else
					dblMatrixB[nTZ][nTP]/= a_nSRTZ;
			}
		}
		
		// normalize scatter cells
		for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
		{
			for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
			{
				
				double dblPower = dblMatrixB[nTZ][nTP];
				
				if(a_bForceRange)
				{
					if(dblPower < 0.0)
					{
						dblPower = 0.0;
					}
					else if(dblPower > 1.0)
					{
						dblPower = 1.0;
					}
				}

				m_matrix[nTZ][nTP].getScatterCell().normalizeToPower(dblPower);
				
				// Test
				double dblPowerNew = m_matrix[nTZ][nTP].getScatterCell().getTotalPower();
				if(dblPowerNew >= 0)
				{
					System.out.println(""+nTZ + " " + nTP);
				} 
			}
		}
	}
	
	/**
	 * This function checks each scatter cell and reports its status.
	 */
	public void checkMatrix()
	{
		int nOK = 0;
		int nNOK = 0;

		double dblMax = 0.0;
		double dblMin = 0.0;
		
		int nTZMax = 0;
		int nTPMax = 0;
		
		int nTZMin = 0;
		int nTPMin = 0;

		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				// prechecks
				double dblTP = m_matrix[nTZ][nTP].getScatterCell().getTotalPower();
				
				if(dblTP < 0 || dblTP > 1)
				{
					nNOK++;
					
					
					if(dblTP > dblMax)
					{
						dblMax = dblTP;
						nTZMax = nTZ;
						nTPMax = nTP;
					}
					
					if(dblTP < dblMin)
					{
						dblMin = dblTP;
						nTZMin = nTZ;
						nTPMin = nTP;
					}	
				}
				else
				{	
					nOK++;
				}
			}
		}
		
		System.out.println("OK: " + nOK);
		System.out.println("NOK: " + nNOK);
		System.out.println("MIN: " + dblMin);
		System.out.println(getFilenameForAngles(nTZMin, nTPMin));
		System.out.println("MAX: " + dblMax);
		System.out.println(getFilenameForAngles(nTZMax, nTPMax));	
	}
	
	/**
	 * This function returns the array of all matrix cells.
	 * 
	 * @return MatrixCell array
	 */
	public MatrixCell[][] getMatrix()
	{
		return m_matrix;
	}
	
	/**
	 * This function sums up the power of the total matrix.
	 * 
	 * @param a_bUseDIRs	If set to <code>true</code>, the power of the impulse reponses is summed up, 
	 * 						otherwise the relative power of the scatter cells is considered
	 */
	public void writeTotalPower(boolean a_bUseDIRs) 
	{
		double dblTotalPower = 0.0;
		
		int nStep = 1;
				
		int nTZMin = 0;
		int nTZMax = m_nTZMaxIndex;
		int nTPMin = 0;
		int nTPMax = m_nTPMaxIndex;

		for(int nTZ = nTZMin; nTZ <= nTZMax; nTZ+=nStep)
		{
			for(int nTP = nTPMin; nTP <= nTPMax; nTP+=nStep)
			{
				double dblP = 0.0;
				
				if(!a_bUseDIRs)
				{
					dblP = m_matrix[nTZ][nTP].getScatterCell().getTotalPower();
				}
				else
				{
					dblP = m_matrix[nTZ][nTP].getDIR().getTotalPower();
				}

				dblTotalPower += dblP;
			}
		}
		
		double dblDivZ = m_nTZMaxIndex+1;
		double dblDivP = m_nTPMaxIndex+1;
	
		System.out.println(nTPMin + "\t" + dblTotalPower/((dblDivZ)*(dblDivP)));
	}

	/**
	 * This function returns the relative distance between two matrices.
	 * 
	 * @return Distance between two matrices
	 */
	public double getRelativeDistance() 
	{
		return m_dblDistanceDelta;
	}

	/**
	 * This function adds a power at adiscrete time.
	 * 
	 * @param a_nTZIndex	Index for theta z of the target cell
	 * @param a_nTPIndex	Index for theta phi of the target cell
	 * @param a_dblNewTime	Time of the power peak
	 * @param a_dblNewPower	Power of the power peak
	 */
	public void addPowerPeak(int a_nTZIndex, int a_nTPIndex, double a_dblNewTime, double a_dblNewPower) 
	{
		m_matrix[a_nTZIndex][a_nTPIndex].getDIR().addPowerPeak(a_dblNewTime, a_dblNewPower);
	}

	/**
	 * This function creates an impulse response from the impulse responses of each cell. 
	 * 
	 * @param a_nSteps			The number of steps for the impulse response
	 * @param a_bAllEqual		<code>true</cdoe> if all cell impulse responses have the same spacing
	 * @param bUseHyperSpace	<code>true</cdoe> if HyperSpace should be applied
	 * @param a_bTestMode		This flag is used to enable a Testmode that execudes special code for debugging purposes
	 * 
	 * @return	The created impulse response
	 * @throws Exception 
	 */
	public DiscreteImpulseResponse createImpulseResponse(int a_nSteps, boolean a_bAllEqual, boolean bUseHyperSpace, boolean a_bTestMode) throws Exception 
	{
		DiscreteImpulseResponse dir = new DiscreteImpulseResponse(bUseHyperSpace);
		
		// first search for min and max time
		double dblMin = Double.NaN;
		double dblMax = Double.NaN;
		
		int nTZStart = 0;
		int nTPStart = 0;
		
		if(a_bTestMode)
		{
			nTZStart = 90;
			m_nTZMaxIndex = 90;
			nTPStart = 0;
			m_nTPMaxIndex = 0;
			
		}
		
		// sum up all Impulse Responses
		for(int nTZ = nTZStart; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = nTPStart; nTP <= m_nTPMaxIndex; nTP++)
			{
				if(null != m_matrix[nTZ][nTP])
				{
					if(m_matrix[nTZ][nTP].getDIR().isPeak())
					{
						Vector<double[]> vecDs = m_matrix[nTZ][nTP].getDIR().getPeaks();
						
						for(double[] da : vecDs)
						{
							//dir.addPowerPeak(ds[0], ds[1]);
							if(dblMin > da[0] || Double.isNaN(dblMin))
							{
								dblMin = da[0];
							}
							
							if(dblMax < da[0] || Double.isNaN(dblMax))
							{
								dblMax = da[0];
							}					
						}
					}
					else
					{
						double dblMinCurrent = m_matrix[nTZ][nTP].getDIR().dblgetMinTime();
						double dblMaxCurrent = m_matrix[nTZ][nTP].getDIR().dblgetMaxTime();
						
						if(dblMin > dblMinCurrent || Double.isNaN(dblMin))
						{
							dblMin = dblMinCurrent;
						}
						
						if(dblMax < dblMaxCurrent || Double.isNaN(dblMax))
						{
							dblMax = dblMaxCurrent;
						}	
					}
				}
			}
		}
		
		dir.createImpulseResponse(a_nSteps, m_matrix, dblMin, dblMax, a_bAllEqual);
		
		return dir;
	}

	/**
	 * This function drops the references to all scatter files to support the garbage collector.
	 */
	public void dropScatterFiles() 
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				m_matrix[nTZ][nTP].dropScatterFile();
				m_matrix[nTZ][nTP] = null;
			}
		}
	}

	/**
	 * This function prints the total power of the impulse responses of the matrix.
	 */
	public void printTotalPower() 
	{
		double dblPower = 0.0;
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				dblPower += m_matrix[nTZ][nTP].getDIR().getTotalPower();
			}
		}
		
		System.out.println("TotalPowerOfMatrix: " + dblPower);
	}
	
	/**
	 * This function writes the total power of the impulse responses to a file.
	 * 
	 * @param strFileOut	Full qualified path to the output file
	 * @param a_bUseDegrees	If <code>true</code> the angle is converted to degrees
	 * @throws IOException	Exception that might occur while writing to the file
	 */
	public void saveMatrixPower(String strFileOut, boolean a_bUseDegrees) throws IOException
	{
		FileWriter fw = new FileWriter(strFileOut);
		BufferedWriter bw = new BufferedWriter(fw);
		
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)

        for(int i = 0; i < getTZSteps(); i++)
        {
        	double dblDeltaTZ = AngleHelper.getInstance().getStepWidthForStep(i);
        	
        	double dblCurrentAngle = AngleHelper.getInstance().getCenterAngleForIndex(i);
        	double dblCurrentPower = 0.0;
        	
        	for(int j = 0; j < getTPSteps(); j++)
        	{
        		
        		dblCurrentPower += getCells()[i][j].getDIR().getTotalPower();
        	}
        	
        	if(!a_bUseDegrees)
        	{
        		dblDeltaTZ *= Math.PI/180.0;
        		dblCurrentAngle *= Math.PI/180.0;
        	}
        	
        	bw.write(dblCurrentAngle + " " + dblCurrentPower/dblDeltaTZ + "\n");
        }
		
		bw.flush();
		bw.close();
	}
	
	/**
	 * This function returns the absolute distance of the matrix.
	 * 
	 * @return	Absolute distance
	 */
	public double getAbsoluteDistance() 
	{
		return this.m_dblDistanceAbsolute;
	}

	/**
	 * This function exports the scatter data to a given folder.
	 * 
	 * @param strFolder		Full qualified path to the destination folder
	 * @throws Exception	Exception that might occur during the export process
	 */
	public void exportScatterFiles(String strFolder) throws Exception 
	{
		File file = new File(strFolder);
		if(!file.exists())
		{
			file.mkdir();
		}
		
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				m_matrix[nTZ][nTP].exportScatterFile(strFolder + "/" + getFilenameForAngles(nTZ, nTP));
			}
		}
	}

	/**
	 * This function applies Fresnel losses to the impulse responses.
	 * 
	 * @param a_ft					The Fresnel losses to be applied
	 * @param a_bUseOutsideAngle	<code>true</code> if the angle is converted to the outside of the fiber before determining the Fresnel losses
	 */
	public void applyFresnelTransmission(FresnelTransmission a_ft, boolean a_bUseOutsideAngle) 
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			double dblAngle = m_matrix[nTZ][0].getScatterCell().getThetaZInside()*Math.PI/180.0;
			if(a_bUseOutsideAngle)
			{
				dblAngle = m_ta.getAngle1(dblAngle);
			}
			double dblFactor = a_ft.getTransmissionForAngle(dblAngle);
			
			
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				DiscreteImpulseResponse dir = m_matrix[nTZ][nTP].getDIR();
				
				double[] dPower = dir.getPower();
				for(int i = 0; i < dPower.length; i++)
				{
					dPower[i] *= dblFactor;
				}
			}
		}
	}
	
	/**
	 * This function limits the power distribution of this matrix to the given angle.
	 * 
	 * @param a_dblMaxAngle	The maximum angle for theta z outside the fiber
	 */
	public void applyMaxAngle(double a_dblMaxAngle) 
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			double dblFactor = 1.0;
			double dblAngle = m_matrix[nTZ][0].getScatterCell().getThetaZInside()*Math.PI/180.0;
			{
				dblAngle = m_ta.getAngle1(dblAngle)*180.0/Math.PI;
			}
			
			if(dblAngle >= a_dblMaxAngle)
			{
				dblFactor = 0.0;
			}
			
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				DiscreteImpulseResponse dir = m_matrix[nTZ][nTP].getDIR();
				
				double[] dPower = dir.getPower();
				for(int i = 0; i < dPower.length; i++)
				{
					dPower[i] *= dblFactor;
				}
			}
		}
	}
	
	/**
	 * This function applies Fresnel reflection to the impulse responses of this matrix.
	 * 
	 * @param a_fT	The Fresnel reflection to be applied
	 */
	public void applyFresnelReflection(FresnelTransmission a_fT)
	{
		applyFresnelReflection(a_fT, 1);
	}
	
	/**
	 * This function applies Fresnel reflections multiple times to the impulse responses of this matrix.
	 * 
	 * @param a_fT	The Fresnel reflection to be applied
	 * @param a_nCount	The numer of times the reflections should be applied
	 */
	public void applyFresnelReflection(FresnelTransmission a_fT, int a_nCount) 
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			double dblFactor = 1-a_fT.getTransmissionForAngle(m_matrix[nTZ][0].getScatterCell().getThetaZInside()*Math.PI/180.0);
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				DiscreteImpulseResponse dir = m_matrix[nTZ][nTP].getDIR();
				
				double[] dPower = dir.getPower();
				for(int i = 0; i < dPower.length; i++)
				{
					dPower[i] *= dblFactor*a_nCount;
				}
			}
		}
	}
	
	/**
	 * This function returns the strain that is applied to the matrix.
	 * 
	 * @return	Strain
	 */
	public double getStrain()
	{
		return m_dblStrain;
	}
	
	/**
	 * This function applies an angular filter to the discrete impulse responses.
	 * 
	 * @param a_lsad	The filter to be applied
	 */
	public void applyAngularFilter(LightSourceAngularDistribution a_lsad)
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			double dblFactor = a_lsad.getValue((m_matrix[nTZ][0].getScatterCell().getThetaZInside()*Math.PI/180.0));
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				DiscreteImpulseResponse dir = m_matrix[nTZ][nTP].getDIR();
				
				double[] dPower = dir.getPower();
				for(int i = 0; i < dPower.length; i++)
				{
					dPower[i] *= dblFactor;
				}
			}
		}
	}

	
	/**
	 * This function applies the double Fresnel reflection that accurs between two fibers.
	 * 
	 * @param f2	The Fresnel losses that should be applied.
	 */
	public void applyConnectorFresnelReflection(FresnelTransmission f2) 
	{
		for(int nTZ = 0; nTZ <= m_nTZMaxIndex; nTZ++)
		{
			// unfortunately we need 4 different factors...
			double dblAngleInternal = m_matrix[nTZ][0].getScatterCell().getThetaZInside()*Math.PI/180.0;
			double dblTInternal = f2.getTransmissionForAngle(dblAngleInternal);
			double dblRInternal = 1-dblTInternal;
			
			double dblAngleExternal = m_ta.getAngle1(dblAngleInternal);
			double dblTExternal = f2.getTransmissionForAngle(dblAngleExternal);
			double dblRExternal = 1-dblTExternal;
			
			for(int nTP = 0; nTP <= m_nTPMaxIndex; nTP++)
			{
				DiscreteImpulseResponse dir = m_matrix[nTZ][nTP].getDIR();
				
				double[] dPower = dir.getPower();
				for(int i = 0; i < dPower.length; i++)
				{
					dPower[i] *= (dblRInternal+dblRExternal*dblTInternal*dblTInternal);
				}
			}
		}
	}
}

