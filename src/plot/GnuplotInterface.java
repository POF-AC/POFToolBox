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

package plot;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;

import javax.swing.JOptionPane;
import javax.swing.JTextField;

/**
 * This class provides access to gnuplot and offers some functionality for function fitting and plotting.
 * Warning: This class contains Linux specific code and expects gnuplot to reside in /usr/bin/gnuplot. It should
 * be easy to port it to any other OS capable of running Gnuplot but there simply was no necessity yet. Especially 
 * the fitting routine relies heavily on the exact syntax of the Output of Gnuplot 5.0 patchlevel 7. 
 * It may break for newer versions of Gnuplot. 
 * 
 * @author Thomas Becker
 *
 */
public class GnuplotInterface 
{
	/**
	 * Buffered reader to read data from gnuplot.
	 */
	private BufferedReader m_reader;
	
	/**
	 * Buffered writer to write data to gnuplot.
	 */
	private BufferedWriter m_writer;
	

	/**
	 * c'tor, starts gnuplot and sets up the communication lines.
	 * 
	 * @throws IOException
	 */
	public GnuplotInterface() throws IOException
	{
		ProcessBuilder builder = new ProcessBuilder("/usr/bin/gnuplot", "-p");
		Process p = builder.start();
	
		InputStream ip = p.getErrorStream();
		OutputStream op = p.getOutputStream();
		
		OutputStreamWriter osw = new OutputStreamWriter(op);
		InputStreamReader isr = new InputStreamReader(ip);
		
		m_reader = new BufferedReader (isr);
		m_writer = new BufferedWriter(osw);
	}
	
	/**
	 * This function is for demonstrational purposes only.
	 * 
	 * @param a_strArgs	Command line arguments (not used)
	 */
	static public void main(String[] a_strArgs)
	{
		
		try 
		{
			GnuplotInterface gi = new GnuplotInterface();
			gi.plotScatterDistributions("/path/to/scatterdistributions/");
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
	
	/**
	 * This function plots all scatter distributions that are stored in the provided folder. 
	 * Angles from 0 to 85 degree are expected as well as blade positions from 0 to 0.485 micro meter.
	 *  	
	 * @param a_strFolder	Folder of the scatter distributions
	 * @throws Exception	Exception if something goes wrong
	 */
	public void plotScatterDistributions(String a_strFolder) throws Exception
	{
		int nRStart = 0;
		int nREnd = 85;
		
		double dblXStart = 0.0;
		double dblXEnd = 0.485;
		
		m_writer.write("plot ");
			
		double a_dblStepsize = 0.005;
		for(double nR = nRStart; nR <= nREnd; nR+=0.1)
		{
			for(double dblX = dblXStart; dblX < dblXEnd; dblX+=a_dblStepsize)
			{
				DecimalFormat dx = new DecimalFormat("#.###");
				String strx1 = dx.format(dblX);
				String strx2 = dx.format(dblX+a_dblStepsize);
				
				DecimalFormat dr = new DecimalFormat("##.#");
				String strdr = dr.format(nR);
				String strFileNameOut = a_strFolder +  "/R=" + strdr + ",X=" + strx1 + "-" + strx2 + ".txt";
				
				m_writer.write("'" + strFileNameOut + "' " +  " with lines title '"+nR + " " +strx1 +"', ");
				m_writer.flush();
			
			}
		}
		
		m_writer.write("\n");
		m_writer.flush();
		
		// read and print whatever gnuplot gives us in return
		String strLine = null;
		while(null != (strLine = m_reader.readLine()))
		{
			System.out.println(strLine);
		}
	}
	
	/**
	 * It rarely happens that gnuplot crashes or hangs up the connection. This function can be used to kill all running instances of Gnuplot, brings it back to life
	 * and reinitiates the connections.
	 */
	private void reset() 
	{
		try 
		{
			Process a = Runtime.getRuntime().exec(new String[]{"killall", "gnuplot"});
			int exitCode = a.waitFor();
			
			ProcessBuilder builder = new ProcessBuilder("/usr/bin/gnuplot", "-p");
			
			Process p;
			p = builder.start();
			InputStream ip = p.getErrorStream();
			OutputStream op = p.getOutputStream();
			
			OutputStreamWriter osw = new OutputStreamWriter(op);
			InputStreamReader isr = new InputStreamReader(ip);
			
			m_reader = new BufferedReader (isr);
			m_writer = new BufferedWriter(osw);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}

	}
	
	/**
	 * This function instructs Gnuplot to fit that data stored in <code>a_strFilename</code>
	 * 
	 * @param a_strFilename		Full qualified path to the file that contains the data for the fit
	 * @param a_nMaxCycles		Maximum count of iterations for the fit
	 * @param a_dblCenter		This parameter is used to initialize the fit parameter that is responsible for the maximum value
	 * @param a_nFitType		The type of the function that should be fitted. See enum <code>FitTypes</code>
	 * @return					ResultSet with the calculated data
	 * @throws Exception		Different Exceptions can occur while communicating with Gnuplot
	 */
	public ResultSet fitFile(String a_strFilename, int a_nMaxCycles, double a_dblCenter, FitTypes a_nFitType) throws Exception
	{
		// we consider the fit to be scuccessfull if the error is less than 1% for each parameter
		boolean bSuccess = false;
		
		// cycle counter
		int nCounter = 0;
		
		// ResultSet with the determined parameters
		ResultSet rs = null;
		
		// variable to check the development of the error of the first parameter
		double OldValue = 0;
	
		boolean bFirstRun = true;
		
		// while we haven't been successfull and the max cycles have not been reached 
		while (!bSuccess && nCounter < a_nMaxCycles)
		{
			// perform the real fit
			rs = fitInternal(a_strFilename, a_dblCenter, a_nFitType, bFirstRun);
			
			if(bFirstRun)
			{
				bFirstRun = false;
			}
			
			// fit successfull?
			if (rs.m_dblErrorPerCent[0] <= 1 && rs.m_dblErrorPerCent[1] <= 1 && rs.m_dblErrorPerCent[2] <= 1)
			{
				bSuccess = true;
			}
			
			// if we are not making any process we kill the fit after 10 iterations
			if(OldValue == rs.m_dblErrorPerCent[0] && nCounter > 10)
			{
				break;
			}
			
			// if the error gets larger again, we kill it as well
			if(OldValue <= rs.m_dblErrorPerCent[0] && nCounter > 2)
			{
				break;
			}
	
			OldValue = rs.m_dblErrorPerCent[0];
			
			System.out.println("Loop " + nCounter);
			nCounter++;
		}
				
		return rs;
	}
	
	/**
	 * This function plots the data residing in the given file.
	 * 
	 * @param a_strFilename	Full qualified path to the data file
	 * @param _a_strUsing	String containing the using pattern describing which columns to plot
	 * 
	 * @throws Exception	Communication errors with Gnuplot can lead to exceptions
	 */
	private void plotFunction(String a_strFilename, String _a_strUsing) throws Exception
	{
		m_writer.write("plot f(x), '" + "' " + _a_strUsing + " with lines\n");
		m_writer.flush();
	}
	
	/**
	 * This function provides the basic fitting routine without any error checking.
	 * 
	 * @param a_strFilename	Full qualified path to the file containing the data
	 * @param a_dblCenter	This parameter is used to initialize the fit parameter that is responsible for the maximum value
	 * @param a_nFitType	The type of the function that should be fitted. See enum <code>FitTypes</code>
	 * @param bFirstRun		Has to be set to <code>true</code> for the first run, since Gnupot neads additional calls in this case
	 * @return				ResultSet with the calculated data
	 * @throws Exception	Different Exceptions can occur while communicating with Gnuplot
	 */
	private ResultSet fitInternal(String a_strFilename, double a_dblCenter, FitTypes a_nFitType, boolean bFirstRun) throws Exception
	{
		// Gaussian Fit
		if(FitTypes.FT_GAUS_2 == a_nFitType)
		{
			if(bFirstRun)
			{
				m_writer.write("f(x) = a*exp( (-(x-b)**2) /2*c**2 ) \n");
				m_writer.write("b = "+a_dblCenter+"\n");
				m_writer.write("a = 1\n");
				m_writer.write("c = 1\n");
			}
			
			m_writer.write("fit f(x) '" + a_strFilename +"' using 1:2  via a,b,c \n");
		}
		// Slightly modified Gaussian fit of the third order
		else if(FitTypes.FT_GAUS_3 == a_nFitType)
		{
			if(bFirstRun)
			{
				m_writer.write("f(x) = a*exp( (-(x-b)**3) /2*c**2 ) \n");
				m_writer.write("b = "+a_dblCenter+"\n");
			}
			m_writer.write("fit f(x) '" + a_strFilename +"' using 1:2  via a,b,c \n");
		}
		// Slightly modified Gaussian fit of the fourth order
		else if(FitTypes.FT_GAUS_4 == a_nFitType)
		{
			if(bFirstRun)
			{
				m_writer.write("f(x) = a*exp( (-(x-b)**4) /2*c**2 ) \n");
				m_writer.write("b = "+a_dblCenter+"\n");
			}
			m_writer.write("fit f(x) '" + a_strFilename +"' using 1:2  via a,b,c \n");
		}
		// This fit type is used for experimental purposes and can be adjusted
		else if(FitTypes.FT_EXPERIMENTAL == a_nFitType)
		{
			if(bFirstRun)
			{
				m_writer.write("f(x) = a*exp( (-(x-b)**4) /2*c**2 + (-(x-d)**2) /2*e**2 ) \n");
				m_writer.write("b = "+a_dblCenter+"\n");
				m_writer.write("d = "+4+"\n");
			}
			m_writer.write("fit f(x) '" + a_strFilename +"' using 1:2  via a,b,c,d,e \n");
		}
				
		m_writer.flush();
		
		String strLine = null;
		
		ResultSet rs = new ResultSet(a_nFitType);
		
		int nCounter = 0;
		
		// read whatever Gnuplot is sending us
		while(null != (strLine = m_reader.readLine()))
		{
			nCounter++;
			System.out.println(strLine);
			
			// that didn't work, try again with b = 1
			if(strLine.startsWith("         line 0: Singular matrix in"))
			{
				m_writer.write("b = 1\n");
				m_writer.write("fit f(x) '" + a_strFilename +"' using 1:2  via a,b,c \n");
				m_writer.flush();
			}
			
			// we have a successful result
			if(strLine.startsWith("Final set of parameters"))
			{
				m_reader.readLine();
				System.out.println(strLine);
				strLine = m_reader.readLine();
				System.out.println(strLine);
				String[] sa = strLine.split(" ");
				
				String spc = sa[sa.length-1];
		
				rs.m_dblErrorPerCent[0] = Double.parseDouble(spc.substring(1, spc.length() -2));
				rs.m_daParameters[0] = Double.parseDouble(sa[16]);
				
				// b
				strLine = m_reader.readLine();
				sa = strLine.split(" ");
				spc = sa[sa.length-1];
				rs.m_dblErrorPerCent[1] = Double.parseDouble(spc.substring(1, spc.length() -2));
				rs.m_daParameters[1] = Double.parseDouble(sa[16]);
								
				// c
				strLine = m_reader.readLine();;
				sa = strLine.split(" ");
				spc = sa[sa.length-1];
				rs.m_dblErrorPerCent[2] = Double.parseDouble(spc.substring(1, spc.length() -2));
				rs.m_daParameters[2] = Double.parseDouble(sa[16]);
				
				break;	
			}
			
			// wild guess: 500 lines of output and not finished means it is not going to finish at all
			if(nCounter > 500)
			{
				rs.m_dblErrorPerCent[0] = 0;
				rs.m_dblErrorPerCent[1] = 0;
				rs.m_dblErrorPerCent[2] = 0;
				rs.m_daParameters[0] = 0;
				rs.m_daParameters[1] = 0;
				rs.m_daParameters[2] = 0;
				
				//reset Gnuplot just to be sure everything is fine again
				this.reset();
				
				break;
			}
		}
		
		return rs;
	}
	
	/**
	 * This enum lists the fit types that are currently supported.
	 * 
	 * @author Thomas Becker
	 *
	 */
	static public enum FitTypes
	{
		FT_LINEAR,
		FT_GAUS_2,
		FT_GAUS_3,
		FT_GAUS_4, 
		FT_EXPERIMENTAL,
		FT_UNKNOWN
	}
	
	/**
	 * This class holds the results for a fit.
	 * 	
	 * @author Thomas Becker
	 *
	 */
	public class ResultSet
	{
		// array for the parameters		
		public double[] m_daParameters;
		
		// array for the errors of the parameters
		public double[] m_dblErrorPerCent;
		
		// number fit attempts
		int m_nNumberOfTries = 0;
		
		// fit type
		FitTypes m_nFitType;
		
		/**
		 * c'tor. Initializes the data arrays.
		 * 
		 * @param a_nFitType The type that should be used for the fit
		 */
		public ResultSet(FitTypes a_nFitType)
		{
			m_daParameters = new double[5];
			m_dblErrorPerCent = new double[5];
			m_nFitType = a_nFitType;
		}
		
		/**
		 * c'tor for unkown fit types.
		 * 
		 * @param a_nNumber If we don't know the fit type, we need to know how many parameters we should handle.
		 */
		public ResultSet(int a_nNumber)
		{
			m_daParameters = new double[a_nNumber];
			m_dblErrorPerCent = new double[a_nNumber];
			m_nFitType = FitTypes.FT_UNKNOWN;
		}

		/**
		 * This function writes the results of this class to a text file.
		 * 
		 * @param string		Full qualified path to the file
		 * @throws IOException	Exceptions that can occur while writing to the output file
		 */
		public void writeToFile(String string) throws IOException 
		{
			FileWriter fs = new FileWriter(string);
			BufferedWriter gw = new BufferedWriter(fs);
			
			gw.write("FitType: " + m_nFitType);
			if(FitTypes.FT_EXPERIMENTAL == m_nFitType)
			{
				gw.write("#Error %: " + m_daParameters[0] + " " + m_daParameters[1] + " " + m_daParameters[2] + " " + m_daParameters[3] + " " + m_daParameters[4] + "\n");
			}
			else
			{
				gw.write("#Error %: " + m_daParameters[0] + " " + m_daParameters[1] + " " + m_daParameters[2] + "\n");
			}
			
			gw.write("#Number of Tries: " + m_nNumberOfTries + "\n");
			gw.write(m_daParameters[0] + " " + m_daParameters[1] + " " + m_daParameters[2] + "\n");
			
			gw.close();
		}
	}
}
