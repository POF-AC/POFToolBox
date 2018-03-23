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

import org.leores.plot.JGnuplot;
import org.leores.util.data.DataTableSet;

import model.AngleHelper;
import model.threeDimensional.Function3D;
import model.threeDimensional.Discrete.DiscreteImpulseResponse;
import model.threeDimensional.Discrete.Matrix;
import model.threeDimensional.Discrete.NormTimeSlicer;

/**
 * This class expands JGnuplot and adds functions to print data from our model
 * 
 * @author Thomas Becker
 * 
 */
public class MyJGnuplot extends JGnuplot 
{
	/**
	 * This method plots a power distribution represented by an object of the class Function3D.
	 * 
	 * @param a_function	The Function describing the power distribution over theta z and theta phi
	 * @param a_nSteps		Number of steps at which the function should be evalutated
	 */
	public void plotFunction(Function3D a_function, int a_nSteps)
	{
		Plot plot = new Plot("") {
            {
            	this.yrange = "[0:"+ Math.PI/2 +"]";
            	this.xrange = "[0:"+ 0.8 +"]";
            	this.zrange = "[0:3]";
            	this.xlabel = "{/Symbol \121}_z";
            	this.ylabel = "{/Symbol \121}_{/Symbol \152}";
            }
        };
        
        int nPoints = a_nSteps;
        double dblPoints = (double)nPoints;
        
        // Evaluate the function for each point
        double[][] dblMatrix = new double[nPoints][nPoints];
        
        for(int i = 0; i < nPoints; i++)
        {
        	for(int j = 0; j < nPoints; j++)
        	{
        		dblMatrix[i][j] = a_function.getValue(((double)i)*(Math.PI/2)/(dblPoints-1), ((double)j)*(Math.PI/2)/(dblPoints-1) );
        	}
        }
        	
        // the data has to be transferred to three arrays, one for each dimension
        double [] dblTZ = new double[nPoints*nPoints];
        double [] dblTP = new double[nPoints*nPoints];
        double [] dblY = new double[nPoints*nPoints];
            
        for(int i = 0; i < nPoints; i++)
        {
        	for(int j = 0; j < nPoints; j++)
        	{
        		dblTZ[i*nPoints+j] = ((double)i)*(Math.PI/2)/(dblPoints-1);
        		dblTP[i*nPoints+j] = ((double)j)*(Math.PI/2)/(dblPoints-1);
        		dblY[i*nPoints+j] = dblMatrix[i][j];
        	}
        }
               
        // ready to plot
        DataTableSet dts = plot.addNewDataTableSet("3D power distribution");
        dts.addNewDataTable("3DPower", dblTZ, dblTP, dblY);
      	
        execute(plot, this.plot3d);
	}
	
	/**
	 * This method plots a power distribution of an instance of the class Matrix.
	 * 
	 * @param a_matrix	The matrix which power distribution is about to be plotted
	 */
	public void plotMatrixPower(Matrix a_matrix)
	{
		// determine the angular ranges
		final double dblTZMin = 0.0;
		final double dblTZMax = a_matrix.getTZMax();
		
		final double dblTPMin = a_matrix.getTPMin();
		final double dblTPMax = a_matrix.getTPMax();
		
		Plot plot = new Plot("") {
            {
            	this.yrange = "[" + dblTPMin + ":"+ dblTPMax +"]";
            	this.xrange = "["+ dblTZMin + ":"+ dblTZMax +"]";
            	this.xlabel = "{/Symbol \121}_z";
            	this.ylabel = "{/Symbol \121}_{/Symbol \152}";
            }
        };
        
        // fill the thee dimensional matrices which are required for the plot
        int nTotalPoints = a_matrix.getTZSteps()*a_matrix.getTPSteps();
        double [] dblTZ = new double[nTotalPoints];
        double [] dblTP = new double[nTotalPoints];
        double [] dblY = new double[nTotalPoints];
        
        for(int i = 0; i < a_matrix.getTZSteps(); i++)
        {
        	double dblDeltaTZ = AngleHelper.getInstance().getStepWidthForStep(i);
        	
        	for(int j = 0; j < a_matrix.getTPSteps(); j++)
        	{
        		
        		double dblDeltaTP = a_matrix.getCells()[i][j].getScatterCell().getThetaPhiMax()-a_matrix.getCells()[i][j].getScatterCell().getThetaPhiMin();
        		dblTZ[i*a_matrix.getTPSteps()+j] = a_matrix.getCells()[i][j].getScatterCell().getThetaZInside();
        		dblTP[i*a_matrix.getTPSteps()+j] = (a_matrix.getCells()[i][j].getScatterCell().getThetaPhiMin()+a_matrix.getCells()[i][j].getScatterCell().getThetaPhiMax())/2.0;
        		dblY[i*a_matrix.getTPSteps()+j] = a_matrix.getCells()[i][j].getDIR().getTotalPower()/(dblDeltaTZ*dblDeltaTP);
        	}
        }
        
        // ready to plot   
        DataTableSet dts = plot.addNewDataTableSet("3D Plot");
        dts.addNewDataTable("Map", dblTZ, dblTP, dblY);
      	execute(plot, this.plot3d);
	}

	/**
	 * This function plots a discrete impulse response.
	 * 
	 * @param a_dir				The discrete impulse response to be plotted
	 * @param a_bUseHyperSpace	<code>true</code> if the impulse response uses hyper space, <code>else</code> otherwise
	 * @param a_strTitle		The title for the plot
	 */
	public void plotDIR(DiscreteImpulseResponse a_dir, boolean a_bUseHyperSpace, String a_strTitle) 
	{
		final double dblMinTime = a_dir.dblgetMinTime();
		final double dblMaxTime = a_dir.dblgetMaxTime();
		int nSteps = a_dir.getSteps();
		double dblStepTime = (dblMaxTime-dblMinTime)/(double)nSteps;
		
		Plot plot = new Plot("") {
            {
            	this.xrange = "["+dblMinTime*1.0E9+":"+ dblMaxTime*1.0E9 +"]";
            	this.xlabel = "t [nano seconds]";
            	this.ylabel = "dP/dT";
            }
        };
        
        double [] dblT = new double[nSteps];
        double [] dblY = new double[nSteps];
        
        for(int i = 0; i < nSteps; i++)
        {
        	if(a_bUseHyperSpace)
    		{
        		dblT[i] = NormTimeSlicer.getInstance(851, a_dir.m_dblStrain).getNormTime(i)*dblMinTime*1.0E9;
    		}
        	else
        	{
            	dblT[i] = (dblMinTime + (double)i*dblStepTime)*1.0E9;
        	}
        	

        	dblY[i] = a_dir.getPower()[i];
        }
        	
        // ready to plot   
        DataTableSet dts = plot.addNewDataTableSet("Impulse Response");
        dts.addNewDataTable(a_strTitle, dblT, dblY);
      	execute(plot, this.plot2d);
	}
}