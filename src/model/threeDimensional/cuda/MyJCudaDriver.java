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
package model.threeDimensional.cuda;

import static jcuda.driver.JCudaDriver.cuMemAllocHost;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

/**
 * This class is a wrapper for the JCudaDriver and allows to switch easily between the memory allocation on the host and on the graphics card.
 * 
 * @author Thomas Becker
 *
 */
public class MyJCudaDriver  
{
	/**
	 * This method allocates memory for JCuda.
	 * 
	 * @param a_cudpointer	Cuda device pointer
	 * @param a_nbytesize	Size of the requested memory in bytes
	 * @param a_bHost	If <code>true</code>, the memory is allocated on the host, otherwise on the graphics card
	 * 
	 * @return Result of JCudaDriver.cuMemAlloc
	 */
	static public int cuMemAlloc(CUdeviceptr a_cudpointer, long a_nbytesize, boolean a_bHost)
	{
		if(!a_bHost)
		{
			return JCudaDriver.cuMemAlloc(a_cudpointer, a_nbytesize);
		}
		else
		{
			return cuMemAllocHost(a_cudpointer, a_nbytesize);
		}
	}
	
	/**
	 * This method frees memory for JCuda.
	 * 
	 * @param a_cudpointer	Cuda device pointer
	 * @param a_bHost		If <code>true</code>, the memory is released on the host, otherwise on the graphics card
	 * 
	 * @return				Result of JCudaDriver.cuMemFree
	 */
	static public int cuMemFree(CUdeviceptr a_cudpointer, boolean a_bHost)
	{
		if(!a_bHost)
		{
			return JCudaDriver.cuMemFree(a_cudpointer);
		}
		else
		{
			return JCudaDriver.cuMemFreeHost(a_cudpointer);
		}
	}
}
