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
#ifndef TRANSFERANGLE
#define TRANSFERANGLE
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * This struct provides functions to convert angles between two medias of different refractive indices with the help of Snell's law.
 * 
 * @author Thomas Becker
 *
 */
struct TransferAngle
{
	/**
	 * Refractive index of the first material.
	 */
	double m_dblIndex1; 
	
	/**
	 * Refractive index of the second material.
	 */
	double m_dblIndex2;

	/**
	 * This method acts as a constructor and sets the refraftive indices.
	 * 
	 * @param a_dblIndex1	Refractive index of the first material
	 * @param a_dblIndex2	Refractive index of the second material
	 */
	__host__ __device__ void construct(double a_dblIndex1, double a_dblIndex2)
	{
		m_dblIndex1 = a_dblIndex1;
		m_dblIndex2 = a_dblIndex2;
	}

	/**
	 * This method calculates the angle in material one from the given angle in material two.
	 * 
	 * @param a_dblAngle2	The angle in material two
	 * @return				The angle in material one
	 */
	__host__ __device__ double getAngle1(double a_dblAngle2)
	{
		return asin
		(
			m_dblIndex2
			*
			sin(a_dblAngle2)
			/
			m_dblIndex1
		);
	}

	/**
	 * This method calculates the angle in material two from the given angle in material one.
	 * 
	 * @param a_dblAngle1	The angle in material one
	 * @return				The angle in material two
	 */
	__host__ __device__ double getAngle2(double a_dblAngle1) 
    {
        return asin
		(
			m_dblIndex1
			*
			sin(a_dblAngle1)
			/
			m_dblIndex2
		);
    }
};
