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

/**
 * 	This file contains constants needed by the CUDA implementation of the POFToolBox.
 *  @author Thomas Becker
 */

#ifndef DEFINES
#define DEFINES
#endif

/**
 * 	Steps of the impulse responses. Typical values are 85 or 851
 */
#define STEPWIDTH_IR  851

/**
 * 	Steps of the scatter files
 */
#define STEPWIDTH_SC  851

/**
 * 	Number of cells of the matrix in theta z direction
 */
#define STEPWIDTH_TZ  851

/**
 * 	Number of cells of the matrix in theta phi direction
 */
#define STEPWIDTH_TP  98

/**
 * 	Refractive index of the core
 */
#define REFRACTIVE_INDEX_CORE 1.49 

/**
 * 	Refractive group index of the core
 */
#define REFRACTIVE__GROUP_INDEX_CORE 1.51 

/**
 * 	Refractive index of the surrounding material
 */
#define REFRACTIVE_INDEX_SURROUND 1.0

/**
 * 	Maximum angle outside
 */
#define MAX_ANGLE_OUTSIDE 85.0
 
/**
 * 	Speed of light in vacuum
 */
__device__ static double m_dblSpeedOfLight = 299792458.0;

