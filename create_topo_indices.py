# -*- coding: utf-8 -*-
"""
Created on Mon April  30 10:04:05 2020

@author: broberts based on @shooper 
Inputs: 
search_dir: parent directory containing all of the tile directories 
search_str: dem file (clipped) that serves as the basis for these indices
FA: focal aspect- this can currently only be given as north ('north') or south ('south'). This can be amended to get a wider variety of aspects
"""

import os
import sys
import time
import fnmatch
import numpy as np
from multiprocessing import Pool
from osgeo import gdal
#source: Barnes, Richard. 2016. RichDEM: Terrain Analysis Software. http://github.com/r-barnes/richdem
import richdem as rd 
from lthacks_py3 import createMetadata
from useful_functions import resample

def create_slope_aspect(input_file): 
    arr = rd.LoadGDAL(input_file,no_data=-9999)#rd.rdarray(input_file,no_data=-9999)
    aspect = rd.TerrainAttribute(arr,attrib='aspect')
    slope = rd.TerrainAttribute(arr,attrib='slope_radians')
    aspect_output = input_file[:-4]+'_aspect.tif'
    slope_output = input_file[:-4]+'_slope.tif'
    rd.SaveGDAL(aspect_output, aspect)
    rd.SaveGDAL(slope_output, slope)
    return aspect_output,slope_output

# def resample(input_file,resolution,temp_dest): 
#     """Downscale resolution for faster processing. DO NOT operate on full dataset."""
#     head,tail = os.path.split(input_file)
#     output_file=temp_dest+tail[:-4]+'_temp.tif'
#     #output_file = input_file[:-4]+'_temp.tif'
#     ds = gdal.Open(input_file)
#     ds = gdal.Warp(output_file, input_file, xRes=resolution, yRes=resolution)
#     #ds.Close()
#     ds = None
#     return output_file 

def aspect_intensity(input_file,FA):
    """Create raster of aspect intensity. Concept from Kirchner, P. et al (2014) "LiDAR measurement of seasonal snow accumulation along an elevation gradient in the southern Sierra Nevada, California", Hydrology and 
    Earth System Sciences, 18,10 pp. 4261-4275."""
    #read in data 
    if os.path.exists(input_file[:-9]+'_aspect.tif') or os.path.exists(input_file[:-9]+'_slope.tif'): 
    	slope = gdal.Open(input_file[:-9]+'_slope.tif')
    	aspect = gdal.Open(input_file[:-9]+'_aspect.tif')
    else: 
    	aspect = gdal.Open(create_slope_aspect(input_file)[0])
    	slope = gdal.Open(create_slope_aspect(input_file)[1]) 
    dem = gdal.Open(input_file)
    #create numpy arrays
    arr_aspect = aspect.ReadAsArray()
    arr_slope = slope.ReadAsArray()
    arr_dem = dem.ReadAsArray()
    tx = dem.GetGeoTransform()
    prj = dem.GetProjection()    
    
    #calculate aspect value- Va = cos(A-FA) where aspect value equals azimuth variable (A) and FA is focal aspect (e.g. FA=0deg is north)
    if FA.lower() == 'north': 
        fill_value=0
        print('processing north')
    elif FA.lower() == 'south':
        fill_value=180
        print('processing south')
    else: 
        print('that is not a valid aspect') 
    aspect_deg = np.full(shape=(arr_aspect.shape),fill_value=57.29578,dtype='float')
    fa_arr =np.full(shape=(arr_aspect.shape),fill_value=fill_value,dtype='float')
    aspect_value = np.cos(np.subtract(np.divide(arr_aspect,aspect_deg),fa_arr))
    #calculate the aspect intensity- Ia = sin(S)*Va where S is the slope angle 
    aspect_intensity = np.multiply((np.sin(arr_slope)),aspect_value)
    #create output filename 
    output_file = input_file[:-9]+f'_aspect_intensity_{FA}_temp.tif'
    #write tif to file 
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_file, aspect_intensity.shape[0], aspect_intensity.shape[1], 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(tx)##sets same geotransform as input
    outdata.SetProjection(prj)##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(aspect_intensity)
    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    #resample (this resolution is to match landsat 30m and is thus hard coded. Change if that is not what you want)
    #outdata = gdal.Warp(output_file,outdata,xRes=30, yRes=30)
    # outdata.FlushCache() ##saves to disk!!
    # outdata.Close()
    outdata = None
    slope = None
    aspect = None
    #ds=None
    #clean up inputs
    try:
        print('destroying...')
        os.remove(input_file[:-4]+'_aspect.tif')
        os.remove(input_file[:-4]+'_slope.tif')
    except OSError:
        print(input_file)
        print("Error while deleting file")

    #write a metadata file
    head,tail = os.path.split(input_file)
    createMetadata(sys.argv,head+'/')
    return None

def curvature(input_file): 
    """Create a curvature layer (combine profile and planform curvature) using richdem."""
    arr = rd.LoadGDAL(input_file)#rd.rdarray(input_file,no_data=-9999)
    curvature = rd.TerrainAttribute(arr, attrib='curvature')
    output_file = input_file[:-9]+'_curvature_temp.tif'
    head,tail = os.path.split(output_file)
    rd.SaveGDAL(output_file, curvature)
    createMetadata(sys.argv,head+'/') #just remove the filename so you are left with the path

def run_funcs(args):
    t0 = time.time()
    #(input_file, FA, resolution,n), n_tiles = args
    (input_file, FA,n), n_tiles = args
    #print(len(args))
    aspect_intensity(input_file,FA) #change functions here to run curvature 
    #curvature(input_file)
    print (f'Transformed aspect for {n} of {n_tiles} tiles: {time.time() -t0} seconds') #% (n, n_tiles, (time.time() -t0))


def main(search_dir,search_str, FA, out_dir=None):
    #print('FA is: ',type(FA))
    t0 = time.time()
    #nodata = int(nodata)
    
    i = 1
    if i < 5: 
        print('i is ', i)
        args = []
        for root, dirs, files in os.walk(search_dir):
            for f in fnmatch.filter(files, search_str):
                path = os.path.join(root, f)
                args.append([path, FA, out_dir])
                i += 1
        n_tiles = len(args)
        args = [[a, n_tiles] for a in args]
        
        pool = Pool(20)
        pool.map(run_funcs, args,1)#'''
        
        print (f'\nFinished in {(time.time() - t0)/60} minutes\n') #% ((time.time() - t0)/60)
    else: 
        print('that is the end')

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
    
    
            