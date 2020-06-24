# -*- coding: utf-8 -*-
"""
Created on Tues May 19 17:33:11 2020

@author: broberts (with some function inputs from jbraaten)

This script is used to collect the first and last day of snow for AK in climate regions and elevation bands. These data are then used to paramaterize the medoid composites that form the basis for the LT algorithm. This is intended to 
minmize conflation of snow and glacier ice in the AK study areas for the AK glaciers project with the NPS. 
Inputs: 

"""

import os
import sys
import pandas as pd 
from osgeo import gdal
import pyParz
import json
import geopandas as gpd


#gdalwarp -cutline INPUT.shp -crop_to_cutline -dstalpha INPUT.tif OUTPUT.tif


def process_climate_regions(climate_regions,column,output_directory):
	"""Read in and process the AK climate regions."""
	with open(tileFile) as f:
		features = json.load(f)[column]
	#create output directory 
	tileOutDir = output_directory+'climate_regions'
	if not os.path.isdir(tileOutDir):
		os.mkdir(tileOutDir)
	# make a list of all the gdal_translate commands needed for the study area
	cmdList = []
	for feature in features:   
		#coords, tileID = get_tile_id_and_coords(feature)
		

	# 	outFile = tileOutDir+'/'+tileID+'_'+name+'_temp.tif'
	# 	inFile = chunkDir+name+'.tif'
	# 	#ds = gdal.Warp(output_file, input_file, xRes=resolution, yRes=resolution)
	# 	cmd = 'gdal_translate -projwin '+coords+' -of GTiff '+inFile+' '+outFile   
	# 	cmdList.append(cmd)  
	# return cmdList
#gdal.Warp(cropToCutline=True,)
def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		climate_regions = variables['climate_regions']
		modis_directory = variables['modis_directory']
		dem = variables['dem']
		column = variables['column']
		output_directory = variables['output_directory']
		process_climate_regions(climate_regions)
if __name__ == '__main__':
	main()

# def run_cmd(cmd):
#   print(cmd)  
#   return subprocess.call(cmd, shell=True)

# def get_tile_id_and_coords(feature):          
#   xmax = feature['properties']['xmax']
#   xmin = feature['properties']['xmin']
#   ymax = feature['properties']['ymax']
#   ymin = feature['properties']['ymin']
#   coords = ' '.join([str(coord) for coord in [xmin,ymax,xmax,ymin]])
   
#   # prepend 0's to tileID so they all have 4 digits
#   tileID = str(feature['properties']['id'])
#   zeros = '0'*(4-len(tileID))
#   tileID = zeros+tileID        
  
#   return (coords, tileID) 

# def make_process_cmds(tileFile,outDir,name,chunkDir): 
#   # load the tile features
#   with open(tileFile) as f:
#     features = json.load(f)['features']
#     print(features)
   
#   # make a list of all the gdal_translate commands needed for the ee conus chunk
#   cmdList = []
#   #ds = gdal.Open(input_file)

#   for feature in features:   
#     coords, tileID = get_tile_id_and_coords(feature)
#     tileOutDir = outDir+tileID
#     if not os.path.isdir(tileOutDir):
#       os.mkdir(tileOutDir)
    
#     outFile = tileOutDir+'/'+tileID+'_'+name+'_temp.tif'
#     inFile = chunkDir+name+'.tif'
#     #ds = gdal.Warp(output_file, input_file, xRes=resolution, yRes=resolution)
#     cmd = 'gdal_translate -projwin '+coords+' -of GTiff '+inFile+' '+outFile   
#     cmdList.append(cmd)  
#   return cmdList
# def main(): 
#   # get the arguments
#   params = sys.argv[1]
#   #print(params)
#   with open(str(params)) as f:
#     variables = json.load(f)
    
#     #construct variables from param file
#     chunkDir = variables["chunkDir"]
#     outDir = variables["outDir"]
#     tileFile = variables["tileFile"]
#     name = variables["name"]
#     nodata = variables["nodata"]
#     proj = variables["proj"]
  
#   # make sure path parts are right
#   if chunkDir[-1] != '/':
#     chunkDir += '/'
#   if outDir[-1] != '/':
#     outDir += '/'
#   #create metadata
#   lthacks.createMetadata(sys.argv, outDir)  

#   # run the commands in parallel 
#   pool = multiprocessing.Pool(processes=20)
#   pool.map(run_cmd, make_process_cmds(tileFile,outDir,name,chunkDir))  
#   pool.close()

# if __name__ == '__main__':
#     main()