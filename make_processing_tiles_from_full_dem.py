# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:33:11 2020

@author: broberts adapted from braatenj

Description: this script clips large DEM files into tiles according to the processing tiles used for the clip and decompose step of the STEM workflow. 
Required args: 
#chunkDir = '/vol/v2/conus_tiles/staging/gdrive/' location of your large dem 
#outDir = '/vol/v2/conus_tiles/tiles_topo/' where do you want all your tile-size chunks to land? 
#tileFile = '/vol/v1/general_files/datasets/spatial_data/conus_tile_system/conus_tile_system_15_sub_epsg5070.geojson' shapefile (converted to geojson) that was used for clip and decompose
#name = 'ned_dem_ee_conus_20170501_elevation' base name for the big dem file
#nodata = 'nan'
#proj = 'EPSG:5070' output proj 

"""

import os
import fnmatch
import json
import sys
import subprocess
import multiprocessing
import lthacks_py3 as lthacks
import json 


def run_cmd(cmd):
  print(cmd)  
  return subprocess.call(cmd, shell=True)

def get_tile_id_and_coords(feature):          
  xmax = feature['properties']['xmax']
  xmin = feature['properties']['xmin']
  ymax = feature['properties']['ymax']
  ymin = feature['properties']['ymin']
  coords = ' '.join([str(coord) for coord in [xmin,ymax,xmax,ymin]])
   
  # prepend 0's to tileID so they all have 4 digits
  tileID = str(feature['properties']['id'])
  zeros = '0'*(4-len(tileID))
  tileID = zeros+tileID        
  
  return (coords, tileID) 

def make_process_cmds(tileFile,outDir,name,chunkDir): 
  # load the tile features
  with open(tileFile) as f:
    features = json.load(f)['features']
    print(features)
   
  # make a list of all the gdal_translate commands needed for the ee conus chunk
  cmdList = []
  #ds = gdal.Open(input_file)

  for feature in features:   
    coords, tileID = get_tile_id_and_coords(feature)
    tileOutDir = outDir+tileID
    if not os.path.isdir(tileOutDir):
      os.mkdir(tileOutDir)
    
    outFile = tileOutDir+'/'+tileID+'_'+name+'_temp.tif'
    inFile = chunkDir+name+'.tif'
    #ds = gdal.Warp(output_file, input_file, xRes=resolution, yRes=resolution)
    cmd = 'gdal_translate -projwin '+coords+' -of GTiff '+inFile+' '+outFile   
    cmdList.append(cmd)  
  return cmdList
def main(): 
  # get the arguments
  params = sys.argv[1]
  #print(params)
  with open(str(params)) as f:
    variables = json.load(f)
    
    #construct variables from param file
    chunkDir = variables["chunkDir"]
    outDir = variables["outDir"]
    tileFile = variables["tileFile"]
    name = variables["name"]
    nodata = variables["nodata"]
    proj = variables["proj"]
  
  # make sure path parts are right
  if chunkDir[-1] != '/':
    chunkDir += '/'
  if outDir[-1] != '/':
    outDir += '/'
  #create metadata
  lthacks.createMetadata(sys.argv, outDir)  

  # run the commands in parallel 
  pool = multiprocessing.Pool(processes=20)
  pool.map(run_cmd, make_process_cmds(tileFile,outDir,name,chunkDir))  
  pool.close()

if __name__ == '__main__':
    main()





