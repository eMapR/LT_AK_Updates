
# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 17:33:11 2020

@author: broberts
"""
#from lthacks_py3.lthacks_py3 import *
import lthacks_py3 as lthacks
import zipfile
import sys
from pathlib import Path
import rasterio
from rasterio.merge import merge
import glob
import os
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from osgeo import gdal
from useful_functions import intersect_polygons
import json 
import geopandas as gpd
import pyParz
from subprocess import call
import subprocess
from fnmatch import fnmatch

def unzip_single(input_file,output_dest): 
	#unzip one file
	with zipfile.ZipFile(input_file, 'r') as zip_ref: #'/vol/v2/ak_glaciers/dem/AK_5m_IFSAR/downloads/AK_5m_IFSAR.zip'
		list_of_files = zip_ref.namelist()
		print(list_of_files)
		zip_ref.extractall(output_dest) 
	return None

# def unzip_files(filepath,output_dest): 
# 	"""Iteratively unzip files in a directory."""
# 	for file in Path(filepath).glob('*.zip'):
# 		print(file) 
# 		with zipfile.ZipFile(file, 'r') as zip_ref:
# 			# list_of_files = zip_ref.namelist()
# 			# print(list_of_files)
# 			zip_ref.extractall(output_dest) #'/vol/v2/ak_glaciers/dem/AK_5m_IFSAR/'
# 	return None

def unzip_files(args): 
	"""Unzip files in a directory."""
	#for file in Path(filepath).glob('*.zip'):
	#print(file) 
	file,output_dest = args
	try: 
		with zipfile.ZipFile(file, 'r') as zip_ref:
			# list_of_files = zip_ref.namelist()
			# print(list_of_files)
			zip_ref.extractall(output_dest) 
	except FileExistsError: 
		print('that folder exists')
		pass
	return None

def create_processing_bounds(input_shp_1,input_shp_2,output_dest,epsg): 
	"""Create shapefiles to restrict the amount of data going into the mosaic functions."""
	return intersect_polygons(input_shp_1,None,input_shp_2,epsg,output_dest)

def resample(input_file,resolution,temp_dest,epsg,region): 
	"""Downscale resolution for faster processing."""
	#input_file,resolution,temp_dest,epsg = args
	head,tail = os.path.split(input_file)
	output_file=temp_dest+f'AK_{resolution}m_{region}_region_resampled_cubic.tif'
	
	if os.path.exists(output_file): 
		pass

	else: 	
		ds = gdal.Open(input_file)
		ds = gdal.Warp(output_file, input_file, xRes=resolution, yRes=resolution, dstSRS=epsg,resampleAlg='cubic')
		print('made file')
	#ds.Close()
	ds = None
	return output_file 

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def make_vrts(filepath,output_dest,shapefile,epsg,field,region): 
	"""Create vrt file of all tiles in a repo and then gdal_translate that to a tif file."""
	dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]

	if not shapefile: 
		pass
	else: 
		print('filtering inputs to mosaic....')
		features = gpd.read_file(shapefile)[field].tolist()
		#features = [i[:3]+'*'+i[3:] for i in features]
		dem_out = []

		for i in dem_fps: 
			for e in features: 
				pattern1 = '*'+e[:3]+'*'+e[3:]+'*'
				pattern2 = '*'+e[:3].lower()+'*'+e[3:].lower()+'*'
				if fnmatch(i,pattern1):# in i: 
					dem_out.append(i)
					#print(i)
					#print(e)
				elif fnmatch(i,pattern2):# in i: 
					dem_out.append(i)
					#print(i)
					#print(e.lower())
				else: 
					continue

	#dem_chunks=list(chunks(dem_out,250))
	output_vrt = output_dest+f'AK_5m_original_mosaic_{region}_region.vrt'
	output_tif = output_dest+f'AK_5m_original_mosaic_{region}_region.tif'

	#input_files = filepath+'*.tif' #[str(file) for file in Path(filepath).glob('*.tif')]

	cmd = gdal.BuildVRT(output_vrt, dem_out, outputSRS = epsg,allowProjectionDifference=True,srcNodata=-10000) 
	#make a tif from the vrt file 
	#print('making the tif file')
	#ds = gdal.Translate(output_tif,output_vrt,format='GTiff',outputSRS=epsg)

	#close the datasets
	#ds = None
	cmd = None

	return 

# def make_mosaic(filepath,output_dest,resolution,temp_dest,shapefile,field,epsg): 
# 	"""Create a mosaic of dem tif tiles using rasterio. Take from https://automating-gis-processes.github.io/CSC/notebooks/L5/raster-mosaic.html"""
	
# 	#make a list of files to mosaic
# 	dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]
# 	dem_chunks=list(chunks(dem_fps,250))
# 	count=0
# 	for chunk in dem_chunks: 
# 		src_files_to_mosaic = []
# 		out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_rgi_original_res_{count}_nodata.tif"
# 		# Iterate over raster files and add them to source -list in 'read mode'
# 		for fp in chunk:
# 			print(fp)
# 			src = rasterio.open(fp)#resample(fp,resolution,temp_dest,epsg))#resample(fp,upscale_factor)#
# 			src_files_to_mosaic.append(src)
# 		# Merge function returns a single mosaic array and the transformation info
# 		mosaic, out_trans = merge(src_files_to_mosaic)
# 		# Copy the metadata
# 		out_meta = src.meta.copy()
# 		# Update the metadata
# 		out_meta.update({"driver": "GTiff",
# 		                 "height": mosaic.shape[1],
# 		                 "width": mosaic.shape[2],
# 		                 "transform": out_trans,
# 		                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 		                 }
# 		                )
# 		# Write the mosaic raster to disk
# 		with rasterio.open(out_fp, "w", **out_meta) as dest:
# 		    dest.write(mosaic)
# 		count += 1
# 	return out_fp

def remove_temp_files(input_dest): 
	"""Remove temp files from the resample function."""
	# get a recursive list of file paths that matches pattern including sub directories
	fileList = [str(file) for file in Path(input_dest).glob('*temp.tif')]#, recursive=False
	# Iterate over the list of filepaths & remove each file.
	for file in fileList:
	    try:
	        print('deleting...')
	        os.remove(file)
	    except OSError:
	        print("Error while deleting file")
	return None
#run functions 
def main(): 
	params = sys.argv[1]
	print(params)
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		unzip_filepath = variables["unzip_filepath"]
		in_filepath = variables["in_filepath"]
		out_filepath = variables["out_filepath"]
		temp_dest = variables["temp_dest"]
		input_shp_1 = variables["input_shp_1"]
		input_shp_2 = variables["input_shp_2"]
		epsg = variables["epsg"]
		region = variables["region"] 
		resolution = variables["resolution"]
	
	#unzip one large file (e.g. original download)
	#unzip_single(unzip_filepath,out_filepath)

	#unzip all zipped files in a directory (not recursive)
	#files = [str(file) for file in Path(unzip_filepath).glob('*.zip')]
	#list(Path(unzip_filepath).glob('*.zip'))
	#unzip = pyParz.foreach(files,unzip_files,args=[in_filepath],numThreads=20)
	#unzip_files(unzip_filepath,out_filepath)

	#create shapefiles to restrict the number of files that go into further processing steps
	#create_processing_bounds(input_shp_1,input_shp_2,out_filepath,epsg)

	#can be called separately but is otherwise called as part of the mosaic function 
	resample(in_filepath,resolution,out_filepath,epsg,region)

	#do the mosaicking 
	#output=make_mosaic(in_filepath,out_filepath,15,temp_dest,input_shp_1,'GEOCELL_ID',epsg) #upscale factor should be your target resolution 
	#make_vrts(in_filepath,out_filepath,input_shp_1,epsg,'GEOCELL_ID',region)
	#mosaic_mosaics(out_filepath,15)
	#large_mosaic(output)
	#clean up 
	#remove_temp_files(temp_dest)
	
	#create some metadata
	#lthacks.createMetadata(sys.argv, out_filepath)	

if __name__ == '__main__': 
    main()
    #params = sys.argv[1]
    #sys.exit(main(params)) #'''


##############################

#working
#ds = gdal.Warp(output_tif,output_vrt,format='GTiff',cropToCutline=True,srcSRS=epsg,dstSRS=epsg,cutlineLayer=shapefile) #'gdalwarp -of GTiff -dstnodata -10000 -cutline -s_srs' + ' EPSG:3338 ' +shapefile+  ' -crop_to_cutline -dstalpha ' + output_vrt + ' ' + output_tif

#('gdalbuildvrt ' + output_vrt + ' ' + dem_out#-projwin '+coords+' -of GTiff '+inFile+' '+outFile  
# def mosaic_mosaics(output_dest,resolution): 

# 	files=list([str(Path(output_dest)/'AK_15m_IFSAR_mosaic_0_nodata.tif'),str(Path(output_dest)/'AK_15m_IFSAR_mosaic_1_nodata.tif'),str(Path(output_dest)/'AK_15m_IFSAR_mosaic_2_nodata.tif')])
# 	out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_rgi_tiles_all_qgis_final.tif"
# 	src_files_to_mosaic = []
# 	for i in files: 
# 		src = rasterio.open(i)#resample(fp,upscale_factor)#
# 		src_files_to_mosaic.append(src)
# 	mosaic, out_trans = merge(src_files_to_mosaic)
# 	# Copy the metadata
# 	out_meta = src.meta.copy()
# 	# Update the metadata
# 	out_meta.update({"driver": "GTiff",
# 	                 "height": mosaic.shape[1],
# 	                 "width": mosaic.shape[2],
# 	                 "transform": out_trans,
# 	                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 	                 }
# 	                )
# 	# Write the mosaic raster to disk
# 	with rasterio.open(out_fp, "w", **out_meta) as dest:
# 	    dest.write(mosaic)

# def make_mosaic(filepath,output_dest,resolution,temp_dest,shapefile,field,epsg): 
# 	"""Create a mosaic of dem tif tiles. Take from https://automating-gis-processes.github.io/CSC/notebooks/L5/raster-mosaic.html"""
	
# 	#make a list of files to mosaic
# 	dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]
# 	print('len of file list is: ',len(dem_fps))
# 	#filter the list to a shapefile 
# 	if not shapefile: 
# 		pass
# 	else: 
# 		print('filtering inputs to mosaic....')
# 		features = gpd.read_file(shapefile)[field].tolist()
# 		#features = [i[:3]+'*'+i[3:] for i in features]
# 		dem_out = []

# 		for i in dem_fps: 
# 			for e in features: 
# 				pattern1 = '*'+e[:3]+'*'+e[3:]+'*'
# 				pattern2 = '*'+e[:3].lower()+'*'+e[3:].lower()+'*'
# 				if fnmatch(i,pattern1):# in i: 
# 					dem_out.append(i)
# 					#print(i)
# 					#print(e)
# 				elif fnmatch(i,pattern2):# in i: 
# 					dem_out.append(i)
# 					#print(i)
# 					#print(e.lower())
# 				else: 
# 					continue 
# 	#create resampled files
# 	resampled = [resample(i,resolution,temp_dest,epsg) for i in dem_out]
# 	# for i in dem_out: 
# 	# 	resample(input_file,resolution,temp_dest,epsg)
# 	#resampled = pyParz.foreach(dem_out,resample,args=[resolution,temp_dest,epsg],numThreads=20)
# 	#print(type(resampled))

# 	#make the mosaic 
# 	print('making mosaic')
	
# 	file_list = glob.glob(temp_dest+"*.tif")
# 	files_string = " ".join(file_list)
# 	#middle_index = len(files_string)/2

# 	dem_chunks=list(chunks(files_string,800))

# 	#first_half = files_string[:middle_index]
# 	#second_half = files_string[middle_index:]
# 	count = 0 
# 	for i in dem_chunks: 
# 		out_fp = str(Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_rgi_tiles_final_{count}.tif")
# 		print(out_fp)
# 		cmd = "gdal_merge.py -o " + out_fp + " -of gtiff " + i
# 		subprocess.call(cmd, shell=True)
# 		count += 1 
	
# 		#dem_out = list(set([i for e in features for i in dem_fps if e[:3] and e[3:] or e[:3].lower() and e[:3].lower() in i]))
# 		#print(features[0][3:])
# 		#print('len features is', len(features))
# 		#print(features)
# 		#print('len dem out is: ',len(dem_out))
# 		#print(dem_out[:1000])
# 		#print(dem_out[:5])
# 	# dem_chunks=list(chunks(dem_out,round(len(dem_out)/3)))
# 	# count = 0
# 	# for chunk in dem_chunks: 
# 	# 	src_files_to_mosaic = []
# 	# 	out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_rgi_tiles_{count}.tif"
# 	# 	# Iterate over raster files and add them to source -list in 'read mode'
# 	# 	print('resampling...')
		 
# 	# 	for fp in chunk:
# 	# 	    print(fp)
# 	# 	    src = rasterio.open(resample(fp,resolution,temp_dest,epsg))#resample(fp,upscale_factor)#
# 	# 	    src_files_to_mosaic.append(src)
# 	# 	    #src.close()
# 	# 	# Merge function returns a single mosaic array and the transformation info
# 	# 	print('making mosaic...')
# 	# 	mosaic, out_trans = merge(src_files_to_mosaic)
# 	# 	# Copy the metadata
# 	# 	out_meta = src.meta.copy()
# 	# 	# Update the metadata
# 	# 	out_meta.update({"driver": "GTiff",
# 	# 	                 "height": mosaic.shape[1],
# 	# 	                 "width": mosaic.shape[2],
# 	# 	                 "transform": out_trans,
# 	# 	                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 	# 	                 }
# 	# 	                )
# 	# 	# Write the mosaic raster to disk
# 	# 	count += 1 
# 	# 	with rasterio.open(out_fp, "w", **out_meta) as dest:
# 	# 	    dest.write(mosaic)

# 	return dem_out

# # def large_mosaic(filepath): 
# # 	#file_list = glob.glob(filepath+"*.tif")

# # 	#files_string = " ".join(file_list)

# # 	# cmd = "gdal_merge.py -o final_mosaic_rgi.tif -of gtiff " + files_string

# # 	# subprocess.call(cmd, shell=True)
# # 	vrt_cmd = "gdal_merge.py -of VRT -o final_mosaic.vrt"+filepath#filepath+'*.tif' #'gdalbuildvrt mosaic.vrt'+filepath+'*.tif'
	# cmd = 'gdal_translate -of GTiff -co TILED=YES final_mosaic.vrt final_mosaic_from_vrt.tif' #-co "COMPRESS=JPEG" -co "PHOTOMETRIC=YCBCR" -co "TILED=YES" mosaic.vrt mosaic.tif'
# # 	subprocess.call(vrt_cmd, shell=True)
# # 	subprocess.call(cmd, shell=True)


# script = sys.argv[0]
	# # first command line param
	# unzip_filepath = sys.argv[1]
	# #second command line param
	# in_filepath = sys.argv[2]
	# #third command line param
	# out_filepath = sys.argv[3]
	# #fourth command line param
	# temp_dest = sys.argv[4]
	# input_shp_1 = sys.argv[5]
	# input_shp_2 = sys.argv[6]
	# epsg = sys.argv[7]
# def make_mosaic(filepath,output_dest,resolution,temp_dest): 
# 	"""Create a mosaic of dem tif tiles. Take from https://automating-gis-processes.github.io/CSC/notebooks/L5/raster-mosaic.html"""
	
# 	#make a list of files to mosaic
# 	dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]
# 	#cut the list into chunks
# 	dem_chunks=list(chunks(dem_fps,800))
# 	print(len(dem_chunks[0]))
# 	#List for the source files
# 	count=0
# 	for chunk in dem_chunks: 
# 		src_files_to_mosaic = []
# 		out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_full_{count}.tif"
# 		# Iterate over raster files and add them to source -list in 'read mode'
# 		for fp in chunk:
# 		    src = rasterio.open(resample(fp,resolution,temp_dest))#resample(fp,upscale_factor)#
# 		    src_files_to_mosaic.append(src)
# 		# Merge function returns a single mosaic array and the transformation info
# 		mosaic, out_trans = merge(src_files_to_mosaic)
# 		# Copy the metadata
# 		out_meta = src.meta.copy()
# 		# Update the metadata
# 		out_meta.update({"driver": "GTiff",
# 		                 "height": mosaic.shape[1],
# 		                 "width": mosaic.shape[2],
# 		                 "transform": out_trans,
# 		                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 		                 }
# 		                )
# 		# Write the mosaic raster to disk
# 		with rasterio.open(out_fp, "w", **out_meta) as dest:
# 		    dest.write(mosaic)
# 		count += 1
# 	return out_fp
# def mosaic_mosaics(filepath,output_dest):
# 	"""Helper function to create a final mosaic of the intermediate mosaics made with make_mosaic above."""
# 	mosaics = [str(file) for file in Path(filepath).glob('*.tif')]
# 	for fp in mosaics: 
		
# out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_full.tif"
# 	print(out_fp)
# 	dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]
# 	list_chunks = dem_fps
	
# 	# List for the source files
# 	src_files_to_mosaic = []

# 	# Iterate over raster files and add them to source -list in 'read mode'
# 	for fp in dem_fps:
# 	    print('fp is ',fp)
# 	    src = rasterio.open(resample(fp,resolution,temp_dest))#resample(fp,upscale_factor)#
# 	    print('src is ',src)
# 	    src_files_to_mosaic.append(src)

# 	# Merge function returns a single mosaic array and the transformation info
# 	mosaic, out_trans = merge(src_files_to_mosaic)
# 	# Copy the metadata
# 	out_meta = src.meta.copy()

# 	# Update the metadata
# 	out_meta.update({"driver": "GTiff",
# 	                 "height": mosaic.shape[1],
# 	                 "width": mosaic.shape[2],
# 	                 "transform": out_trans,
# 	                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 	                 }
# 	                )
# 	# Write the mosaic raster to disk

# 	with rasterio.open(out_fp, "w", **out_meta) as dest:
# 	    dest.write(mosaic)
# 		# # Plot the result
# 		# plt.plot(mosaic, cmap='terrain')
# 		# plt.show()
# 	#remove temp files 
# 	return out_fp

##
#in development
# class Mosaic(object):
# 	"""Creates mosaics in chunks and then mosaics the mosaics. Inspired by: https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html"""
# 	def __init__(self, filepath, resolution,temp_dest,output_dest):
#         """Return a Customer object whose name is *name* and starting
#         balance is *balance*."""
#         self.filepath = filepath
#         self.resolution = resolution
#         self.temp_dest = temp_dest
#         self.output_dest = output_dest 
 	
#  	def resample(self,input_file,resolution,temp_dest): 
# 		"""Downscale resolution for faster processing. DO NOT operate on full dataset."""
# 		head,tail = os.path.split(input_file)
# 		output_file=temp_dest+tail[:-4]+'_temp.tif'
# 		#output_file = input_file[:-4]+'_temp.tif'
# 		ds = gdal.Open(input_file)
# 		ds = gdal.Warp(output_file, input_file, xRes=resolution, yRes=resolution)
# 		#ds.Close()
# 		ds = None
# 		return output_file 
	
# 	def chunks(self,lst, n):
# 		"""Yield successive n-sized chunks from lst."""
# 		for i in range(0, len(lst), n):
# 			yield lst[i:i + n]

# 	def make_file_list(self,filepath):
# 		dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]
 


# 	def read_write(self,input_list,output_dest,resolution): 
# 		out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_final.tif"
# 		# Merge function returns a single mosaic array and the transformation info
# 		mosaic, out_trans = merge(input_list)
# 		# Copy the metadata
# 		out_meta = src.meta.copy()
# 		# Update the metadata
# 		out_meta.update({"driver": "GTiff",
# 		                 "height": mosaic.shape[1],
# 		                 "width": mosaic.shape[2],
# 		                 "transform": out_trans,
# 		                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 		                 }
# 		                )
# 		# Write the mosaic raster to disk
# 		with rasterio.open(out_fp, "w", **out_meta) as dest:
# 		    dest.write(mosaic)
# 		return out_fp




# def make_mosaic(filepath,output_dest,resolution,temp_dest): 
# 	"""Create a mosaic of dem tif tiles. Take from https://automating-gis-processes.github.io/CSC/notebooks/L5/raster-mosaic.html"""
	
# 	#make a list of files to mosaic
# 	dem_fps = [str(file) for file in Path(filepath).glob('*.tif')]
# 	#cut the list into chunks
# 	dem_chunks=list(chunks(dem_fps,800))
# 	print(len(dem_chunks[0]))
# 	#List for the source files
# 	count=0
# 	for chunk in dem_chunks: 
# 		src_files_to_mosaic = []
# 		out_fp = Path(output_dest)/f"AK_{resolution}m_IFSAR_mosaic_full_{count}.tif"
# 		# Iterate over raster files and add them to source -list in 'read mode'
# 		for fp in chunk:
# 		    src = rasterio.open(resample(fp,resolution,temp_dest))#resample(fp,upscale_factor)#
# 		    src_files_to_mosaic.append(src)
# 		# Merge function returns a single mosaic array and the transformation info
# 		mosaic, out_trans = merge(src_files_to_mosaic)
# 		# Copy the metadata
# 		out_meta = src.meta.copy()
# 		# Update the metadata
# 		out_meta.update({"driver": "GTiff",
# 		                 "height": mosaic.shape[1],
# 		                 "width": mosaic.shape[2],
# 		                 "transform": out_trans,
# 		                 "crs": 'EPSG:3338'#"+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
# 		                 }
# 		                )
# 		# Write the mosaic raster to disk
# 		with rasterio.open(out_fp, "w", **out_meta) as dest:
# 		    dest.write(mosaic)
# 		count += 1
# 	return out_fp
# def mosaic_mosaics(filepath,output_dest):
# 	"""Helper function to create a final mosaic of the intermediate mosaics made with make_mosaic above."""
# 	mosaics = [str(file) for file in Path(filepath).glob('*.tif')]
# 	for fp in mosaics: 
