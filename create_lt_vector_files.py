# -*- coding: utf-8 -*-
"""
Created on 01/21/2020

@author: broberts

Edit shapefiles for creating the GEE image chunk outputs and for running the clip_and_decompose.py script.
    
"""
import os 
import sys
from osgeo import ogr
import numpy as np 
from subprocess import call
import subprocess
import geopandas as gp
#specify rows and cols wanted for the fishnets 
gee_rows = 3
gee_cols = 14
cnd_rows = 12
cnd_cols = 26

#input bounding file
TILE_PATH = '/vol/v3/ben_ak/vector_files/alaska_bounds_buffer1000_epsg3338.shp'
#path where you want the outputs
OUT_PATH = '/vol/v3/ben_ak/vector_files/'           


def create_gee_vectors(input_tiles,out_file,nrows,ncols): 
	"""Takes a boundary shapefile and creates a fishnet version. Based on make_processing_tiles.py."""
	#create gee file output name
	output_filename = out_file+'ak_tiles_{nrows}_{ncols}.shp'.format(nrows=nrows,ncols=ncols)
	#create cnd file output name
	clipped_output = output_filename[:-4]+'_clipped.shp'
	try: 
		if os.path.exists(output_filename):
			print 'file already exists' 
		else:
			print 'making file'
			subprocess.call(['python', 'make_processing_tiles.py', '{nrows},{ncols}'.format(nrows=nrows,ncols=ncols), '--tile_path',input_tiles, '--out_path', output_filename, '--snap','False'])
	except RuntimeError:
		print 'That file does not exist' 
		pass

	#clip vector file 
	try: 
		#create the name of the output file
		
		if os.path.exists(clipped_output): 
			print 'clipped file exists'
		else: 
			subprocess.call(['ogr2ogr', '-clipsrc', input_tiles, clipped_output, output_filename])
	except: 
		pass
	return clipped_output

def create_cnd_vector(file_path,rows,cols): 
	"""Create a vector with a finer fishnet than the GEE version and edit attributes."""
	#create a clipped vector with a finer fishnet
	# Open a Shapefile, and get field names
	source = ogr.Open(file_path, update=True)
	layer = source.GetLayer()
	layer_defn = layer.GetLayerDefn()
	field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
	new_field_name = 'eetile{rows}x{cols}'.format(rows=rows,cols=cols)
	print field_names
	print new_field_name
	#check to see if the field has been added, if not add the field
	if not new_field_name in field_names: 
		# Add a new field
		new_field = ogr.FieldDefn('eetile{rows}x{cols}'.format(rows=rows,cols=cols), ogr.OFTInteger)
		layer.CreateField(new_field)
		for i in layer: 
			#layer.SetFeature(i)
			i.SetField('new_field_name','example')
			layer.SetFeature(i)
		
	else: 
		print 'That field already exists'

	inFeature = layer.GetNextFeature()
# 	while inFeature:

# 		# get the cover attribute for the input feature
# 		example = inFeature.GetField('new_field_name')

# 		# check to see if cover == grass
# 		if example == 'example':
# 			print "that is there"
# 		else: 
# 			print 'that is not here'

# for feat in layer: 
# 	feat.SetField('Name','myname') 
# 	layer.SetFeature(feat) 
	#dataSource=None 


# field_defn = ogr.FieldDefn( "Area", ogr.OFTReal )
# lyr.CreateField(field_defn)

# for i in lyr:
#     # feat = lyr.GetFeature(i) 
#     geom = i.GetGeometryRef()
#     area = geom.GetArea()
#     print 'Area =', area
#     lyr.SetFeature(i)
#     i.SetField( "Area", area )
#     lyr.SetFeature(i)
# ds = None
	# Close the Shapefile
	source = None


def main(): 
	gee_file = create_gee_vectors(TILE_PATH,OUT_PATH,gee_rows,gee_cols)
	cnd_file = create_gee_vectors(TILE_PATH,OUT_PATH,cnd_rows,cnd_cols)
	create_cnd_vector(cnd_file,gee_rows,gee_cols)

if __name__ == '__main__':
    main()

