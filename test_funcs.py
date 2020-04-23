import os 
import sys
from osgeo import ogr
import numpy as np 
#test issues with create_lt_vector_files.py
write_to_path = '/vol/v3/ben_ak/test_data/alaska_bounds_buffer1000_epsg3338.shp'


# def reproject(input_shape,epsg): 
# 	reprojected_filename = input_shape[:-4]+'_reprojected.shp'
# 	subprocess.call(['ogr2ogr', '-f','ESRI Shapefile', '-t_srs', 'EPSG:{epsg}'.format(epsg=epsg), '-s_srs', 'EPSG:{epsg}'.format(epsg=epsg), reprojected_filename , input_shape])
# 	return reprojected_filename
print write_to_path
source = ogr.Open(write_to_path, update=True)
layer = source.GetLayer()


srs = layer.GetSpatialRef()
print srs

layer_defn = layer.GetLayerDefn()
for i in range(0, layer.GetFeatureCount()):
	# Get the input Feature
	feature = layer.GetFeature(i)
	#feature = layer.GetFeature()
	geom = feature.GetGeometryRef()
	print(geom)
# field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
# new_field_name = 'eetile{rows}x{cols}'.format(rows=2,cols=15)
# print field_names
# print new_field_name


# driver = ogr.Open(input_file).GetDriver()
# datasource = driver.Open(input_file, 0)
# input_layer = datasource.GetLayer()

# dest_srs = ogr.osr.SpatialReference()
# dest_srs.ImportFromEPSG(32632)
# dest_layer = output_data_source.CreateLayer(table_name,
#                             dest_srs,
#                             input_layer.GetLayerDefn().GetGeomType(),
#                             ['OVERWRITE=YES', 'GEOMETRY_NAME=geom', 'DIM=2', 'FID=id')

# # adding fields to new layer
# layer_definition = ogr.Feature(input_layer.GetLayerDefn())
# for i in range(layer_definition.GetFieldCount()):
#     dest_layer.CreateField(layer_definition.GetFieldDefnRef(i))

# # adding the features from input to dest
# for i in range(0, input_layer.GetFeatureCount()):
#     feature = input_layer.GetFeature(i)
#     dest_layer.CreateFeature(feature)