#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:32:31 2018

@author: braatenj
"""

import multiprocessing
from osgeo import gdal, ogr, osr
import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import sys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import geopandas
import seaborn as sns
sns.set(style="ticks")

def get_dims(fileName):
  src = gdal.Open(fileName)
  ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
  sizeX = src.RasterXSize
  sizeY = src.RasterYSize
  lrx = ulx + (sizeX * xres)
  lry = uly + (sizeY * yres)
  return [ulx,uly,lrx,lry,xres,-yres,sizeX,sizeY]

def get_intersec(files):
  ulxAll=[]
  ulyAll=[]
  lrxAll=[]
  lryAll=[]
  for fn in files:
    dim = get_dims(fn)
    ulxAll.append(dim[0])
    ulyAll.append(dim[1])
    lrxAll.append(dim[2])
    lryAll.append(dim[3])
  return([max(ulxAll),min(ulyAll),min(lrxAll),max(lryAll)])

def get_zone_pixels(feat, input_zone_polygon, input_value_raster, band, coords=[]): #, raster_band
  """
  feat =feature
  input_zone_polygon = shpF
  input_value_raster = trainF
  band = 1
  coords=[commonBox[0], commonBox[2], commonBox[3], commonBox[1]]
  """
  
  
  
  # Open data
  raster = gdal.Open(input_value_raster)
  shp = ogr.Open(input_zone_polygon)
  lyr = shp.GetLayer()
  
  # Get raster georeference info
  transform = raster.GetGeoTransform()
  xOrigin = transform[0]
  yOrigin = transform[3]
  pixelWidth = transform[1]
  pixelHeight = transform[5]
  
  sizeX = raster.RasterXSize
  sizeY = raster.RasterYSize
  lrx = xOrigin + (sizeX * pixelWidth)
  lry = yOrigin + (sizeY * pixelHeight)
  
  
  
  # Reproject vector geometry to same projection as raster
  #sourceSR = lyr.GetSpatialRef()
  #targetSR = osr.SpatialReference()
  #targetSR.ImportFromWkt(raster.GetProjectionRef())
  #coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
  #feat = lyr.GetNextFeature()
  #geom = feat.GetGeometryRef()
  #geom.Transform(coordTrans)
  
  # Get extent of feat
  geom = feat.GetGeometryRef()
  if (geom.GetGeometryName() == 'MULTIPOLYGON'):
    count = 0
    pointsX = []; pointsY = []
    for polygon in geom:
      geomInner = geom.GetGeometryRef(count)
      ring = geomInner.GetGeometryRef(0)
      numpoints = ring.GetPointCount()
      for p in range(numpoints):
        lon, lat, z = ring.GetPoint(p)
        pointsX.append(lon)
        pointsY.append(lat)
      count += 1
  elif (geom.GetGeometryName() == 'POLYGON'):
    ring = geom.GetGeometryRef(0)
    numpoints = ring.GetPointCount()
    pointsX = []; pointsY = []
    for p in range(numpoints):
      lon, lat, z = ring.GetPoint(p)
      pointsX.append(lon)
      pointsY.append(lat)

  else:
    sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

  #xmin = min(pointsX)  
  #xmax = max(pointsX)
  #ymin = min(pointsY)
  #ymax = max(pointsY)
  
  
  if len(coords) == 0: 
    xmin = xOrigin if (min(pointsX) < xOrigin) else min(pointsX)
    xmax = lrx if (max(pointsX) > lrx) else max(pointsX)
    ymin = lry if (min(pointsY) < lry) else min(pointsY)
    ymax = yOrigin if (max(pointsY) > yOrigin) else max(pointsY)
  else:
    xmin = coords[0] if (min(pointsX) < coords[0]) else min(pointsX)
    xmax = coords[1] if (max(pointsX) > coords[1]) else max(pointsX)
    ymin = coords[2] if (min(pointsY) < coords[2]) else min(pointsY)
    ymax = coords[3] if (max(pointsY) > coords[3]) else max(pointsY)
    
  # Specify offset and rows and columns to read
  xoff = int((xmin - xOrigin)/pixelWidth)
  yoff = int((yOrigin - ymax)/pixelWidth)
  xcount = int((xmax - xmin)/pixelWidth) #+1 !!!!!!!!!!!!!!!!!!!!! This adds a pixel to the right side
  ycount = int((ymax - ymin)/pixelWidth) #+1 !!!!!!!!!!!!!!!!!!!!! This adds a pixel to the bottom side
  
  #print(xoff, yoff, xcount, ycount)
              
  # Create memory target raster
  target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
  target_ds.SetGeoTransform((
    xmin, pixelWidth, 0,
    ymax, 0, pixelHeight,
  ))

  # Create for target raster the same projection as for the value raster
  raster_srs = osr.SpatialReference()
  raster_srs.ImportFromWkt(raster.GetProjectionRef())
  target_ds.SetProjection(raster_srs.ExportToWkt())

  # Rasterize zone polygon to raster
  gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

  # Read raster as arrays
  dataBandRaster = raster.GetRasterBand(band)
  data = dataBandRaster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
  bandmask = target_ds.GetRasterBand(1)
  datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

  # data zone of raster
  dataZone = np.ma.masked_array(data,  np.logical_not(datamask))

  raster_srs = None
  raster = None
  shp = None
  lyr = None
  return [dataZone, [xmin,xmax,ymin,ymax]]



def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]
    uint8 = [int(255*fc) for fc in rgb]
    return '#%02x%02x%02x' % (uint8[0],uint8[1],uint8[2])





def mainFunction(f):

  #############################################################################
  """
  # biomass
  predF = '/vol/v3/lt_stem_v3.1/models/biomassfiaald_20180708_0859/2000/biomassfiaald_20180708_0859_2000_mean.tif'
  trainF = '/vol/v2/datasets/biomass/nbcd/fia_ald/nbcd_fia_ald_biomass_clipped_to_conus.tif'
  shpF = '/vol/v2/datasets/Eco_Level_III_US/us_eco_l3_no_states_multipart.shp'
  trainND = -32768
  predND = -9999
  trgField = 'US_L3CODE'
  descrField = 'US_L3NAME'
  outDir = '/vol/v3/lt_stem_v3.1/evaluation/biomassfiaald_20180708_0859/ecoregion_correlation'
  xyLim = (500, 500)
  xLab = 'Reference (tons/ha)'
  yLab = 'Prediction (tons/ha)'
  annoXY = (15,420)
  """
  
  
  # biomass hexagon
  predF = '/vol/v3/lt_stem_v3.1/models/biomassfiaald_20180708_0859/2000/biomassfiaald_20180708_0859_2000_mean.tif'
  trainF = '/vol/v2/datasets/biomass/nbcd/fia_ald/nbcd_fia_ald_biomass_clipped_to_conus.tif'
  shpF = '/vol/v1/general_files/datasets/spatial_data/hexagons/hexagons_conus_albers_30km_with_id.shp'
  trainND = -32768
  predND = -9999
  trgField = 'id'
  descrField = 'id'
  outDir = '/vol/v3/lt_stem_v3.1/evaluation/biomassfiaald_20180708_0859/hexagon_correlation'
  xyLim = (500, 500)
  xLab = 'Reference (tons/ha)'
  yLab = 'Prediction (tons/ha)'
  annoXY = (15,420)
  
  
  """
  # cc
  predF = '/vol/v3/lt_stem_v3.1/models/canopy_20180915_1631/2001/canopy_20180915_1631_2001_mean.tif'
  trainF = '/vol/v2/stem/conus/reference_rasters/nlcd_2001_canopy_clipped_to_conus_train.tif'
  #shpF = '/vol/v2/datasets/Eco_Level_III_US/us_eco_l3_no_states_multipart.shp'
  shpF = '/vol/v1/general_files/datasets/spatial_data/hexagons/hexagons_conus_albers_30km_with_id.shp'
  trainND = 255
  predND = 255
  trgField = 'id'
  descrField = 'id'
  #trgField = 'US_L3CODE'
  #descrField = 'US_L3NAME'
  #outDir = '/vol/v3/lt_stem_v3.1/evaluation/canopy_20180915_1631/ecoregion_correlation'
  outDir = '/vol/v3/lt_stem_v3.1/evaluation/canopy_20180915_1631/hexagon_correlation'
  xyLim = (100, 100)
  xLab = 'Reference (%)'
  yLab = 'Prediction (%)'
  annoXY = (5,82)
  """
  #############################################################################


  # get color setup
  norm = colors.Normalize(vmin=0, vmax=1)
  f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('YlGnBu_r'))
  
  # open the shapefile	
  vDriver = ogr.GetDriverByName("ESRI Shapefile")
  vSrc = vDriver.Open(shpF, 0)
  vLayer = vSrc.GetLayer()
  
  commonBox = get_intersec([predF, trainF])

#for f in range(vLayer.GetFeatureCount()):
  feature = vLayer[f]
  name = feature.GetField(trgField)
  print('f: '+str(f))
  outFig = os.path.join(outDir, (trgField.replace(' ','_').lower()+'_'+str(name)+'.png'))
  if os.path.exists(outFig):
    #break
    return
    
  descr = feature.GetField(descrField)
  
  predP, coords = get_zone_pixels(feature, shpF, predF, 1, [commonBox[0], commonBox[2], commonBox[3], commonBox[1]])#.compressed() [commonBox[0], commonBox[2], commonBox[3], commonBox[1]]
  trainP, coords = get_zone_pixels(feature, shpF, trainF, 1, [coords[0], coords[1], coords[2], coords[3]])#.compressed()
  
  predP = ma.masked_equal(predP, predND)
  trainP = ma.masked_equal(trainP, trainND)
  trainP = ma.masked_equal(trainP, 0)

  combMask = np.logical_not(np.logical_not(predP.mask) * np.logical_not(trainP.mask))
  predP[combMask] = ma.masked
  trainP[combMask] = ma.masked
  predP = predP.compressed()
  trainP = trainP.compressed()
  if (predP.shape[0] == 0) | (trainP.shape[0] == 0) | (predP==0).all() | (trainP==0).all():
    predP = np.array([0,0,1,1], dtype='float64')
    trainP = np.array([0,0,1,1], dtype='float64')
  mae = round(np.mean(np.absolute(np.subtract(predP, trainP))),1)
  rmse = round(np.sqrt(np.mean((predP-trainP)**2)),1)
  

  totPixs = trainP.shape[0]
  sampSize = round(totPixs*1)
  pickFrom = range(sampSize)
  #sampIndex = np.random.choice(pickFrom, size=sampSize)
  sampIndex = pickFrom

  r = round(np.corrcoef(trainP[sampIndex], predP[sampIndex])[0][1], 2)
  if (mae == 0) & (r == 1):
    r = 0.0
  rColor = f2hex(f2rgb, r)
  p = sns.jointplot(trainP[sampIndex], predP[sampIndex], kind="hex", color='blue', xlim=(0,xyLim[0]), ylim=(0,xyLim[1]), size=5)
  p.ax_joint.set_xlabel(xLab)
  p.ax_joint.set_ylabel(yLab)
  p.ax_joint.annotate('r: '+str(r)+'\nrmse: '+str(rmse)+'\nmae: '+str(mae), annoXY)
  plt.tight_layout()
  outFig = os.path.join(outDir, (trgField.replace(' ','_').lower()+'_'+str(name)+'.png'))
  p.savefig(outFig)
  
  df = pd.DataFrame({'id':name, 'descr':descr, 'r':r, 'rmse':rmse, 'mae':mae, 'color':rColor, 'img':os.path.basename(outFig)}, index=[0])
  outCSV = outFig.replace('.png','.csv')
  df.to_csv(outCSV, ',', index=False)


#################################################################################################

"""
# cc  
#shpF = '/vol/v2/datasets/Eco_Level_III_US/us_eco_l3_no_states_multipart.shp'
shpF = '/vol/v1/general_files/datasets/spatial_data/hexagons/hexagons_conus_albers_30km_with_id.shp'
outDir = '/vol/v3/lt_stem_v3.1/evaluation/canopy_20180915_1631/hexagon_correlation'
"""

# biomass hexagon
shpF = '/vol/v1/general_files/datasets/spatial_data/hexagons/hexagons_conus_albers_30km_with_id.shp'
outDir = '/vol/v3/lt_stem_v3.1/evaluation/biomassfiaald_20180708_0859/hexagon_correlation'



vDriver = ogr.GetDriverByName("ESRI Shapefile")
vSrc = vDriver.Open(shpF, 0)
vLayer = vSrc.GetLayer()
nFeat = vLayer.GetFeatureCount()  

"""
trgField = 'id'
doThese = [i for i in range(nFeat) if not os.path.exists(os.path.join(outDir, (trgField.replace(' ','_').lower()+'_'+str(i)+'.png')))]
len(doThese)
"""

nb_cpus = 20
pool = multiprocessing.Pool(processes=nb_cpus)
results = pool.map(mainFunction, range(nFeat)) #doThese




  
  

