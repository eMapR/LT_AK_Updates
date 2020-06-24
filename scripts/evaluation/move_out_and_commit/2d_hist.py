#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:32:31 2018

@author: braatenj
"""

from osgeo import gdal, ogr, osr
import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
sns.set(style="ticks")



def get_zone_pixels(feat, input_zone_polygon, input_value_raster, band): #, raster_band

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

  xmin = min(pointsX)
  xmax = max(pointsX)
  ymin = min(pointsY)
  ymax = max(pointsY)

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
  return dataZone



def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]
    uint8 = [int(255*fc) for fc in rgb]
    return '#%02x%02x%02x' % (uint8[0],uint8[1],uint8[2])

#############################################################################
predF = '/vol/v3/lt_stem_v3.1/models/biomassfiaald_20180708_0859/2000/biomassfiaald_20180708_0859_2000_mean.tif'
trainF = '/vol/v2/datasets/biomass/nbcd/fia_ald/nbcd_fia_ald_biomass_clipped_to_conus.tif'
shpF = '/vol/v2/datasets/Eco_Level_III_US/us_eco_l3_no_states_multipart.shp'
trainND = -32768
predND = -9999
trgField = 'US_L3CODE'
descrField = 'US_L3NAME'
outDir = '/vol/v3/lt_stem_v3.1/evaluation/biomassfiaald_20180708_0859/ecoregion_correlation'
xyLim = (500, 500)






#############################################################################


# get color setup
norm = colors.Normalize(vmin=0, vmax=1)
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))

# open the shapefile	
vDriver = ogr.GetDriverByName("ESRI Shapefile")
vSrc = vDriver.Open(shpF, 0)
vLayer = vSrc.GetLayer()
nFeat = vLayer.GetFeatureCount()
for f, feature in enumerate(vLayer):
  f=0
  feature = vLayer[f]
  name = feature.GetField(trgField)
  print(name)
  descr = feature.GetField(descrField)
  predP = get_zone_pixels(feature, shpF, predF, 1)#.compressed()
  trainP = get_zone_pixels(feature, shpF, trainF, 1)#.compressed()
  
  predP = ma.masked_equal(predP, predND)
  trainP = ma.masked_equal(trainP, trainND)
  trainP = ma.masked_equal(trainP, 0)
  """
  print(np.unique(predP.compressed()))
  print(np.unique(trainP.compressed()))
  
  hist, bin_edges = np.histogram(predP.compressed(), density=True)
  plt.hist(predP.compressed(), bins=bin_edges)
  plt.show()
  
  hist, bin_edges = np.histogram(trainP.compressed(), density=True)
  plt.hist(trainP.compressed(), bins=bin_edges)
  plt.show()
  
  print(predP.shape)
  print(trainP.shape)
  """
  combMask = np.logical_not(np.logical_not(predP.mask) * np.logical_not(trainP.mask))
  predP[combMask] = ma.masked
  trainP[combMask] = ma.masked
  predP = predP.compressed()
  trainP = trainP.compressed()
  mae = round(np.mean(np.absolute(np.subtract(predP, trainP))),1)
  rmse = round(np.sqrt(np.mean((predP-trainP)**2)),1)
  
  
  
  """
  print(predP.shape)
  print(trainP.shape)
  
  hist, bin_edges = np.histogram(predP, density=True)
  plt.hist(predP, bins=bin_edges)
  plt.show()
  
  plt.hist(trainP, bins=bin_edges)
  plt.show()
  """  
  #maxV = np.max([np.max(trainP[sampIndex]), np.max(predP[sampIndex])])
  totPixs = trainP.shape[0]
  sampSize = round(totPixs*0.2)
  pickFrom = range(sampSize)
  sampIndex = np.random.choice(pickFrom, size=sampSize)

  r = round(np.corrcoef(trainP[sampIndex], predP[sampIndex])[0][1], 2)
  rColor = f2hex(f2rgb, r)
  p = sns.jointplot(trainP[sampIndex], predP[sampIndex], kind="hex", color='green', xlim=(0,xyLim[0]), ylim=(0,xyLim[1]), size=5)
  p.ax_joint.set_xlabel('Reference (tons/ha)')
  p.ax_joint.set_ylabel('Prediction (tons/ha)')
  p.ax_joint.annotate('r: '+str(r)+'\nrmse: '+str(rmse)+'\nmae: '+str(mae), (15,420))
  plt.tight_layout()
  outFig = os.path.join(outDir, (name.replace(' ','_')+'_r'+str(r)+'.png').lower())
  p.savefig(outFig)
  
  df = pd.DataFrame({'id':name, 'descr':descr, 'r':r, 'rmse':rmse, 'mae':mae})
  
  
  
  
  
  
  
  
  
  
