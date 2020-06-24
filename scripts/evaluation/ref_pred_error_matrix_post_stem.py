# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 08:49:44 2018

@author: braatenj
"""

from osgeo import gdal, ogr
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import os


def get_dims(fileName):
  src = gdal.Open(fileName)
  print(src)
  print(fileName)
  ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
  sizeX = src.RasterXSize
  sizeY = src.RasterYSize
  lrx = ulx + (sizeX * xres)
  lry = uly + (sizeY * yres)
  return [ulx,uly,lrx,lry,xres,-yres,sizeX,sizeY]

def make_geo_trans(fileName, trgtDim):
  src   = gdal.Open(fileName)
  ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
  return((trgtDim[0], xres, xskew, trgtDim[1], yskew, yres))

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

def get_offsets(fileName, trgtDim):
  dim = get_dims(fileName)
  xoff = math.floor(abs(dim[0]-trgtDim[0])/dim[4])
  yoff = math.ceil(abs(dim[1]-trgtDim[1])/dim[4])
  xsize = abs(trgtDim[0]-trgtDim[2])/dim[4]
  ysize = abs(trgtDim[1]-trgtDim[3])/dim[4]
  return([int(i) for i in [xoff, yoff, xsize, ysize]])

def get_band(fileName, trgtDim, band):
  offsets = get_offsets(fileName, trgtDim)
  src = gdal.Open(fileName)
  band = src.GetRasterBand(band)
  array = band.ReadAsArray(
            offsets[0],
            offsets[1],
            offsets[2],
            offsets[3])
  return(array)

def write_img(outFile, refImg, trgtDim, nBands, dataType, of):
  convertDT = {
    'uint8': 1,
    'int8': 1,
    'uint16': 2,
    'int16': 3,
    'uint32': 4,
    'int32': 5,
    'float32': 6,
    'float64': 7,
    'complex64': 10,
    'complex128': 11
  }
  dataType = convertDT[dataType]
  geoTrans = make_geo_trans(refImg, trgtDim)
  proj = gdal.Open(refImg).GetProjection()
  dims = get_offsets(refImg, trgtDim)
  driver = gdal.GetDriverByName(of)
  driver.Register()
  outImg = driver.Create(outFile, dims[2], dims[3], nBands, dataType) # file, col, row, nBands, dataTypeCode
  outImg.SetGeoTransform(geoTrans)
  outImg.SetProjection(proj)
  return(outImg)



######################################################################################################################

ref = '/vol/v3/ben_ak/raster_files/nlcd/NLCD_2001_epsg_3338_southern_region_30m.tif'
#pred = '/vol/v3/lt_stem_v3.1/models/landcover_20180726_1429/2001/landcover_20180726_1429_2001_vote.tif'
#pred = '/vol/v3/lt_stem_v3.1/models/landcover_20180914_1205/landcover_20180914_1205_vote.tif'
pred = '/vol/v3/ben_ak/param_files_rgi/southern_region/models/2001_06052020_updated_nlcd/06062020_model_run_updated_nlcd.tif'
noDataV = 255
vector = '/vol/v3/ben_ak/vector_files/ak_climate_regions/AK_divisions_epsg_3338_southern_region.shp'
outDir = '/vol/v3/ben_ak/param_files_rgi/southern_region/error_analysis/'
######################################################################################################################


np.set_printoptions(suppress=True)

vDriver = ogr.GetDriverByName("ESRI Shapefile")
vSrc = vDriver.Open(vector, 0)
vLayer = vSrc.GetLayer()
for f, feature in enumerate(vLayer):
  feat = vLayer[f]
  region = feat.GetField('FEATURE')
  print(region)
  geom=feat.GetGeometryRef()
  trgtDim = geom.GetEnvelope()
  trgtDim = [trgtDim[0],trgtDim[3],trgtDim[1],trgtDim[2]]

  refr = get_band(ref, trgtDim, 1).flatten()
  predr = get_band(pred, trgtDim, 1).flatten()
  uniqRef =   np.unique(refr)
  uniqPred =   np.unique(predr)
  uniq = np.unique(np.append(uniqRef, uniqPred))
  noDataR = np.where(uniqRef == noDataV)[0]
  noDataP = np.where(uniqPred == noDataV)[0]


  """
  # test the subsetting
  outRef = os.path.join(outDir, region+'_ref.tif')
  outPred = os.path.join(outDir, region+'_pred.tif')
  outImg = write_img(outRef, ref, trgtDim, 1, 'int8', 'GTIFF')
  outBand = outImg.GetRasterBand(1)
  outBand.WriteArray(get_band(ref, trgtDim, 1))

  outImg = write_img(outPred, pred, trgtDim, 1, 'int8', 'GTIFF')
  outBand = outImg.GetRasterBand(1)
  outBand.WriteArray(get_band(pred, trgtDim, 1))
  outImg = None
  outBand = None
  """
  confuse = (confusion_matrix(refr, predr)).astype('float32')
  if len(noDataR) != 0:
    confuse[noDataR,:] = 0
  if len(noDataP) != 0:
    confuse[:,noDataP] = 0



  confuse = np.append(confuse, np.asmatrix(np.sum(confuse, 0)),0)
  confuse = np.append(confuse, np.asmatrix(np.sum(confuse, 1)),1)
  # add row and col for com om errs
  confuse = np.append(confuse, np.asmatrix(np.sum(confuse, 0)),0)
  confuse = np.append(confuse, np.asmatrix(np.sum(confuse, 1)),1)
  total = confuse.shape[0]-1-1
  fill = confuse.shape[0]-1
  totAcc = 0
  for i in range(total):
    # omission
    confuse[fill,i] = round((confuse[total,i]-confuse[i,i])/(confuse[total,i]+0.0),4)
    # comission
    confuse[i,fill] = round((confuse[i,total]-confuse[i,i])/(confuse[i,total]+0.0),4)
    totAcc += confuse[i,i]

  confuse[fill,fill] = round(totAcc/(confuse[total,total]+0.0), 4)

  colLab = np.concatenate((uniq,[0,0]),0).astype('float32')
  rowLab = np.concatenate(([0.0], colLab))

  confuse = np.append(np.asmatrix(colLab), confuse ,0)
  confuse = np.append(np.transpose(np.asmatrix(rowLab)), confuse ,1)

  outName = os.path.join(outDir, region+'_confusion_matrix.csv')
  np.savetxt(outName, confuse, delimiter=',', fmt='%5.2f')
