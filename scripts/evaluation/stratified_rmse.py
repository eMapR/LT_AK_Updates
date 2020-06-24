# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:04:47 2018

@author: shooper
"""

import sys, os, warnings
from osgeo import gdal
import pandas as pd
import numpy as np
from evaluation import get_samples, area_weighted_rmse

package_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(package_dir)
from get_stratified_random_pixels import parse_bins
import mosaic_by_tsa as mosaic


def main(sample_txt, ref_raster, pred_raster, p_nodata, t_nodata, target_col, bins, out_txt, match=None, predict_col=None):
    
    p_nodata = int(p_nodata)
    t_nodata = int(t_nodata)
    
    ds_p = gdal.Open(pred_raster)
    ar_p = ds_p.ReadAsArray()
    
    ds_r = gdal.Open(ref_raster)
    ar_r = ds_r.ReadAsArray()
    
    r_xsize = ds_r.RasterXSize
    r_ysize = ds_r.RasterYSize
    p_xsize = ds_p.RasterXSize
    p_ysize = ds_p.RasterYSize
    tx_r = ds_r.GetGeoTransform()
    tx_p = ds_p.GetGeoTransform()
    # If two arrays are different sizes, make prediction array match reference
    if not r_xsize == p_xsize or r_ysize == p_ysize or tx_r != tx_p:
        warnings.warn('Prediction and reference rasters do not share the same extent. Snapping prediction raster to reference....')
        offset = mosaic.calc_offset((tx_r[0], tx_r[3]), tx_p)
        t_inds, p_inds = mosaic.get_offset_array_indices((r_ysize, r_xsize), (p_ysize, p_xsize), offset)
        ar_buf = np.full(ar_r.shape, p_nodata, dtype=ar_p.dtype)
        ar_buf[t_inds[0]:t_inds[1], t_inds[2]:t_inds[3]] = ar_p[p_inds[0]:p_inds[1], p_inds[2]:p_inds[3]]
        ar_p = ar_buf.copy()
        del ar_buf
        
    bins = parse_bins(bins)
    
    sample = pd.read_csv(sample_txt, sep='\t')
    if target_col in sample.columns:
        t_sample = sample[target_col]
    else:
        raise IndexError('target_col "%s" not in sample' % target_col)
    
    if match:
        t_sample, p_sample = get_samples(ar_p, ar_r, p_nodata, t_nodata, sample, match=match)
    elif predict_col:
        p_sample = sample[predict_col]
    else:
        p_sample = ar_p[sample.row, sample.col]
        t_sample = ar_r[sample.row, sample.col]
    
    rmse = area_weighted_rmse(ar_p, ar_r, p_sample, t_sample, bins, p_nodata, out_txt=out_txt)
    
    return rmse
    
    

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
    