# -*- coding: utf-8 -*-
"""
Created on Fri May 26 23:52:24 2017

@author: shooper
"""
import gdal
import seaborn as sns
import os
import sys
import pandas as pd

from evaluation import get_samples, histogram_2d

def main(p_path, t_path, nodata_p, nodata_t, sample_txt, out_png):
  
  ds_p = gdal.Open(p_path)
  ar_p = ds_p.ReadAsArray()
  ds_p = None
  
  ds_t = gdal.Open(t_path)
  ar_t = ds_t.ReadAsArray()
  ds_t = None
  
  df = pd.read_csv(sample_txt, sep='\t', index_col='obs_id')
  p_samples, t_samples = get_samples(ar_p, ar_t, df, nodata_p, nodata_t, match=False)
  if not os.path.isdir(os.path.dirname(out_png)):
    os.makedirs(os.path.dirname(out_png))
    
  sns.set_context(context='paper', font_scale=1.4)
  histogram_2d(t_samples, p_samples, out_png, bins=50, hexplot=True, vmax=4000)
  print(out_png)
  ar_p = None
  ar_t = None
  p_samples = None
  t_samples = None

if __name__ == '__main__':
  p_path = sys.argv[1]
  t_path = sys.argv[2]
  nodata_p = sys.argv[3]
  nodata_t = sys.argv[4]
  sample_txt = sys.argv[5]
  out_png = sys.argv[6]

  sys.exit(main(p_path, t_path, nodata_p, nodata_t, sample_txt, out_png))

""" 
p_path = /vol/v3/lt_stem_v3.1/models/biomassfiaald_20180708_0859/2000/biomassfiaald_20180708_0859_2000_mean.tif
t_path = /vol/v2/datasets/biomass/nbcd/fia_ald/nbcd_fia_ald_biomass_clipped_to_conus.tif
nodata_p = -9999
nodata_t = -32768
sample_txt = /vol/v3/lt_stem_v3.1/samples/biomassfiaald_proportional_5243184_20180707_1233/biomassfiaald_proportional_5243184_20180707_1233_predictors.txt
out_png = /vol/v3/lt_stem_v3.1/evaluation/biomassfiaald_20180708_0859/pred_vs_train_2d_hist.png
"""

