# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:12:14 2018

@author: shooper
"""

import sys, os
import gdal
import pandas as pd

package_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(package_dir)
import extract_xy_by_mosaic as extract
from get_stratified_random_pixels import parse_bins
from lthacks import attributes_to_df
import stem


MOSAIC_SHP = '/vol/v1/general_files/datasets/spatial_data/conus_tile_system/conus_tile_system_15_sub_epsg5070.shp'


def get_point_dict(df, psu_ids):
    return {tile_id: df.loc[df.tile_id == tile_id].index for tile_id in psu_ids}

def main(txt, n_sample, out_txt, bins, train_params, by_psu=True, extract_predictors=True):
    
    n_sample = int(n_sample) 
    bins = parse_bins(bins)
    
    df = pd.read_csv(txt, sep='\t', dtype={'tile_id': object})
    sample = pd.DataFrame(columns=df.columns)
    n_bins = len(bins)
    psu_ids = df.tile_id.unique()
    
    train_params = stem.read_params(train_params)
    for var in train_params:
        exec ("{0} = str({1})").format(var, train_params[var])
    tiles = attributes_to_df(MOSAIC_SHP)
    
    if extract_predictors:
        var_info = pd.read_csv(var_info, sep='\t', index_col='var_name')
        for i, tile in enumerate(psu_ids):
            print("extracting %s of %s" % (i, len(psu_ids)))
            sample_mask = df.tile_id == tile
            this_sample = df.loc[sample_mask]
            tile_ul = tiles.loc[tiles['name'] == tile, ['xmin', 'ymax']].values[0]
            #point_dict = get_point_dict(df, psu_ids)
            mosaic_tx, extent = stem.tx_from_shp(MOSAIC_SHP, 30, -30)
            
            row_off, col_off = stem.calc_offset([mosaic_tx[0], mosaic_tx[3]], tile_ul, mosaic_tx)
            this_sample['local_row'] = this_sample.row - row_off
            this_sample['local_col'] = this_sample.col - col_off
    
            for var_name, var_row in var_info.iterrows():
                #tiles = pd.DataFrame({'tile_id': psu_ids, 'tile_str': psu_ids})
                file_path = stem.find_file(var_row.basepath, var_row.search_str, tile)
                ds = gdal.Open(file_path)
                ar = ds.GetRasterBand(var_row.data_band).ReadAsArray()
                try:
                    if len(this_sample) == ar.size:
                        df.loc[sample_mask, var_name] = ar.ravel()
                    else:
                        df.loc[sample_mask, var_name] = ar[this_sample.local_row, this_sample.local_col]
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
                ds = None
        df.to_csv(txt.replace('.txt', '_predictors.txt'))
    #df[var_name], _ = extract.extract_var('', var_name, var_row.by_tile, var_row.data_band, var_row.data_type, tiles, df, point_dict, var_row.basepath, var_row.search_str, var_row.path_filter, mosaic_tx, 0, 0, silent=True)
                
    if by_psu: 
        
        n_per_psu = n_sample/len(psu_ids)
        n_per_bin = n_per_psu/n_bins
        
        for i, pid in enumerate(psu_ids):
            psu_pixels = df.loc[df.tile_id == pid]
            print("Sampling for %s of %s PSUs" % (i + 1, len(psu_ids)))
            for l, u in bins:
                this_bin = psu_pixels.loc[(l < psu_pixels.value) & (psu_pixels.value <= u)]
                if len(this_bin) > 0:
                    bin_sample_size = min(n_per_bin, len(this_bin))
                    sample = pd.concat([sample, this_bin.sample(bin_sample_size)])
                    print("Sampled %s for bin %s-%s" % (n_per_bin, l, u))
                else:
                    print("No pixels between %s and %s found" % (l, u))
            print("")
    
    else:
        n_per_bin = n_sample/n_bins
        for l, u in bins:
            sample = pd.concat([sample, df.sample(n_per_bin)])
    
    sample.to_csv(out_txt, index=False)
    
    print 'Sample written to ', out_txt
    

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))