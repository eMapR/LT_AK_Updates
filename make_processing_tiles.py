# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:28:45 2018

@author: shooper

Make a shapefile of tiles to process LandTrendr runs on Google Earth Engine. 
Optionally append a column to the tile_path given indicating for each feature
which processing tile it falls within.

Usage:
    make_processing_tiles.py <n_tiles> [--tile_path=<str>] [--add_field=<bool>] [--out_path=<str>] [--snap=<bool>]
    make_processing_tiles.py -h | --help

Options:
    -h --help           Show this screen.
    --tile_path=<str>   Path to a vector file of tiles that correspond to files
                        stored locally. Default is /vol/v1/general_files/datasets/spatial_data/conus_tile_system/conus_tile_system_15_sub_epsg5070.shp
    --add_field=<bool>  Boolean indicating whether a field should be appended to
                        tile_path with the ID of the overlapping processing tile
    --out_path=<str>    Path of the output vector file. If not specified, the 
                        file will be written to /vol/v1/general_files/datasets/spatial_data/conus_tile_system/ee_conus_chunks/ with the filename "ee_processing_tiles_<nrows>x<ncols>.shp
    --snap=<bool>       Boolean indicating whether tiles should be snapped to the nearest x,y coordinate from any vertex of any feature in tile_path. If False, tiles will be divided equally and may not align with borders of features in tile_path.[default: True]
    
Examples:
    python make_processing_tiles.py "1,3"
    python make_processing_tiles.py "2,10" --snap=False
    
"""

import os, sys, docopt, re
from osgeo import ogr
import numpy as np

sys.path.append('/vol/v2/stem/stem-git/scripts')
from stem import get_tiles, coords_to_shp, tx_from_shp, get_overlapping_sets, get_coords
from lthacks import attributes_to_df, df_to_shp, createMetadata

TILE_PATH = '/vol/v3/ben_ak/vector_files/final_vectors/tiles_for_cnd.shp'
XRES = 30
YRES = -30
OUT_DIR = '/vol/v3/ben_ak/vector_files/final_vectors/'


def main(n_tiles, tile_path=None, add_field=True, out_path=None, snap=True, clip=True):
    
    try:
        if add_field.lower() == 'false':
            add_field = False
    except:
        pass
    try:
        if snap.lower() == 'false':
            snap = False
    except:
        pass
        
    if tile_path is None:
        tile_path = TILE_PATH
    
    if not os.path.exists(tile_path):
        raise RuntimeError('tile_path does not exist: %s' % tile_path)
    
    try:
        n_tiles = tuple([int(i) for i in n_tiles.split(',')])
    except:
        raise ValueError('Could not parse n_tiles %s. It must be given as "n_tiles, n_x_tiles"' % n_tiles)
        
    # Get processing tiles
    tx, (xmin, xmax, ymin, ymax) = tx_from_shp(tile_path, XRES, YRES)
    xsize = abs(int(xmax - xmin)/XRES)
    ysize = abs(int(ymax - ymin)/YRES)
    tiles, _, _ = get_tiles(n_tiles, xsize, ysize, tx=tx)
    tile_id_field = 'eetile%sx%s' % n_tiles
    tiles[tile_id_field] = tiles.index
    
    if snap:
        coords, _ = get_coords(tile_path, multipart='split')
        coords = np.array(coords)#shape is (nfeatures, ncoords, 2)
        xcoords = np.unique(coords[:,:,0])
        ycoords = np.unique(coords[:,:,1])
        for i, processing_coords in tiles.iterrows():
            tiles.loc[i, 'ul_x'] = xcoords[np.argmin(np.abs(xcoords - processing_coords.ul_x))]
            tiles.loc[i, 'lr_x'] = xcoords[np.argmin(np.abs(xcoords - processing_coords.lr_x))]
            tiles.loc[i, 'ul_y'] = ycoords[np.argmin(np.abs(ycoords - processing_coords.ul_y))]
            tiles.loc[i, 'lr_y'] = ycoords[np.argmin(np.abs(ycoords - processing_coords.lr_y))]
    
    if not out_path:
        out_path = os.path.join(OUT_DIR, 'ee_processing_tiles_%sx%s.shp' % n_tiles)
    coords_to_shp(tiles, tile_path, out_path)
    descr = ('Tiles for processing data on Google Earth Engine. The tiles ' + 
            'have %s row(s) and %s col(s) and are bounded by the extent of %s') %\
            (n_tiles[0], n_tiles[1], tile_path)
    
    '''if clip:
        ds = ogr.Open(tile_path)
        lyr = ds.GetLayer()
        geoms = ogr.Geometry(ogr.wkbMultiPolygon)
        for feature in lyr:
            g = feature.GetGeometryRef()
            geoms.AddGeometry(g)
        union = geoms.UnionCascaded()
        base_path, ext = os.path.splitext(tile_path)
        temp_file = tile_path.replace(ext, '_uniontemp' + ext)
        feature'''
    
    
    createMetadata(sys.argv, out_path, description=descr)
    print '\nNew processing tiles written to', out_path
    
    # Find which features processing tile touches which each CONUS storage tile
    #   use get_overallping_sets() to find which
    # Read in the CONUS storage tiles
    if add_field:
        conus_tiles = attributes_to_df(tile_path)
        
        
        # Make a temporary copy of it 
        base_path, ext = os.path.splitext(tile_path)
        temp_file = tile_path.replace(ext, '_temp' + ext)
        df_to_shp(conus_tiles, tile_path, temp_file, copy_fields=False)
        
        # Loop through each processing tile and find all overlapping
        conus_tiles[tile_id_field] = -1
        ds = ogr.Open(tile_path)
        lyr = ds.GetLayer()
        for p_fid, processing_coords in tiles.iterrows():
            wkt = 'POLYGON (({0} {1}, {2} {1}, {2} {3}, {0} {3}, {0} {1}))'.format(processing_coords.ul_x, processing_coords.ul_y, processing_coords.lr_x, processing_coords.lr_y)
            p_geom = ogr.CreateGeometryFromWkt(wkt)
            p_geom.CloseRings()
            for c_fid in conus_tiles.index:
                feature = lyr.GetFeature(c_fid)
                geom = feature.GetGeometryRef()
                if geom.Intersection(p_geom).GetArea() > 0:
                    conus_tiles.loc[c_fid, tile_id_field] = p_fid
        lyr, feature = None, None

        # re-write the CONUS tiles shapefile with the new field
        df_to_shp(conus_tiles, tile_path, tile_path, copy_fields=False)
        
        # delete temporary file
        driver = ds.GetDriver()
        driver.DeleteDataSource(temp_file)
        ds = None
        print '\nField with processing tile ID added to', tile_path
    
        # if the metadata text file exists, add a line about appending the field.
        #   otherwise, make a new metadata file.
        meta_file = tile_path.replace(ext, '_meta.txt')
        if os.path.exists(meta_file):
            with open(meta_file, 'a') as f:
                f.write('\n\nAppended field %s with IDs from the overlapping feature of %s' % (tile_id_field, out_path))
        else:
            descr = 'Tile system with appended field %s with IDs from the overlapping feature of %s' % (tile_id_field, out_path)
            createMetadata(sys.argv, tile_path, description=descr)


if __name__ == '__main__':
    
    cl_args = {k: v for k, v in docopt.docopt(__doc__).iteritems() if v is not None}
    #cl_args = docopt.docopt(__doc__)
    #cl_args = {k:v for k, v in cl_args.iteritems() for arg in sys.argv[1:] if k in arg}
    
    # get rid of extra characters from doc string and 'help' entry
    args = {re.sub('[<>-]*', '', k): v for k, v in cl_args.iteritems()
            if k != '--help' and k != '-h'}
            
    sys.exit(main(**args))
        
    
        
        