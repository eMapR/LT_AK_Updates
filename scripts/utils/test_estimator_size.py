# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:04:33 2017

@author: shooper
"""

import sys
import os
import time
import random
import numpy as np
import pandas as pd
from osgeo import ogr
from sklearn.externals.joblib import Parallel, delayed

package_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(package_dir)
import stem


def intersecting_cells(grid_cells, mosaic_path):
    
    ds = ogr.Open(mosaic_path)
    lyr = ds.GetLayer()
    geom = ogr.Geometry(ogr.wkbMultiPolygon)
    for feature in lyr:
        geom.AddGeometry(feature.GetGeometryRef())
    
    overlapping = []
    for set_id, row in grid_cells.iterrows():
        wkt = 'POLYGON (({0} {1}, {2} {1}, {2} {3}, {0} {3}, {0} {1}))'.format(row.ul_x, row.ul_y, row.lr_x, row.lr_y)
        set_geom = ogr.CreateGeometryFromWkt(wkt)
        set_geom.CloseRings()
        if geom.Intersects(set_geom):
            overlapping.append(set_id)
    grid_cells = grid_cells.ix[overlapping]
    
    ds, lyr, geom, feature = None, None, None, None
    
    return grid_cells


def train_estimator(i, n_sets, start_time, df_train, predict_cols, target_col, support_set, model_function, model_type, max_features, max_val, min_obs=500):
    """ train a single estimator as a member of a queue in parallel """
    
    format_tuple = i + 1, n_sets, float(i)/n_sets * 100, (time.time() - start_time)/60
    sys.stdout.write('\rTraining %s/%s DTs (%.1f%%) || %.1f minutes' % format_tuple)
    sys.stdout.flush()

    # Get all samples within support set
    sample_inds = df_train.index[(df_train['x'] > support_set[['ul_x', 'lr_x']].min()) &
    (df_train['x'] < support_set[['ul_x', 'lr_x']].max()) &
    (df_train['y'] > support_set[['ul_y', 'lr_y']].min()) &
    (df_train['y'] < support_set[['ul_y', 'lr_y']].max())]
    if len(sample_inds) < min_obs:
        return {}
        
    x_train = df_train.ix[sample_inds, predict_cols]
    y_train = df_train.ix[sample_inds, target_col]
    train_inds = random.sample(x_train.index, int(len(sample_inds) * .63))
    oob_inds = x_train.index[~x_train.index.isin(train_inds)]
    bootstrap_x = x_train.ix[train_inds]
    bootstrap_y = y_train.ix[train_inds]
    dt_model = model_function(bootstrap_x, bootstrap_y, max_features)
    
    oob_x = x_train.ix[oob_inds]
    oob_y = y_train.ix[oob_inds]
    oob_metrics = stem.calc_oob_metrics(dt_model, oob_y, oob_x, model_type, max_val)
    oob_metrics['set_id'] = support_set.name
    
    return oob_metrics


def _par_train_estimator(n_jobs, n_sets, df_train, predict_cols, target_col, support_sets, model_func, model_type, max_features, max_target_val):
    
    start_time = time.time()
    s = Parallel(n_jobs, backend="threading")(
            delayed(train_estimator)(i, n_sets, start_time, df_train, predict_cols, target_col, support_set, model_func, model_type, max_features, max_target_val) for i, (si, support_set) in enumerate(support_sets.iterrows()))
    
    return s


def main(params, snap_coord=None, resolution=30, n_sizes=5, max_features=None, n_jobs=1):
    t0 = time.time()
    
    inputs, df_var = stem.read_params(params)
    
    # Convert params to named variables and check for required vars
    for i in inputs:
        exec ("{0} = str({1})").format(i, inputs[i])
    
    try:
        sets_per_cell = int(sets_per_cell)
        cell_size = [int(s) for s in cell_size.split(',')]
        min_size = int(min_size)
        max_size = int(max_size)
    except NameError as e:
        missing_var = str(e).split("'")[1]
        msg = "Variable '%s' not specified in param file:\n%s" % (missing_var, params)
        raise NameError(msg)
        
    # Read in training samples and check that df_train has exactly the same
    #   columns as variables specified in df_vars    
    df_train = pd.read_csv(sample_txt, sep='\t')
    n_samples = len(df_train)
    unmatched_vars = [v for v in df_var.index if v not in [c for c  in df_train]]
    if len(unmatched_vars) != 0:
        unmatched_str = '\n\t'.join(unmatched_vars)
        msg = 'Columns not in sample_txt but specified in params:\n\t' + unmatched_str
        import pdb; pdb.set_trace()
        raise NameError(msg)
    if target_col not in df_train.columns:
        raise NameError('target_col "%s" not in sample_txt: %s' % (target_col, sample_txt))
    if 'max_target_val' in inputs:
        max_target_val = int(max_target_val)
    else:
        max_target_val = df_train[target_col].max()
    if 'n_jobs' in inputs:
        n_jobs = int(n_jobs)
        
    predict_cols = sorted(np.unique([c for c in df_train.columns for v in df_var.index if v in c]))
    df_var = df_var.reindex(df_var.index.sort_values())# Make sure predict_cols and df_var are in the same order

    if snap_coord:
        snap_coord = [int(c) for c in snap_coord.split(',')]
    
    t1 = time.time()
    if model_type.lower() == 'classifier':
        model_func = stem.fit_tree_classifier
    else:
        model_func = stem.fit_tree_regressor        
    
    # Make grid
    x_res = resolution
    y_res = -resolution
    tx, extent = stem.tx_from_shp(mosaic_path, x_res, y_res, snap_coord=snap_coord)
    min_x, max_x, min_y, max_y = [int(i) for i in extent]
    cells = stem.generate_gsrd_grid(cell_size, min_x, min_y, max_x, max_y, x_res, y_res)
    grid = pd.DataFrame(cells, columns=['ul_x', 'ul_y', 'lr_x', 'lr_y'])
    grid.to_csv(out_txt.replace('.txt','_grid.txt'))
    #import pdb; pdb.set_trace()
    grid = intersecting_cells(grid, mosaic_path)
    stem.coords_to_shp(grid, '/vol/v2/stem/extent_shp/CAORWA.shp', out_txt.replace('.txt','_grid.shp'))
    
    if 'set_sizes' in inputs:
        set_sizes = np.sort([int(s) for s in set_sizes.split(',')])
    else:
        if 'n_sizes' in inputs:
            n_sizes = int(n_sizes)
        set_sizes = np.arange(min_size, max_size + 1, (max_size - min_size)/n_sizes)

    
    # Sample grid
    dfs = []
    for i, cell in grid.iterrows():
        ul_x, ul_y, lr_x, lr_y = cell
        min_x, max_x = min(ul_x, lr_x), max(ul_x, lr_x)
        min_y, max_y = min(ul_y, lr_y), max(ul_y, lr_y)
        
        # Calculate support set centers 
        x_centers = [int(stem.snap_coordinate(x, snap_coord[0], x_res)) for x in random.sample(xrange(min_x, max_x + 1), sets_per_cell)]
        y_centers = [int(stem.snap_coordinate(y, snap_coord[1], y_res)) for y in random.sample(xrange(min_y, max_y + 1), sets_per_cell)]

        for size in set_sizes:
            df = stem.sample_gsrd_cell(sets_per_cell, cell, size, size, x_res, y_res, tx, snap_coord, center_coords=(zip(x_centers, y_centers)))
            df['set_size'] = size
            df['cell_id'] = i
            dfs.append(df)
    
    support_sets = pd.concat(dfs, ignore_index=True)
    n_sets = len(support_sets)
    #import pdb; pdb.set_trace()
    print 'Testing set sizes with %s jobs...\n' % n_jobs
    oob_metrics = _par_train_estimator(n_jobs, n_sets, df_train, predict_cols, target_col, support_sets, model_func, model_type, max_features, max_target_val)
    '''args = [[i, n_sets, start_time, df_train, predict_cols, target_col, support_set, model_func, model_type, max_features, max_target_val] for i, (si, support_set) in enumerate(support_sets.ix[:100].iterrows())]
    oob_metrics = []
    for arg in args:
        oob_metrics.append(par_train_estimator(arg))'''

    oob_metrics = pd.DataFrame(oob_metrics)
    oob_metrics.set_index('set_id', inplace=True)
    support_sets = pd.merge(support_sets, oob_metrics, left_index=True, right_index=True)
    #import pdb; pdb.set_trace()
    support_sets.to_csv(out_txt)
            
if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
        