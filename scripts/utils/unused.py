# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:08:38 2017

@author: shooper
"""

        '''aggregated = []
        for i, pixel_inds in enumerate(pxl_inds):
            t2 = time.time()
            aggregated.append(stem.predict_pixel(df_var, tile_str, tile_ul, mosaic_tx, pixel_inds, overlapping_sets, agg_stats))
            print i, time.time() - t2#'''
        '''t2 = time.time()
        args = [(i, n_pixels, t2, df_var, tile_str, tile_ul, mosaic_tx, pxl_inds[i], overlapping_sets, agg_stats, predictors[i].reshape(1, n_predictors)) for i in xrange(n_pixels)]
        for arg in args: stem.par_predict_pixel(arg)'''
        '''print 'Building args: %.1f seconds' % ((time.time() - t2))
        pool = Pool(n_jobs_pred)
        
        t2 = time.time()
        predictions = pd.concat(pool.map(stem.par_predict_pixel, args, 1))
        pool.close()
        pool.join()#'''
        print '\nPredicting: %1.f minutes' % ((time.time() - t2)/60)
        #aggregated = parallel_helper(n_jobs_pred, df_var, tile_str, tile_ul, mosaic_tx, overlapping_sets, agg_stats, pxl_inds)
        print (time.time() - t1)/60
        #import pdb; pdb.set_trace()
        last_estimators = overlapping_sets.dt_model#'''
        
    
    import pdb; pdb.set_trace()
    '''#Aggregate predictions by tile and stitch them back together
    t1 = time.time()
    agg_stats = [s.strip().lower() for s in agg_stats.split(',')]
    if 'n_jobs_agg' in inputs:
        n_jobs_agg = int(n_jobs_agg)
    
    if mosaic_type == 'vector':
        nodata_mask = mosaic_ds
    else:
        if 'mosaic_nodata' in inputs: mosaic_nodata = int(mosaic_nodata)
        nodata_mask = mosaic_ds.ReadAsArray() != mosaic_nodata
    
    #  check for sets that errored - if there are any, remove them from the df_sets DF so that the aggregation step doesn't expect them
    if len(df_sets) != len(glob.glob(os.path.join(predict_dir, '*prediction*.tif'))):
        setErrorLog = os.path.dirname(predict_dir)+'/prediction_errors.txt'    
        if os.path.isfile(setErrorLog):   
          with open(setErrorLog) as f:
            lines = f.readlines()
          badSets = [int(line.split(':')[1].rstrip().strip()) for line in lines if 'set_id' in line]
          df_sets.drop(badSets, inplace=True)#'''
          
        

    
    
'''def predict_pixel(df_var, tile_str, tile_ul, tx, pixel_inds, sets, agg_stats, predictors=None, estimators=None):
    
    t0 = time.time()
    if not isinstance(predictors, np.ndarray):
        predictors = get_pixel_predictors(df_var, pixel_inds, tile_str)
        print 'retrieving predictors', time.time() - t0
    if not isinstance(estimators, pd.core.frame.Series):
        x = tile_ul[0] + pixel_inds[0] * tx[1] # geographic coord for pixel
        y = tile_ul[1] + pixel_inds[1] * tx[5]
        estimators = sets.ix[(sets.ul_x <= x) & (sets.lr_x > x) & (sets.ul_y >= y) & (sets.lr_y < y), 'dt_model']
    
    predictions = np.array([e.predict(predictors) for e in estimators]).T

    #del estimators, sets, predictors
    last_dim = predictions.ndim - 1
    stats = {name: func(predictions, axis=last_dim) for name, func in agg_stats.iteritems()}
    import pdb; pdb.set_trace()
    return stats'''


    # establish DB connection and create empty relationship table for sample inds
    '''cmd = ('CREATE TABLE set_samples (set_id INTEGER, sample_id INTEGER, in_bag INTEGER);')
    with sqlite3.connect(db_path) as connection:
        connection.executescript(cmd)
        connection.commit()
    insert_cmd = 'INSERT INTO set_samples (set_id, sample_id, in_bag) VALUES (?,?,?);' '''
    
def predict_tree(i, n_sets, start_time, dt_model, predictors, set_id):

    predictions = dt_model.predict(predictors)
    format_tuple = i, n_sets, float(i)/n_sets * 100, (time.time() - start_time)/60
    sys.stdout.write('\r%s of %s tiles (%.1f%%) || run time: %.1f mins.' % format_tuple)
    sys.stdout.flush()
    return set_id, predictions



def predict_tile(tile_info, mosaic_path, mosaic_tx, df_sets, df_var, support_size, agg_stats, path_template, prj, nodata, snap_coord=None, tile_id_field='name', n_jobs=20):
    
    ''' make mosaic_path optional and to use on-the-fly tiles'''
    
    t1 = time.time()
    
    #get un-picklable objects
    mosaic_dataset = ogr.Open(mosaic_path)
    mosaic_ds = mosaic_dataset.GetLayer()
    agg_dict = get_agg_dict()
    agg_stats = agg_dict[agg_stats]
    driver = gdal.GetDriverByName('gtiff')
        
    tile_str = tile_info[tile_id_field]
    x_res = mosaic_tx[1]
    y_res = mosaic_tx[5]
    
    # Make mask
    fid = tile_info.name # .name gets idx of thre row from oringal df
    
    mask, _ = mosaic_by_tsa.kernel_from_shp(mosaic_ds, fid, mosaic_tx, -9999, return_mask=True)
    tile_size = mask.shape
    tile_tx, tile_extent = tx_from_shp(mosaic_ds, x_res, y_res, snap_coord, fid=fid)
    xmin, xmax, ymin, ymax = tile_extent
    ul_x = get_ul_coord(xmin, xmax, x_res, snap_coord[0])
    ul_y = get_ul_coord(ymin, ymax, y_res, snap_coord[1])
    tile_ul = ul_x, ul_y
    mask_offset = calc_offset(tile_info[['ul_x','ul_y']], tile_ul, tile_tx)
    
    feature = mosaic_ds.GetFeature(fid)
    tile_geom = feature.GetGeometryRef()
    t1 = time.time()
    overlapping_sets = get_overlapping_sets(df_sets, tile_geom)
    del tile_geom, feature
    
    t2 = time.time()
    index_cols = ['ul_r', 'lr_r', 'ul_c', 'lr_c']
    for c in index_cols:
        overlapping_sets[c] = -1
    for set_id, set_info in overlapping_sets.iterrows():
        # load estimators
        try:
            with open(overlapping_sets.ix[set_id, 'dt_file'], 'rb') as f: 
                overlapping_sets.ix[set_id, 'dt_model'] = pickle.load(f)
        except:
            overlapping_sets.ix[set_id, 'dt_model'] = joblib.load(overlapping_sets.ix[set_id, 'dt_file'])
        offset = calc_offset(tile_ul, set_info[['ul_x', 'ul_y']], tile_tx)
        t_inds, s_inds = mosaic_by_tsa.get_offset_array_indices(tile_size, support_size, offset)
        overlapping_sets.ix[set_id, index_cols] = t_inds
        overlapping_sets.ix[set_id,'nrows'] = t_inds[1] - t_inds[0]
        overlapping_sets.ix[set_id,'ncols'] = t_inds[3] - t_inds[2]
        
    predictors = get_pixel_predictors(df_var, mask_offset, tile_str, tile_size)
    nrows, ncols, n_predictors = predictors.shape
    n_sets = len(overlapping_sets)
    
    start_time = time.time()
    p = Parallel(n_jobs, backend="threading")(
        delayed(predict_tree)(
            i, n_sets, start_time, 
            set_info.dt_model, 
            predictors[set_info.ul_r:set_info.lr_r, set_info.ul_c:set_info.lr_c].reshape(set_info.nrows * set_info.ncols, n_predictors),
            set_id) 
            for i, (set_id, set_info) in enumerate(overlapping_sets.iterrows()))
    print '\n\n'
    out_dir = '/home/server/pi/homes/shooper/delete_test'
    aggregate(dict(p), overlapping_sets, nodata, mask, agg_stats, tile_info[tile_id_field], out_dir, tile_tx, prj, driver)
    import pdb; pdb.set_trace()

 
def aggregate(predictions, overlapping_sets, nodata, nodata_mask, agg_stats, tile_id, out_dir, tx, prj, driver):
    
    t0 = time.time()
    #importance_bands = []
    tile_size = nodata_mask.shape
    ar_tile = np.full((tile_size[0], tile_size[1], len(overlapping_sets)), nodata, dtype=np.uint8)
    for i, (s_ind, s_info) in enumerate(overlapping_sets.iterrows()):
        
        # Fill tile with prediction
        ar_tile[s_info.ul_r : s_info.lr_r, s_info.ul_c : s_info.lr_c, i] = predictions[s_ind].reshape(s_info[['nrows', 'ncols']])
        
        # Get feature with maximum importance and fill tile with that val
        '''if 'importance' in agg_stats:
            try:
                ar_import = np.full(ar_pred.shape, df_sets.ix[s_ind, 'max_importance'], dtype=np.uint8)
                import_band = fill_tile_band(tile_size, ar_import, tile_inds, nodata)
                importance_bands.append(import_band)
            except Exception as e:
                print e
                continue#'''    
    #ar_tile = np.dstack(pred_bands)
    '''if 'importance' in agg_stats:
        ar_impr = np.dstack(importance_bands)'''
    #del pred_bands#, importance_bands#, ar_import
    #print 'Filling tiles: %.1f seconds' % ((time.time() - t1))
    ar_tile = np.ma.masked_array(ar_tile, mask=ar_tile==nodata)
    driver = gdal.GetDriverByName('gtiff')
    path_template = os.path.join(out_dir, 'tile_{0}_%s.tif'.format(tile_id))
    nans = np.isnan(ar_tile).all(axis=2)
    if 'mean' in agg_stats:
        ar_mean = np.nanmean(ar_tile, axis=2)
        ar_mean[nans | ~nodata_mask] = nodata
        out_path = path_template %  'mean'
        mosaic_by_tsa.array_to_raster(ar_mean, tx, prj, driver, out_path, nodata=nodata, silent=True)
        ar_mean = None
        
    if 'stdv' in agg_stats or 'stdv' in agg_stats:
        ar_stdv = np.nanstd(ar_tile, axis=2)
        ar_stdv[nans | ~nodata_mask] = -9999
        out_path = path_template %  'stdv'
        mosaic_by_tsa.array_to_raster(ar_stdv, tx, prj, driver, out_path, nodata=-9999, silent=True)
        ar_stdv = None

    if 'vote' in agg_stats:
        ar_vote = mode(ar_tile, axis=2)
        ar_vote[nans | ~nodata_mask] = nodata
        out_path = path_template %  'vote'
        mosaic_by_tsa.array_to_raster(ar_vote, tx, prj, driver, out_path, nodata=nodata, silent=True)

    if 'count' in agg_stats:
        ar_count = np.sum(~np.isnan(ar_tile), axis=2)
        ar_count[nans | ~nodata_mask] = nodata
        out_path = path_template %  'count'
        mosaic_by_tsa.array_to_raster(ar_count, tx, prj, driver, out_path, nodata=nodata, silent=True)

    if 'importance' in agg_stats:
        ar_impor = mode(ar_impr, axis=2)
        ar_impor[nans | ~nodata_mask] = nodata
        out_path = path_template %  'importance'
        mosaic_by_tsa.array_to_raster(ar_impor, tx, prj, driver, out_path, nodata=nodata, silent=True)
        ar_impor = None
            
    if 'pct_vote' in agg_stats:
        if not 'vote' in agg_stats:
            ar_vote = np.mode(ar_tile, axis=2)
        if not 'count' in agg_stats:
            ar_count = np.sum(ar_tile.mask, axis=2)#~np.isnan(ar_tile), axis=2)  
        ar_pcvt = pct_vote(ar_tile, ar_vote, ar_count)
        ar_pcvt[nans | ~nodata_mask] = nodata
        out_path = path_template %  'pct_vote'
        mosaic_by_tsa.array_to_raster(ar_pcvt, tx, prj, driver, out_path, nodata=nodata, silent=True)
    
    if 'oob' in agg_stats:
        ar_oob = np.nanmean(ar_tile, axis=2)
        ar_oob[nans | ~nodata_mask] = nodata
        out_path = path_template %  'oob'
        mosaic_by_tsa.array_to_raster(ar_oob, tx, prj, driver, out_path, nodata=nodata, silent=True)
        ar_oob = None
    
    print 'Total time for tile %s: %.1f minutes\n' % (tile_id, ((time.time() - t0)/60)) 

