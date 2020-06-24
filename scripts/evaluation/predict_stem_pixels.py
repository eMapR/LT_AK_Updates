# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 08:21:17 2018

@author: shooper
"""

import os, sys, time
import pandas as pd
import numpy as np
from mutliprocessing import Pool

import evaluation
package_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(package_dir)
import stem


def print_message(i, n_iterations, t0, max_bar_size=70, pattern='\nFinished %s of %s iterations (%.1f%%)\n%s\n%.1f minutes\n'):
    ''' Configure the terminal message'''
    
    lapsed_time = (time.time() - t0)/60
    percent_done = float(i)/n_iterations
    n_chars = int(percent_done * max_bar_size)
    n_blank = max_bar_size - n_chars
    progress_bar = '||' + u'\u2588' * n_chars + '_'  * n_blank + '||'
    msg = pattern % (i + 1, n_iterations, percent_done * 100, progress_bar, lapsed_time)
    prefix = '\033[A' * n_returns
    
    print('\n' * n_returns)
    sys.stdout.write(prefix + msg)
    sys.stdout.flush()


def within(args):
    i, n_pixels, t0, sets, obs_id, pixel = args
    set_ids = sets.loc[(pixel.x >= sets.ul_x) & (pixel.x < sets.lr_x) &
                       (pixel.y > sets.lr_y) & (pixel.y <= sets.ul_y),
                        ].index
    print_message(i, n_pixels, t0)
    
    return obs_id, sorted(set_ids), len(set_ids)


def contains(args):
    a=0
           mask = (df_xy.x >= min_x) &
                (df_xy.x < max_x) & 
                (df_xy.y > min_y) & 
                (df_xy.y <= max_y)] 


def main(sample_txt, set_txt, out_dir, njobs=20):
    
    
    sample = pd.read_csv(sample_txt, sep='\t', index_col='obs_id')
    sets = pd.read_csv(set_txt, sep='\t', usecols=['lr_x', 'lr_y', 'ul_x', 'ul_y', 'set_id'], index_col='set_id')
    n_pixels = len(sample)
    n_sets = len(sets)
    
    t0 = time.time()
    if len(sample) < len(sets):
        args = [[i, n_pixels, t0, sets, obs_id, pixel]  for i, (obs_id, pixel) in enumerate(samples.iterrows())]:
        pool = Pool(njobs)
        pixels = pool.map(args, within, 1)
        pool.close()
        pool.join()
    
        obs_ids, set_ids, n_overlapping_sets = zip(*pixels)
        max_overlapping = max(n_overlapping_sets)
        
        # Make an array where each pixel is a row
        set_id_array = np.full((n_pixels, max_overlapping), sets.index.max() + 1, dtype=np.uint16)
        for i, ids in enumerated(set_ids):
            set_id_array[i, :len(ids)] = ids
        unique_cells, _, cell_inds = stem.unique_pixels(set_id_array)
        for i, cell in enumerate(unique_cells):
            set_inds = cell[cell != sets.index.max() + 1]
            
        
        
    else:
        a = 0
    
    
 