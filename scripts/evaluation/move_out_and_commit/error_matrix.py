#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:27:33 2018

@author: braatenj
"""

"""
def kappa_coeff(df, labels, accuracy):
    
    total_pxl = np.nansum(df.loc[labels, labels].values)
    marg_t = np.nansum(df.loc[labels, labels], axis=0)
    marg_p = np.nansum(df.loc[labels, labels], axis=1)
    acc_e = np.nansum((marg_t * marg_p)/total_pxl)/total_pxl # Expected accuracy
    kappa = (accuracy - acc_e)/(1 - acc_e) 
    
    return kappa

def calc_ci(df, labels, z_score=1.96):
    ''' Calc confidence intervals in place'''
    # Calc user's confidence interval
    user = df.loc[labels, 'user']
    total = df.loc[labels, 'total']
    variance_u = user * (1 - user)/(total - 1) #From Oloffson et al. 2014
    df.loc[labels, 'u_ci'] = np.round(variance_u**.5 * z_score, 3) * 100
    
    # Calc overall confidence interval
    variance_o = np.sum(df.loc[labels, 'pct_area']**2 * variance_u) #From Oloffson 2014
    df.loc['producer', 'acc_ci'] = round(variance_o**.5 * z_score, 3) * 100
    
    # Calc producer's confidence interval
    variance_p = []
    for j in labels:
        # estimated total pixel count in reference map for this class
        Nj_hat = np.sum(df.loc[labels, 'total_pxl']/df.loc[labels, 'total'] * df.loc[labels, j])
        Nj = df.loc[j, 'total_pxl'] #total pixel count in this map class
        nj = df.loc[j, 'total'] # total sample count in this reference class
        Ni = df.loc[labels != j, 'total_pxl'] #total pixel count in each map class
        ni = df.loc[labels != j, 'total'] #total sample count in each map class
        p = df.loc['producer', j]
        u = df.loc[j, 'user']
        nij_ni = df.loc[labels != j, j]/ni #ratio of sample count correct to total sample
        v = (1/Nj_hat**2) * ((Nj**2 * (1 - p)**2 * u * (1 - u))/(nj - 1) +\
                            p**2 * np.sum(Ni**2 * nij_ni * (1 - nij_ni)/(ni - 1)))
        variance_p.append(v)
    df.loc['p_ci', labels] = np.round(np.array(variance_p)**.5 * z_score, 3) * 100

def confusion_matrix_by_area(ar_p, ar_t, bins=30, out_txt=None, get_totals=True, total_counts=None, silent=False):
    ''' 
    Return a dataframe of an area-adjusted confusion matrix
    '''
    # Check if bins is an int (could be an iterable of bin ranges). If so,
    #   calcualte bin ranges.
    out_txt = '/vol/v3/lt_stem_v3.1/evaluation/test/error_matrix.txt'
    ar_p = predP
    ar_t = trainP
    t_samples = ar_t
    p_samples = ar_p
    
    t0 = time.time()
    if not get_totals and total_counts==None:
        get_totals = True
    
    # if bins are not defined, then generate them based on range of training array    
    if type(bins) == int:
        t_range = (ar_t.max() - ar_t.min())
        bin_sz = t_range/bins
        bins = [(-1, 0)] + [(i, i + bin_sz) for i in range(0, int(t_range), int(bin_sz))]
    
    """
    if match:
        t_samples, p_samples = get_samples(ar_p, ar_t, p_nodata, t_nodata, samples, match=match)
    else:
        t_samples = samples[target_col].values
        #tt = ar_t[samples.row, samples.col]
        #import pdb; pdb.set_trace()
        if 'prediction' not in samples.columns:
            p_samples = ar_p[samples.row, samples.col]
        else:
            p_samples = samples.prediction
    
    if not np.any(mask):
        mask = (ar_p == p_nodata) | (ar_t == t_nodata)
    ar_p = ar_p[~mask]
    ar_t = ar_t[~mask]
    n_pixels = ar_t.size
    del ar_t
    """
    
    n_pixels = ar_t.size

    
    # For each bin in the target array, count how many pixels are in each bin
    #   in the prediction array.
    #rows = []
    cols = {}
    labels = []
    sample_counts = []
    if get_totals:
        total_counts  = {}
    empty_bins = []
    for i, (l, u) in enumerate(bins):
        #i=2
        #(l, u) = (75, 150.7)
        if not silent:
            print('Getting counts for reference class from %s to %s' % (l, u))
        t2 = time.time() 
        t_mask = (t_samples <= u) & (t_samples > l)# Create a mask of bin values from target
        p_mask = (p_samples <= u) & (p_samples > l)
        label = '%s_%s' % (l, u)
        labels.append(label)
        if not p_mask.any():
            cols[label] = np.zeros(len(bins))
            total_counts[label] = 0
            empty_bins.append(label)
            if not silent:
                print('No samples found in prediction raster betwen %s and %s...\n' % (l, u))
            continue#'''
            
        counts = []
        for this_l, this_u in bins:
            this_p_mask = (p_samples <= this_u) & (p_samples > this_l)
            this_p = p_samples[t_mask & this_p_mask]

            counts.append(len(this_p))
        
        if get_totals:
            total_count = len(ar_p[(ar_p <= u) & (ar_p > l)])#total pixels in bin in predicted map
        n_samples = len(p_samples[p_mask])#total samples in this bin in predicted samples
        #counts = counts #+ [n_samples, total_count] 
        
        #rows.append(counts)
        cols[label] = counts
        sample_counts.append(n_samples)
        if get_totals:
            total_counts[label] = total_count
        
        if not silent:
            print('Time for this class: %.1f seconds\n' % (time.time() - t2))
    
    #import pdb; pdb.set_trace()
    df = pd.DataFrame(cols)#, columns=labels + ['n_samples', 'total'])
    df['bin'] = labels
    df = df.set_index('bin')
    df.drop(empty_bins, axis=0, inplace=True)
    df.drop(empty_bins, axis=1, inplace=True)
    labels = df.columns.tolist()
    
    df['total_pxl'] = pd.Series(total_counts)
    

    # Calculate proportional accuracy
    df['pct_area'] = df.total_pxl/float(n_pixels)
    df['total'] = df.loc[labels, labels].sum(axis=1)
    df.loc['total'] = df.loc[labels, labels].sum(axis=0)
    #df_smp = df.ix[labels, labels]
    df_adj = df.loc[labels, labels].div(df.total, axis=0).mul(df.pct_area, axis=0)

    #import pdb; pdb.set_trace()
    
    # Calculate user's and producer's accuracy
    for l in labels:
        correct = df_adj.loc[l,l]
        df.loc[l, 'user'] = correct / df_adj.loc[l, labels].sum()
        df.loc['producer', l] = correct / df_adj.loc[labels, l].sum()

    calc_ci(df, np.array(labels)) # calculates in place    
    
    '''# Calculate confidence intervals for user's and producer's
    df['u_ci'] = df.ix[labels].apply(
    lambda x: confidence_interval(x[labels], 
                                  x['total'], 
                                  x['pct_area'], 
                                  x['user']
                                  ),
                                     axis=1).round(3) * 100
    df.ix['p_ci'] = df[labels].apply(
    lambda x: confidence_interval(x[labels], 
                                  df.ix[labels, 'total'],
                                  df.ix[labels, 'pct_area'], 
                                  x['producer']
                                  ),
                                     axis=0).round(3) * 100'''

    
    # Calc overall accuracy and confidence interval
    total_pxl = df_adj.values.sum()
    correct = np.diag(df_adj)
    accuracy = correct.sum()
    '''df.ix['producer', 'acc_ci'] = confidence_interval(np.diag(df.ix[labels, labels]), 
                                                      df.ix[labels, 'total'],
                                                      df.ix[labels, 'pct_area'],
                                                      accuracy
                                                      ) * 100'''
    df['user'] = df['user'].round(3) * 100 # Round for readibility
    df.loc['producer', labels] = df.loc['producer', labels,].round(3) * 100
    # Recalc totals with area-adjusted estimates
    #df.ix[labels, labels] = df_adj
    df['total_adj'] = df_adj.loc[labels, labels].sum(axis=1)
    df.loc['total_adj'] = df_adj.loc[labels, labels].sum(axis=0)
    
    # Calc kappa and disagreement
    kappa = kappa_coeff(df, labels, accuracy)
    df.loc['producer', 'user'] = round(100 * accuracy, 1)
    df.loc['producer', 'kappa'] = round(kappa, 3)
    '''disagree_q, total_q = quantity_disagreement(df, labels)
    disagree_a, total_a = allocation_disagreement(df, labels)
    df.ix[labels, 'quanitity'] = disagree_q
    df.ix['producer', 'quanitity'] = total_q
    df.ix[labels, 'allocation'] = disagree_a
    df.ix['producer', 'allocation'] = total_a'''

    # Fill in the empty rows
    labels += empty_bins
    out_cols = labels + ['total_pxl','pct_area','total', 'total_adj', 'user','u_ci','acc_ci','kappa','quanitity','allocation']
    out_index = labels + ['producer', 'total', 'total_adj', 'p_ci']
    df = df.reindex(index=out_index, columns=out_cols)
    
    df_smp = df.copy()
    df.loc[labels, labels] = df_adj.loc[labels, labels]
    
    if out_txt:
        df_smp.to_csv(out_txt.replace('.txt', '_sample.txt'), sep='\t')
        df.to_csv(out_txt.replace('.txt', '_proportion.txt'), sep='\t')
        if not silent: print('\nDataframe written to: ', out_txt)
        
    
    if not silent: print('\nTotal time: %.1f minutes' % ((time.time() - t0)/60))
    return df, df_smp#"""
