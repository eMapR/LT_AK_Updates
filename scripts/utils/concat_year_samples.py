# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:12:28 2018

@author: shooper
"""

import sys, os
import pandas as pd


def main(txt_path, target_column, years=[2001, 2011]):
    
    df = pd.read_csv(txt_path, sep='\t')
    target_cols = ['%s_%s' % (target_column, yr) for yr in years]
    missing_cols =  df.columns[~df.columns.isin(target_cols)]
    if len(missing_cols) > 0:
        raise ValueError('Columns not in columns of dataframe:\n\t' % '\n\t'.join(missing_cols))
    
    # Figure out sample size per year. Make sure all samples sum to n_sample
    n_sample = len(df)
    n_years = len(years)
    n_per_year = [n_sample/n_years for c in target_cols]
    n_per_year[-1] = n_sample - sum(n_per_year[:-1])  
    
    df[target_column] = None
    for n, col in zip(n_per_year, target_cols):
        
        # Sample from only records where target_col hasn;t been assigned
        sample = df.loc[df[target_col].isnull()].sample(n)
    