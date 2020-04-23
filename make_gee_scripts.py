# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:33:14 2018

@author: shooper
""" 

import os, sys
import pandas as pd


'''FEATURE_LISTS = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]'''
FEATURE_LISTS = [['r0805','r1504','r0902','r1301','r1101','r0802','r1002','r0904','r0801','r1001','r0602','r0803','r1201','r0701','r0804','r0903','r0502','r0906','r0503','r0905']]
FEATURE_COLLECTION = '15CzXYLkiBdhybFUP1WDzYX0yXd6-EBb9PHyQpnhj'
FEATURE_ID_COLUMN = 'epa_code'
START_YEAR = 1988
END_YEAR = 2017

COEFFICIENTS = {'TCB': 1,
              'NBR': -1,
              'B5': 1,
              'NDVI': -1,
              'NDSI': -1, 
              'TCG': -1,
              'B3': 1
              }
SCRIPT_TEMPLATE = './script_template.txt'

def main(lt_run_info, out_dir, script_template=None):
    
    #read in the template and join it into oen long string
    if script_template is None:
        script_template = SCRIPT_TEMPLATE
    with open(script_template) as f:
        template_lines = [l for l in f.readlines()]
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    lt_runs = pd.read_csv(lt_run_info, dtype={'date':object})
    for i, info in lt_runs.iterrows():
        # make a dict from info
        format_dict = {}
        format_dict['feature_collection'] = FEATURE_COLLECTION
        format_dict['feature_id_column'] = FEATURE_ID_COLUMN
        format_dict['start_year'] = START_YEAR
        format_dict['end_year'] = END_YEAR 
        format_dict['start_date'] = "['%s-%s']" % (info.date[:2], info.date[2:4])
        format_dict['end_date'] = "['%s-%s']" % (info.date[4:6], info.date[6:8])
        if 'index_coefficient' not in info:
            info['index_coefficient'] = COEFFICIENTS[info.seg_ind]
        format_dict['seg_index'] = "[['%s', %s]]" % (info.seg_ind, info.index_coefficient)
        format_dict['ftv_list'] = info.ftv_list
        format_dict['vert_list'] = info.vert_list
        
        
        for feature_values in FEATURE_LISTS:
            format_dict['feature_values'] = str(feature_values)
            format_dict['start_feat'] = feature_values[0]
            format_dict['end_feat'] = feature_values[-1]
            format_dict['seg_ind'] = info.seg_ind
            format_dict['date_range'] = info.date
            out_bn = ('ee_script_'
                     r'%(seg_ind)s_'
                     r'%(date_range)s_'
                     r'%(start_feat)s_'
                     r'%(end_feat)s.js') % format_dict
            out_js = os.path.join(out_dir, out_bn)
            with open(out_js, 'w') as js:
                for line in template_lines:
                    js.write(line % format_dict)
        
        print 'File written to', out_js


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))