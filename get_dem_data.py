
# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 17:33:11 2020

@author: broberts
inputs: 
path: path where you want the download to go 
"""

import requests 
from pathlib import Path 
import sys
import pandas as pd 
from multiprocessing.pool import ThreadPool
import os
import pyParz
#path = Path('/vol/v3/ben_ak/raster_files/dem/')

def download_massive(path): 
	"""Download one very large file"""
	file_url = 'https://elevation.alaska.gov/download?geojson=%7B%22type%22%3A%22Polygon%22%2C%22coordinates%22%3A%5B%5B%5B-157.4561%2C66.6530%5D%2C%5B-157.4561%2C70.0506%5D%2C%5B-140.9766%2C70.0506%5D%2C%5B-140.9766%2C66.6530%5D%2C%5B-157.4561%2C66.6530%5D%5D%5D%7D&ids=152'

	  
	r = requests.get(file_url, stream = True) 
	  
	with open(path/"AK_5m_IFSAR.zip","wb") as tif: 
		for chunk in r.iter_content(chunk_size=1024*1024): 
			print(f'downloading chunk {chunk}')
			# writing one chunk at a time to pdf file 
			if chunk: 
				tif.write(chunk) 
	tif.close()

	return 

def get_ifsar(args): 
	"""Collect AK IFSAR 5m dem data in tiles from: http://dggs.alaska.gov/public_lidar/dds4/ifsar/dtm/."""
	input_url,out_path = args 
	print(input_url)
	r = requests.get(input_url, stream = True) 
	head,tail = os.path.split(input_url)

	with open(Path(out_path)/tail,"wb") as tif: 
		for chunk in r.iter_content(chunk_size=1024*512): 
			#print(f'downloading chunk {chunk}')
			# writing one chunk at a time to pdf file 
			if chunk: 
				tif.write(chunk) 
	tif.close()
	 

def run_download(param_file,out_path,data): 
	"""Helper function for get_ifsar."""
	df = pd.read_csv(param_file,sep='\t')
	#print(df.head())
	if data.lower() == 'ifsar': 
		url_struct='http://dggs.alaska.gov/public_lidar/dds4/ifsar/dtm/'
		download_list = list(df.iloc[:,0])
		download_list = [url_struct+str(i) for i in download_list]
	else: 
		download_list = list(df.iloc[:,0])

	#execute downloads in parallel 
	results_classes=pyParz.foreach(download_list,get_ifsar,args=[out_path],numThreads=8)

	# results = ThreadPool(5).imap_unordered(get_ifsar,download_list)
	# for path in results: 
	# 	print(path)

def main(): 
	# # script param
	script = sys.argv[0]
	# first command line param
	path = sys.argv[1] #param file or input file 
	out_path = sys.argv[2] #where you want your download to land
	#optional: add url_struct as arg to command line to change and remove hard coding 
	run_download(path,out_path,'modis')
	#get_ifsar(path,out_path)
if __name__ == '__main__':
    main()
