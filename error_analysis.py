import os
import sys
from osgeo import ogr 
import json 
from osgeo import gdal
import geopandas as gpd 
from rasterstats import zonal_stats,point_query
import rasterio
import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import pickle 
import matplotlib as mpl
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
#from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
def rasterize(shp,resolution,output_dest): 
	"""Convert vector to raster."""
	input_shp = ogr.Open(shp)
	shp_layer = input_shp.GetLayer()

	pixel_size = resolution
	xmin, xmax, ymin, ymax = shp_layer.GetExtent()
	head,tail = os.path.split(shp)
	output_raster = output_dest+tail[:-4]+'.tif'
	print(output_raster)
	ds = gdal.Rasterize(output_raster, shp, xRes=pixel_size, yRes=pixel_size, 
	                    burnValues=255, outputBounds=[xmin, ymin, xmax, ymax], 
	                    outputType=gdal.GDT_Byte)
	ds = None
	return output_raster

def calc_zonal_stats(raster,shp,resolution,stat,source): 
	"""Calculate pixel counts inside polygons."""
	geo_df = gpd.read_file(shp)
	
	with rasterio.open(raster) as src: 
		#construct the transform tuple in the form: top left (x coord), west-east pixel res, rotation (0.0), top left northing (y coord), rotation (0.0), north-south pixel res (-1*res)
		transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
		arr = src.read(1)#.astype('float')
		#arr[arr == None] = 0
		#arr[arr == np.nan] = 0
	#rasterstats zonal stats produces a list of dicts, get the value
	stats = zonal_stats(geo_df,arr,stats=stat,transform=transform,nodata=255)
	output_geodf = geo_df.join(pd.DataFrame(stats))#.drop(['left','right','top','bottom'])
	#rename cols so they don't get angry when we join
	old_names = output_geodf.columns
	new_names = [source+'_'+i for i in old_names]
	column_names = dict(zip(old_names,new_names))
	output_geodf.rename(columns=column_names,inplace=True)

	#output_geodf = output_geodf.rename(columns={'count':source+'_count'},inplace=True)
	
	#print(output_geodf)
	return output_geodf
	# def extract_raster_pts(input_raster,input_shp,resolution): 
	# 	"""Extract raster pixel values at random pts for error analysis."""
	# 	geo_df = gpd.read_file(input_shp)
	# 	with rasterio.open(input_raster) as src:
	# 		#transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
	# 		#arr = src.read(1)
	# 	pts = zonal_stats(input_shp, input_raster,stats='majority',nodata=255,band=1)
	# 	print(pts)


def read_pickles(*argv): 
	#head,tail = os.path.split(raster_2)
	actual_file = argv[9]+f'{argv[5]}_{argv[10]}_zonal_stats_df'
	predicted_file = argv[9]+f'{argv[7]}_{argv[6]}_{argv[10]}_zonal_stats_df'
	if argv[8].lower()=='true': 
		print('pickling...')
		#generate zonal stats and pickle
		
		if not os.path.exists(predicted_file): 
			print(f'creating new files {actual_file} and {predicted_file}')
			actual = calc_zonal_stats(argv[0],argv[2],argv[3],argv[4],argv[5])
			predicted = calc_zonal_stats(argv[1],argv[2],argv[3],argv[4],argv[6])
			#df = pd.concat([stem_df,rgi_df],axis=1)
			actual_ds = pickle.dump(actual, open(actual_file, 'ab' ))
			predicted_ds = pickle.dump(predicted,open(predicted_file,'ab'))
			return actual, predicted 
		else: 
			print('both files already exist')
			pass
	else: #read in the pickled df if its the same data
		print('reading from pickle...')
		actual_df = pickle.load(open(actual_file,'rb'))
		predicted_df = pickle.load(open(predicted_file,'rb'))
		#print(actual_df.head())
		return actual_df,predicted_df
#argv order: #0:nlcd_raster,1:stem_raster,2:random_pts,3:resolution,4:stat,5:actual_source,6:predicted_source,7:model_run,8:write_to_pickle,9:pickle_dir,10:modifier)

def calc_confusion_matrix(*argv):#actual_source,predicted_source,stat,): 
	"""Calculate a confusion matrix to compare nlcd or rgi and classification."""
	data = read_pickles(*argv)
	actual = data[0].replace(np.nan,0)#read_pickles(*argv)[0].replace(np.nan,0)
	predicted = data[1].replace(np.nan,0)#read_pickles(*argv)[1].replace(np.nan,0)#calc_zonal_stats(predicted_raster,shp,resolution,stat,predicted_source)
	#print(actual.head())
	#print(predicted.head())
	actual_col = actual[str(argv[5]+'_'+argv[4])]
	predicted_col = predicted[str(argv[6]+'_'+argv[4])]
	print(actual_col.unique())
	print(predicted_col.unique())
	actual_ls = [float(i) for i in list(actual_col)]
	predicted_ls = [float(i) for i in list(predicted_col)]
	labels = sorted(list(set(list(actual_col)+list(predicted_col))))# sorted(actual[str(argv[5]+'_'+argv[4])].unique())
	print(labels)
	results = confusion_matrix(actual_ls, predicted_ls,labels) 
	#disp = plot_confusion_matrix(None,actual_ls,predicted_ls,display_labels=labels,cmap=plt.cm.Blues)
	#fig,(ax,ax1) = plt.subplots(nrows=1,ncols=2)
	ax=plt.subplot()
	sns.heatmap(results,annot=True,ax=ax,fmt='g')
	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
	ax.set_title(f'Confusion Matrix: {argv[5]} {argv[6]} {argv[7]} model run') 
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	print(results) 
	print ('Accuracy Score :',accuracy_score(actual_ls, predicted_ls))
	print ('Report : ')
	print (classification_report(actual_ls, predicted_ls))
	plt.show()
	plt.close('all')


def create_zonal_stats_df(stem_raster,rgi_raster,shp,resolution,output_dir,boundary,zoom,pickle_dir,read_from_pickle): 
	"""A helper function for calc_zonal_stats."""
	head,tail = os.path.split(stem_raster)

	if read_from_pickle.lower()=='true': 
		print('pickling...')
		#generate zonal stats and pickle
		output_file = pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'
		if os.path.exists(output_file): 
			pass
		else: 
			stem_df = calc_zonal_stats(stem_raster,shp,resolution,'stem')
			rgi_df = calc_zonal_stats(rgi_raster,shp,resolution,'rgi')
			df = pd.concat([stem_df,rgi_df],axis=1)
			pickle_data=pickle.dump(df, open(output_file, 'ab' ))
	else: #read in the pickled df if its the same data
		print('reading from pickle...')
		df = pickle.load(open(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df','rb'))
	#calculate the percent error aggregated by cell (pixelwise doesn't make sense because its binary)
	df['pct_err'] = (((df['stem_count']-df['rgi_count'])/df['rgi_count'])*100)
	#rename a col to geometry because the plot function wants that
	df.rename(columns={'rgi_geometry':'geometry'},inplace=True)
	#get rid of garbage 
	df = df.drop(['stem_left','stem_top','stem_right','stem_bottom','stem_geometry','rgi_left','rgi_top','rgi_right','rgi_bottom'],axis=1)
	#select a subset by getting rid of infs
	df_slice = df.replace([np.inf, -np.inf],np.nan).dropna(axis=0)#df.query('stem_count!=0')#[df['stem_count']!=0 and df['rgi_count']!=0]
	#read in plotting shapefiles
	inset = gpd.read_file(boundary)
	background = gpd.read_file(zoom)
	#do the plotting 
	fig,ax = plt.subplots()
	#make the colorbar the same size as the plot
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right",size="5%",pad=0.1)
	left, bottom, width, height = [0.1, 0.525, 0.25, 0.25]
	ax1 = fig.add_axes([left, bottom, width, height])
	#specify discrete color ramp 
	cmap = mpl.colors.ListedColormap(['#005a32','#238443','#41ab5d','#78c679','#addd8e','#d9f0a3','#ffffcc',#'#ffffcc','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32',
	'#F2B701','#E73F74','#180600','#E68310','#912500','#CF1C90','#f23f01',
	'#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']) 
	#'#855C75','#D9AF6B','#AF6458','#736F4C','#526A83','#625377','#68855C','#9C9C5E','#A06177','#8C785D','#467378','#7C7C7C'])
	#'#5F4690','#1D6996','#38A6A5','#0F8554','#73AF48','#EDAD08','#E17C05','#CC503E','#94346E','#6F4070','#994E95','#666666'])#'#5D69B1','#52BCA3','#99C945','#CC61B0','#24796C','#DAA51B','#2F8AC4','#764E9F','#ED645A','#CC3A8E','#63665b'])#["#f5b460","#F1951C","#a86813", "#793200", "#004039","#006B5F", "#62BAAC","#ba6270"])#'#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'])#
	norm = mpl.colors.BoundaryNorm([-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap.N) 
	background.plot(color='lightgray',ax=ax)
	df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax,legend=True,cax=cax,cmap=cmap,norm=norm)
	ax.set_title(f'{tail[0:8]} model run AK southern processing region percent error')

	inset.plot(color='lightgray',ax=ax1)
	df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax1,cmap=cmap,norm=norm)
	ax1.get_xaxis().set_visible(False)
	ax1.get_yaxis().set_visible(False)
	plt.tight_layout() 
	plt.show()
	plt.close('all')

def nlcd_disagree_summary(input_raster): 
	"""Summarize the nlcd classes where LT/stem and rgi disagree."""
	with rasterio.open(input_raster) as src: 
		arr = src.read(1)#.astype('float')
		arr[arr<0] = 0
		arr = arr.astype('int32')
		bins=np.unique(arr.flatten())[1:]
		bin_count = np.bincount(arr.flatten())
		bin_count = bin_count[bin_count != 0][1:]
		head,tail = os.path.split(input_raster)
		fig,ax = plt.subplots()
		ax.bar(bins,bin_count,width=1,align='center',color='red',edgecolor='black')#,tick_label=bins_str)
		ax.set_xticks(bins)
		ax.set_xlabel('NLCD Class')
		ax.set_ylabel('Counts')
		ax.set_title(tail[:-4])
		plt.show()

def reclassify(input_raster,nlcd_version): 
	with rasterio.open(input_raster) as src: 
		arr = src.read()
		profile = src.profile
		#print(profile)
		class_pairs = {0:0,11:7,12:9,22:18,23:16,31:27,41:36,42:38,43:46,51:48,52:56,71:67,72:78,90:87,95:92}
		#do the reclassify
		if nlcd_version.lower()=='old': 
			for k,v in class_pairs.items(): 
				arr[np.where(arr==k)]=v
		else: 
			inverted = {v:k for k,v in class_pairs.items()}
			for k,v in inverted.items(): 
				arr[np.where(arr==k)] = v
			

	output_file = input_raster[:-4]+'_reclassify.tif'
	with rasterio.open(output_file, 'w', **profile) as dst: 
		dst.write(arr)

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		shapefile = variables["shapefile"]
		resolution = int(variables["resolution"])
		output_dir = variables["output_dir"]
		pickle_dir = variables["pickle_dir"]
		rgi_raster = variables["rgi_raster"]
		stem_raster = variables["stem_raster"]
		nlcd_raster = variables["nlcd_raster"]
		boundary = variables["boundary"]
		zoom = variables["zoom"]
		hist_raster = variables["hist_raster"]
		random_pts = variables["random_pts"]
		write_to_pickle = variables["write_to_pickle"]
		stat = variables["stat"]
		actual_source = variables["actual_source"]
		predicted_source = variables["predicted_source"]
		model_run = variables["model_run"]
		nlcd_version = variables["nlcd_version"]
		modifier = variables["modifier"]
	#reclassify(nlcd_raster,nlcd_version)
	#nlcd_disagree_summary(stem_raster)
	#create_zonal_stats_df(stem_raster,rgi_raster,shapefile,resolution,output_dir,boundary,zoom,pickle_dir,write_to_pickle)
	#calc_zonal_stats(nlcd_raster,random_pts,resolution,stat,'nlcd')
	calc_confusion_matrix(rgi_raster,stem_raster,random_pts,resolution,stat,actual_source,predicted_source,model_run,write_to_pickle,pickle_dir,modifier)
	#extract_raster_pts(nlcd_raster,random_pts,resolution)
	#rasterize(shapefile,resolution,output_directory)
if __name__ == '__main__':
	main()

#   "/vol/v3/ben_ak/param_files_rgi/southern_region/models/2001_06032020/models_2001_vote.tif",
#"/vol/v3/ben_ak/param_files_rgi/southern_region/models/2001_06062020_updated_nlcd/06062020_model_run_updated_nlcd.tif",
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==9)] = 12
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7
# arr[np.where(arr==7)] = 7

#arr[np.where(arr==1)] = 9 
# arr[np.where(arr==11)] = 7
# arr[np.where(arr==12)] = 9
# arr[np.where(arr==22)] = 18
# arr[np.where(arr==31)] = 27
# arr[np.where(arr==41)] = 36
# arr[np.where(arr==42)] = 38
# arr[np.where(arr==43)] = 46
# arr[np.where(arr==51)] = 48
# arr[np.where(arr==52)] = 56
# arr[np.where(arr==71)] = 67
# arr[np.where(arr==72)] = 78
# arr[np.where(arr==90)] = 87
# arr[np.where(arr==95)] = 92