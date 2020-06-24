  
"""
Author: @broberts
Script to build figures for pixel counts and assessment of medoid composites and fitted LT outputs for AK glaciers project.
Example run from command line: python pixel_counts_v1.1.py /vol/v3/ben_ak/figure_building/stats_by_tile_new_mask_pre_2000_north /vol/v3/ben_ak/figure_building/figure_outputs/ 3,4,5,6 1984 2003

"""

import os
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
from pathlib import Path 
import glob 
import seaborn as sns 
import numpy as np 

def data_clean(csv,epoch,start_year): 
	df=pd.read_csv(csv) 
	#clean up the df 
	df=df.drop(['B2','B3','B4','B5','B7','.geo'],axis=1)
	df.rename(columns = {'system:index':'year','B1':'count','eetile2x15':'eetile2x15'}, inplace = True) 
	if int(start_year) < 1999: 
		pass
		if epoch == 3: 
			df['year'].replace({0: 1986, 1: 1989,2: 1992,3: 1995,4: 1998,5:2001}, inplace=True)
		elif epoch == 4:
			df['year'].replace({0: 1987,1:1991,2:1995,3:1999,4:2003}, inplace=True)
		elif epoch == 5: 
			df['year'].replace({0:1988,1:1993,2:1998,3:2003}, inplace=True)
		elif epoch == 6: 
			df['year'].replace({0:1989,1:1995,2:2001},inplace=True)
	else: 	
		pass
		if epoch == 1: 
			df['year'].replace({0:2000,1:2001,2:2002,3:2003,4:2004,5:2005,6:2006,7:2007,8:2008,9:2009,10:2010,11:2011,12:2012,13:2013,14:2014,15:2015,16:2016,17:2017,18:2018,19:2019},inplace=True)
		elif epoch == 2: 
			df['year'].replace({0:2001,1:2003,2:2005,3:2007,4:2009,5:2011,6:2013,7:2015,8:2017,9:2019},inplace=True)
		elif epoch == 3: 
			df['year'].replace({0:2001,1:2004,2:2007,3:2010,4:2013,5:2016,6:2019},inplace=True)
		elif epoch == 4: 
			df['year'].replace({0:2003,1:2007,2:2011,3:2015,4:2019},inplace=True)
		
	
	return df

def get_data(filepath,epoch,start_year,end_year): 
	df_ls = []
	for file in Path(filepath).glob('*.csv'): 
		#separate the files by epoch
		if f'{epoch}_year' in str(file): 
			df = data_clean(file,epoch,start_year)
			#get just the years you want (split around 1999)
			df = df[df['year'].isin(list(range(int(start_year),int(end_year)+1)))] #add one because its non-inclusive
			#remove the tiles with no glaciers
			df = df[df['eetile2x15'].isin(list(range(16,23)))] #this is hardcoded and needs to be changed if you are going to run for the entire state 
			df['epoch_len'] = epoch
			if not df.empty: 
				df_ls.append(df)

	return df_ls

def plotting(dfs,start_year,end_year,epoch,out_path): 
	"""Plot the data to compare epochs."""
	rows=1 #2
	cols=4 #5
	#create the xtick labels

	fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(25,13),sharex=True) #create a figure
	axs = axs.flatten()
	tile_labels = [16, 18, 20, 22]#[11,13,15,17,19,21,23,25,27,29]
	width = 1.0         # the width of the bars
	#assign the lists that are common to both eras
	three = dfs[3]
	four = dfs[4]
	#get the post 2000 era
	if int(start_year) >= 2000: 
		one =dfs[1]
		two = dfs[2]
	else: 
		#get years before 2000
		five = dfs[5]
		six = dfs[6]

	for n in range(rows*cols): 
		try:   
			if int(start_year) < 2000: #plot the earlier era 
				axs[n].bar(six[n]['year'],six[n]['count'],width=width,label='6 year epoch',align='center',color='#e66101',edgecolor='black',linewidth=0.25) 
				axs[n].bar(five[n]['year'],five[n]['count'],width=width,label='5 year epoch',align='center',color='#fdb863',edgecolor='black',linewidth=0.25) 
				axs[n].bar(four[n]['year'],four[n]['count'],width=width,label='4 year epoch',align='center',color='#b2abd2',edgecolor='black',linewidth=0.25) 
				axs[n].bar(three[n]['year'],three[n]['count'],width=width,label='3 year epoch',align='center',color='#5e3c99',edgecolor='black',linewidth=0.25) 
			else: 
				axs[n].bar(four[n]['year'],four[n]['count'],width=width,label='4 year epoch',align='center',color='#b2abd2',edgecolor='black',linewidth=0.25) 
				axs[n].bar(three[n]['year'],three[n]['count'],width=width,label='3 year epoch',align='center',color='#5e3c99',edgecolor='black',linewidth=0.25) 
				axs[n].bar(two[n]['year'],two[n]['count'],width=width,label='2 year epoch',align='center',color='#2ca25f',edgecolor='black',linewidth=0.25) 
				axs[n].bar(one[n]['year'],one[n]['count'],width=width,label='1 year epoch',align='center',color='#e5f5f9',edgecolor='black',linewidth=0.25) 


			axs[n].set_title(label='GEE tile ' + str(tile_labels[n]),fontsize=15) #make a y axis label that is the number of the month for that dataframe 
			axs[n].set(ylabel='Valid pixel count') 
			#axs[n].set(xlabel='End year of epoch')
			axs[n].tick_params(axis='x', rotation=90)
			axs[n].set_xticks(range(start_year,end_year+1))
			axs[n].tick_params(axis='x', labelsize= 13)
			axs[n].tick_params(axis='y', labelsize= 15)

			handles, labels = axs[n].get_legend_handles_labels() #plot this is taking from is somewhat random. If they are fall filled in correctly this can be reset to n
			fig.legend(handles, labels, loc='upper left')
		except (IndexError): 
			print('there was an indexerror error')
			continue
		
	#plt.legend()
	plt.suptitle(f'{epoch} epochs {start_year}-{end_year} valid pixel comparison with L4 (north)', fontsize=15) #make a title.Change back when all of the things have been run for group 5
	#plt.subplots_adjust(top=0.85)
	plt.tight_layout(rect=[0,0.03,1,0.95])
	#plt.show()
	
	out_path=Path(out_path)
	plt.savefig(out_path/f'{epoch}_epochs_{start_year}_{end_year}_valid_pixel_comparison_new_masks_north.png') #save the output

	#close everything 
	plt.close('all') 

def main(): 

	# # script param
	script = sys.argv[0]
	# first command line param
	path = sys.argv[1]
	#second command line arg
	out_path = sys.argv[2] #where you want to save the figs
	#third command line arg
	epoch = sys.argv[3] #list of epochs
	#fourth command line arg
	start_year = sys.argv[4]
	#fifth command line arg 
	end_year = sys.argv[5]
	
	# check to see if the file path is real 
	if os.path.exists(path):
		print(path,' : is real')
	else:
		print('Check the first filepath you entered.')
	df_dict = {}
	epoch_ls = [int(n) for n in sys.argv[3].split(',')]  # if you want ints
	print(epoch_ls)
	for i in epoch_ls: 
		df_dict.update({i:get_data(path,i,start_year,end_year)})
	print(df_dict)
	plotting(df_dict,int(start_year),int(end_year),epoch_ls,out_path)



if __name__ == '__main__':
    main()

import sys

# for n in range(rows*cols): 

	# 	for key,value in dfs.items(): #dict of count data 
	# 		try: 
	# 			if tile_labels[n]==value[n]['eetile2x15'].iloc[0]:
	# 				print('the tile id is: ',value[n]['eetile2x15'].iloc[0])
	# 				#axs[n].bar(value[n]['year'],value[n]['count'],width,label=f'{key} year epoch')

	# 				# if value[n]['year'].iloc[0] in [1995,1998,2001,2003,2007,2013,2019]: #check for duplicate years  
	# 				# 	print('that year was a duplicate')
	# 				# 	axs[n].bar(value[n]['year']-width/2,value[n]['count'],width,label=f'{key} year epoch')
	# 				# 	axs[n].bar(value[n]['year']+width/2,value[n]['count'],width,label=f'{key} year epoch')

	# 				if key == 5:
	# 					axs[n].bar(value[n]['year'],value[n]['count'],width,label=f'{key} year epoch')
	# 				elif key == 4:
	# 					axs[n].bar(value[n]['year'],value[n]['count'],width,label=f'{key} year epoch')
	# 				elif key == 3:  
	# 					axs[n].bar(value[n]['year'],value[n]['count'],width,label=f'{key} year epoch')
	# 				# else: 
	# 				# 	print('plotting here')
	# 				# 	axs[n].bar(value[n]['year'],value[n]['count'],width,label=f'{key} year epoch')

	# 				#sns.lineplot(x='year',y='count', data=value[n],ax=axs[n],hue='epoch_len',palette={epoch[0]:(0.79,0.41,0.02),epoch[1]:(0.36,0.11,0.41),epoch[2]:(0.18,0.55,0.34)},lw=2) 
	# 				axs[n].set_title(label='GEE tile ' + str(tile_labels[n]),fontsize=10) #make a y axis label that is the number of the month for that dataframe 
	# 				axs[n].set(ylabel='Valid pixel count')
	# 				axs[n].set(xlabel='End year of epoch')
	# 				axs[n].tick_params(axis='x', rotation=90)
	# 				axs[n].set_xticks(range(start_year,end_year+1))

	# 				handles, labels = axs[n].get_legend_handles_labels() #plot this is taking from is somewhat random. If they are fall filled in correctly this can be reset to n
	# 				fig.legend(handles, labels, loc='lower right')

	# 			else: 
	# 				print('fail')
	# 				print(key,value[n]['eetile2x15'].iloc[0])
	# 				continue
	# 		except IndexError: 
	# 			continue
	# #plt.legend()
	# plt.suptitle(f'{epoch[0]},{epoch[1]},{epoch[2]} epochs {start_year}-{end_year} valid pixel comparison', fontsize=10) #make a title.Change back when all of the things have been run for group 5
	# #plt.subplots_adjust(top=0.85)
	# plt.tight_layout(rect=[0,0.03,1,0.95])
	# plt.show()
	
	# out_path=Path(out_path)
	# #plt.savefig(out_path/f'{epoch[0]}_{epoch[1]}_{epoch[2]}_epochs_{start_year}_{end_year}_valid_pixel_comparison.png') #save the output

	# #close everything 
	# plt.close('all')