import requests 
from pathlib import Path 
import sys
#path = Path('/vol/v3/ben_ak/raster_files/dem/')

def get_dem(path): 
	for i in range(52,73): #iterate through rows (lat)53,73
		for j in range(128,181): #iterate through cols (lon)128,181
			print(i,j)
			image_url = f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/2/TIFF/n{i}w{j}/USGS_2_n{i}w{j}.tif"
			  
			# URL of the image to be downloaded is defined as image_url 
			# req = requests.head(image_url,allow_redirects=True)
			# header = req.headers
			# content_type = header.get('content-type')
			# if content_type.lower() =='application/xml': 
			# 	print('its a file,downloading...')
			r = requests.get(image_url) # create HTTP response object 
		  
			# send a HTTP request to the server and save 
			# the HTTP response in a response object called r 
			with open(path/f"USGS_2_n{i}w{j}.tif",'wb') as f: 

			  
			    # Saving received content as a png file in 
			    # binary format 
			  
			    # write the contents of the response (r.content) 
			    # to a new file in binary mode. 
			    f.write(r.content) 
			# else: 
			# 	print('its not a file, skipping...')
			#print(req.headers['content-length'])	
			

def main(): 
	# # script param
	script = sys.argv[0]
	# first command line param
	path = sys.argv[1]
	out_path = Path(path)
	get_dem(out_path)
if __name__ == '__main__':
    main()
