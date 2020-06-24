import os 
import sys
import pyParz
import glob
import json
from osgeo import gdal
from lthacks_py3 import createMetadata

def get_files(search_dir): 
	# Returns a list of names in list files. 
	files = glob.glob(search_dir+'**/*temp.tif', recursive = True) 
	print(len(files))
	return files

def resample(args): 
    """Downscale resolution for faster processing. This is hardcoded for AK and landsat data."""
    #input_file,resolution,temp_dest,epsg = args
    #head,tail = os.path.split(input_file)
    input_file,resolution,epsg,resample_method = args
    #remove 'temp.tif' and write new files. Temp files will eventually be deleted.
    output_file=input_file[:-9]+f'_{resolution}m_resampled_final.tif'
    print(output_file)
    if os.path.exists(output_file): 
        pass

    else:   
        ds = gdal.Open(input_file)
        ds = gdal.Warp(output_file, input_file, xRes=int(resolution), yRes=int(resolution), dstSRS=epsg,resampleAlg='cubic')
        print('made file')
    #ds.Close()
    ds = None
    return output_file 

def remove_temp_files(fileList): 
	"""Remove temp files from the resample function."""
	# Iterate over the list of filepaths & remove each file.
	for file in fileList:
	    try:
	        print('deleting...')
	        os.remove(file)
	    except OSError:
	        print("Error while deleting file")
	return None

def main(): 
	params = sys.argv[1]
	#print(params)
	with open(str(params)) as f:
		variables = json.load(f)

		#construct variables from param file
		search_dir = variables["search_dir"]
		resolution = variables["resolution"]
		epsg = variables["epsg"]
		resample_method = variables["resample_method"]
	resample_files = get_files(search_dir)
	output = pyParz.foreach(resample_files,resample,args=[resolution,epsg,resample_method],numThreads=20)
	remove_temp_files(resample_files)
	#lthacks.createMetadata(sys.argv, out_filepath)	

if __name__ == '__main__':
	main()
		#sys.exit(main(*sys.argv[1:]))