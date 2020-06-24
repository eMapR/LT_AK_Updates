'''lthacks.py
(Formerly validation_funs.py)

Created by David Miller (dmil1991@gmail.com)
Updated by Tara Larrue (tlarrue2991@gmail.com)

Miscellaneous functions that are useful for map pixel extraction,
validation of LandTrendr outputs, and output and script organization.
'''
import os, sys, csv, gdal, math, glob, subprocess
import numpy as np
from gdalconst import *
from numpy.lib.recfunctions import append_fields
from sklearn.metrics import confusion_matrix
import pickle 
from datetime import datetime
import getpass

SCENES_DIR = "/vol/v1/scenes"

def txtToDict(txtfile):

	txt = open(txtfile, 'r')

	dictionary = {}
	for line in txt:
		comps = line.split(":")
		dictionary[int(comps[0])] = comps[1].strip()

	return dictionary

def getLastCommit(scriptPath):
    '''Returns last git commit hash, user, and time of specified script.'''
    cwd = os.getcwd()
    os.chdir(os.path.dirname(scriptPath))

    cmd = "git log -1 --pretty='%h %cn %N %cd' -- " + os.path.basename(scriptPath)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        if line == '' and p.poll() != None:
            break
    
    lastCommit = ''.join(stdout).strip()
    os.chdir(cwd)

    return lastCommit

def createMetadata(arguments, outputPath_data, altMetaDir=None, description=None, lastCommit="UNKNOWN"):
    '''Creates a meta.txt file describing a dataset. 
    Add this function to any script that produces significant data.'''

    timestamp = datetime.now().strftime('%Y%m%d %H:%M:%S')
    user = getpass.getuser()
    commandline = " ".join(arguments)
    if altMetaDir:
        metaPath = os.path.join(altMetaDir, os.path.splitext(os.path.basename(outputPath_data))[0] + "_meta.txt")
    else:
        metaPath = os.path.splitext(outputPath_data)[0] + "_meta.txt"

    f = open(metaPath, "w")
    if description:
        f.write(description + "\n\n")
    f.write("FULL DATA PATH: " + os.path.realpath(outputPath_data))
    f.write("\nCREATED BY: " + os.path.realpath(arguments[0]))
    f.write("\nSCRIPT LAST COMMIT: " + lastCommit)
    f.write("\nCOMMAND USED TO CREATE: " + commandline)
    f.write("\nTIME CREATED: " + timestamp)
    f.write("\nUSER: " + user)
    f.close()
    
    return metaPath


def write_params_to_meta(meta_path, param_path):
    ''' Add each line of a parameter file to a metadata text file'''
    with open(param_path) as f:
        lines = ['\t' + l for l in f]
    
    with open(meta_path, 'a') as f:
        f.write('\nPARAMETERS:\n')
        for l in lines:
            f.write(l)


def loadPickle(path):
    with open(path, 'rb') as handle:
        ds = pickle.load(handle)
    return ds

def savePickle(ds, path):
    with open(path, 'wb') as handle:
        pickle.dump(ds, handle)
    if os.path.exists(path):
        print ("\nNew data structure pickled: ", path)

def csvToArray(filepath, names=True):
    '''converts CSV file to structured numpy array - MUST NOT HAVE COMMAS WITHIN ENTRIES!'''
    f = open(filepath, 'rb')
    data = np.genfromtxt(f, delimiter=',', names=names, case_sensitive=False, dtype=None) #structured array of strings
    f.close()
    return data

def arrayToCsv(array, outpath):
    '''saves a structured numpy array with headers as a CSV'''
    if array.dtype.names:
        np.savetxt(outpath, array, delimiter=",", header=",".join(i for i in array.dtype.names), comments="", fmt='%s')
    else:
        np.savetxt(outpath, array, delimiter=",", fmt='%s')
    if os.path.exists(outpath):
        print ("\nNew File Saved:", outpath)

def extractTSArows(inputData, tsas, tsa_col="TSA"):
    '''extract rows in inputData (structured array) that match given TSAs'''
    #convert tsas to compatible format with array data
    tsa_list_4dig = [fourDigitTSA(i) for i in tsas]
    if np.issubdtype(inputData.dtype[tsa_col], np.number):
        tsa_list_4dig = [int(i) for i in tsa_list_4dig]

    #extract rows that are equal to any of TSAs
    outputData = inputData[np.any([inputData[tsa_col] == i for i in tsa_list_4dig],0)]
    return outputData

def extract_kernel(spec_ds,x,y,width,height,band,transform):
    # Modified original code from Zhiqiang Yang (read_spectral) at Oregon State University
    """read spectral value from band centered around [x,y] with width and height"""
    xoffset = int(x - transform[0])/30 - width/2
    yoffset = int(y - transform[3])/-30 - height/2

    # plot is outside the image boundary
    if xoffset <0 or yoffset > spec_ds.RasterYSize - 1:
        return [-9999]
    this_band = spec_ds.GetRasterBand(band)
    specs = this_band.ReadAsArray(xoffset, yoffset, width, height)
    return specs

def extract_kernel_and_coords(spec_ds,x,y,width,height,band,transform):
    """read spectral value from band centered around [x,y] with width and height. 
    Also returns corresponding coordinates."""
    xoffset = int(x - transform[0])/30 - width/2
    yoffset = int(y - transform[3])/-30 - height/2

    x_indeces = numpy.arange(xoffset, xoffset+width)
    y_indeces = numpy.arange(yoffset, yoffset+height)
    x_coords = x_indeces * transform[1] + transform[0] 
    y_coords = y_indeces * transform[5] + transform[3] 
    all_coords = numpy.zeros([x_coords.size,y_coords.size,2])
    for ind, i in enumerate(x_coords):
        for jnd, j in enumerate(y_coords):
            all_coords[jnd,ind] = (i,j) 

    # plot is outside the image boundary
    if xoffset <0 or yoffset > spec_ds.RasterYSize - 1:
        return [-9999]
    this_band = spec_ds.GetRasterBand(band)
    specs = this_band.ReadAsArray(xoffset, yoffset, width, height)
    return specs, all_coords 
   
def getStatFunc(astring, options=None):
	'''Returns a statistical function from a "stat string". '''
	
	astring = astring.strip(' ').lower()

	if astring == 'mean':
		def func(anarray):
			return np.mean(anarray)
			
	elif astring == 'max':
		def func(anarray):
			return np.max(anarray)
			
	elif astring == 'median':
		def func(anarray):
			return np.median(anarray)
			
	elif astring == 'mode':
		def func(anarray):
			return mode(mode(anarray)[0][0])
			
	elif astring == 'min':
		def func(anarray):
			return np.min(anarray)
			
	elif astring == "num_pix_with_data":
		def func(anarray):
			return (anarray != options).sum()
			
	elif astring == "num_pix_equal":
		def func(anarray):
			return (anarray == options).sum()
			
	elif astring == "num_pix_between":
		def func(anarray):
			inds = np.where(np.logical_and(anarray>=options[0], anarray<options[1]))
			return len(inds[0])
			#return ((anarray >= options[0]) and (anarray <= options[1])).sum()
			
	elif astring == "stdev":
		def func(anarray):
			#mean = np.mean(anarray)
			#mean_array = np.zeros(anarray.shape)
			#mean_array[:] = mean
			#return np.mean(np.square(anarray - mean_array))
			return np.std(anarray)
			
	elif astring == "mid_pix":
		def func(anarray):
			anarray = np.array(anarray)
			middle = lambda x: x[[slice(np.floor(d/2.), np.ceil(d/2.)) for d in x.shape]]
			return middle(anarray)[0]
			
	else:
		print (sys.exit("Stat input not understood:"+ astring))
	
	return func 

def sixDigitTSA(pathrow):
    """converts TSA to 6-digit string for searching directories"""
    # pass pathrow, first coerce to string if not already
    if type(pathrow) != str: pathrow = str(pathrow)
    # check length, and make TSA six digit
    # e.g. for 4529, --> 045029
    pathrow = pathrow.strip()
    if len(pathrow) < 4:
        sys.exit("Enter TSA with at least 4 digits")
    elif len(pathrow) == 4:
        pathrow = '0' + pathrow[:2] + '0' + pathrow[2:]
    elif len(pathrow) == 5:
        if pathrow[0] == '0':
            pathrow = pathrow[:3] + '0' + pathrow[3:]
        elif pathrow[2] == '0':
            pathrow = '0' + pathrow
        else:
            sys.exit("Provide TSA of form PPRR e.g. 4529")
    return pathrow

def fourDigitTSA(pathrow):
    """converts TSA to 4-digit string for lookup in a CSV table"""
    pathrow6 = sixDigitTSA(pathrow)
    pathrow4 = pathrow6[1:3] + pathrow6[4:]
    return pathrow4

def findTSA(tsa_ref_mask, x_coord, y_coord):
    '''returns 6-digit Landsat TSA as string for given coordinates'''
    ds = gdal.Open(tsa_ref_mask)
    transform = ds.GetGeoTransform()
    tsa = extract_kernel(ds, x_coord, y_coord, 1, 1, 1, transform)[0][0]
    
    return sixDigitTSA(tsa)
    
def expandPathRows(sceneSets):
    '''takes in list of scene sets, returns list of 6 digit scene numbers'''
    sceneList = []
    for i in sceneSets:
        scenePart = i.split('/')
        if '-' in scenePart[0]:
            rng = scenePart[0].split('-')
            paths = range(int(rng[0]), int(rng[1])+1)
        else:
            paths = [str(scenePart[0])]

        if '-' in scenePart[1]:
            rng = scenePart[1].split('-')
            rows = range(int(rng[0]), int(rng[1])+1)
        else:
            rows = [str(scenePart[1])]

        for row in rows:
            for path in paths:
                pathRow = str(path) + str(row)
                sceneList.append(pathRow)

    return [sixDigitTSA(i) for i in sceneList]

def getLTFile(pathrow, search_strings):
    '''Finds file within LT scenes directory'''
    pathrow = sixDigitTSA(pathrow)
    topdir = os.path.join(SCENES_DIR, pathrow)
    filelist = []
    for i in search_strings:
        filelist.extend(glob.glob(os.path.join(topdir,i)))

    if len(filelist) == 0:
        sys.exit("No applicable files found for search strings: '" + "' ; '".join(search_strings) + "'")
    elif len(filelist) > 1:
        print ("2 files found for search strings: '" + "' ; '".join(search_strings) + "'")
        print ("Choosing first file found: " + filelist[0])
        ltfile = filelist[0]
    else:
        ltfile = filelist[0]

    return ltfile

def appendSumKernels(csvData, columnPrefixes):
    '''Calculate the sum of matching pixels from different maps. Maps indicated by columnPrefixes. 
    Append sum as column to structured array'''

    #examine column headers for common kernels
    headers = []
    kernels = []
    for ind,prefix in enumerate(columnPrefixes):
        headers.append(list(filter(lambda x: x.startswith(prefix.upper()), csvData.dtype.names)))
        kernels.append([])
        for i in headers[ind]:
            kernels[ind].append(i.split(prefix)[1])

    #confirm common kernels for all maps, then append sum for each kernel
    swap = np.transpose(kernels)
    check_common = all(all(x==swap[i][0] for x in swap[:][i]) for i in range(len(kernels[0])))

    if check_common:
        #append new headers for sum calculation
        addHeaders = ["_".join(["SUM"] + columnPrefixes + [i.strip("_")]) for i in kernels[0]]
        csvData = append_fields(csvData, addHeaders, data=[np.zeros(csvData.size) for i in addHeaders], dtypes='f8')

        for ind,row in enumerate(csvData):
            for h in addHeaders:
                headers_to_sum = filter(lambda x: x.endswith(h[-1]), np.asarray(headers).flatten())
                sum = 0
                for i in headers_to_sum: sum += int(row[i])
                csvData[h][ind] = sum

    else:
        sys.exit("Cannot append Kernel Sum; Headers in unfamiliar format.")

    return csvData, '_'.join(addHeaders[0].split('_')[:-1])

def appendMetric(csvData, metric, columnPrefix, options=None):
    '''Append a metric (mean,median, etc. calculated from all fields starting wtih columnPrefix)
     column to structured array'''

    #append new header for metric calculations
    columnHeaders = list(filter(lambda x: x.startswith(columnPrefix.upper()), csvData.dtype.names))
    addHeader = metric.upper() + "_" + columnPrefix.upper()
    csvData = append_fields(csvData, addHeader, data=np.zeros(csvData.size), dtypes='f8')
    
    #get stat function
    func = getStatFunc(metric, options)

    #calculate indicated metric & populate csv data
    columns = csvData[columnHeaders].copy()
    for ind,row in enumerate(columns):
        row_list = [int(i) for i in row]
        
        csvData[addHeader][ind] = func(row_list)

    return csvData

def makeConfusion(y_test, predictions, classes):
    '''Creates a confusion matrix & calculated producers, users & overall accuracies.
    All inputs are array-like type.'''
    cm = confusion_matrix(y_test, predictions, labels=classes)
    numPred = np.sum(cm,axis=0).astype('f8')
    numTruth = np.sum(cm,axis=1).astype('f8')
    dtypes = [(' ', 'a25')] + [(str(i),'f8') for i in classes] + [('No. Truth', 'a25'), ('Producers Accuracy', 'f8')] #horizontal labels
    full_cm = np.zeros(cm.shape[0]+3, dtype=dtypes) #structured array
    full_cm[' '] = [str(i) for i in classes] + ['No. Predictions', 'Users Accuracy', 'KAPPA'] #vertical labels
    totalCorrect = 0
    for ind,i in enumerate(classes):
        numCorrect = float(cm[ind,ind])
        totalCorrect += numCorrect
        full_cm[str(i)] = list(cm[:,ind]) + [numPred[ind], numCorrect/numPred[ind]] + [None]
        full_cm['Producers Accuracy'][ind] = numCorrect/numTruth[ind] 
    full_cm['No. Truth'] = list(numTruth.astype('a25')) + [str(np.sum(cm)), 'Overall'] + [None]
    observedAccuracy = totalCorrect/np.sum(cm)
    full_cm['Producers Accuracy'][-3:-1] = [None, observedAccuracy]

    marginalFreq = (numPred*numTruth).astype('f8')/np.sum(cm).astype('f8')
    print (marginalFreq)
    expectedAccuracy = np.sum(marginalFreq).astype('f8')/np.sum(cm).astype('f8')
    kappa = (observedAccuracy - expectedAccuracy)/(1- expectedAccuracy)
    full_cm[classes[0]][-1] = kappa

    return full_cm

def makeConfusion_diffLabels(data, truthCol, predictionCol):
    '''Creates a confusion matrix for datasets w/ different truth & prediction labels. 
    Does NOT calculate users/producers accuracy. truthCol/predictionCol are strings.'''
    truthLabels = np.unique(data[truthCol]) 
    predictionLabels = np.unique(data[predictionCol])
    confusion = np.zeros((truthLabels.size+1, predictionLabels.size+1)).astype('str')
    confusion[:,0] =  [""] + list(truthLabels)
    confusion[0,:] =  [""] + list(predictionLabels)

    #populate confusion matrix
    for row in data:
        x = np.where(confusion == row[truthCol])[0]
        y = np.where(confusion == row[predictionCol])[1]
        confusion[x,y] = str(float(confusion[x,y][0]) + 1)

    return confusion



    
