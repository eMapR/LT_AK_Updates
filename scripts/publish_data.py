import sys,os
from multiprocessing import Pool
import subprocess as sp
from glob import glob
from osgeo import gdal, gdal_array
import yaml
from writeNetCDF import writeNetCDF
import mosaic_tiles_fix

AGGREGATION_MODES = {
    'avg': ('avg','average', 'Average value of resampled points from original 30 m resolution dataset.'),
    'med': ('med','med', 'Median value of resampled points from original 30 m resolution dataset.'),
    'mode': ('mode', 'mode', 'Most frequent (mode) value of resample points from original 30 m resolution dataset.')
    }

# Make and return the desired output path
def get_out_path(ftp_dir, basename, stat):
    if not os.path.exists(ftp_dir):
        os.mkdir(ftp_dir)
    if not os.path.exists(ftp_dir+'/'+basename):
        os.mkdir(ftp_dir+'/'+basename)
    out_dir = ftp_dir+'/'+basename+'/'+stat
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    return out_dir

# Helper function to call shell commands
def callSilent(cmd, sout=sp.PIPE):
    s = sp.Popen(cmd, stdout=sout, stderr=sp.STDOUT, shell=True)
    out = s.communicate()

# Helper function with zip command
def callZip(c, sout=sp.PIPE):
    print "Zipping ", c[1]
    cmd = 'ln -s {fn} {arcname}.tif \n zip -j {arcname} {arcname}.tif \n rm {arcname}.tif'.format(fn=c[0], arcname=c[1])
    callSilent(cmd)


# Zip up individual years of 30 m data and put them in the FTP directory
def zip_to_ftp(P):

    data_dir = P['data_dir']
    ftp_dir = P['ftp_dir']
    stat = P['stat']
    basename = P['name']

    # Make directories if non existant
    out_dir = get_out_path(ftp_dir, basename, stat)

    # Loop through years in data dir, zip up tif and save to FTP
    cmds = []
    for fn in glob(data_dir+'/????/*'+stat+'*.tif'):
        year = fn.split('/')[-2]
        arcname = '{0}/{1}_{2}_{3}'.format(out_dir, basename, year, stat)
        if not os.path.exists(arcname+'.zip') or P['overwrite']:
            cmds.append((fn, arcname))

    par = Pool(P['n_jobs'])
    par.map(callZip, cmds)


# Make a VRT with each year of data as a separate band
def build_composite(data_dir, basename, stat, years):
    vrtfile = '{0}/{1}_{2}.vrt'.format(data_dir, basename, stat)
    tifs = ''
    for y in range(years[0], years[1]+1):
        tifs += ' ' + glob(data_dir+'/'+str(y)+'/*'+stat+'*.tif')[0]
    callSilent('gdalbuildvrt -separate {0} {1}'.format(vrtfile, tifs))
    return vrtfile

# Read data and then write a NetCDF in nc4
def convertToNC4(infile, years, V):
    ds = gdal.Open(infile)
    data = ds.GetVirtualMemArray()
    outfile = infile.replace('tif', 'nc4')
    writeNetCDF(data, ds, outfile, years, variables=V)

# Make a composite of all years and resample given a warping template
def composite_and_resample(P, warpcmd, name):

    data_dir = P['data_dir']
    ftp_dir = P['ftp_dir']
    stat = P['stat']
    basename = P['name']

    # Make an output directory for this resampling
    out_dir = get_out_path(ftp_dir, basename, stat)
    out_dir += '/'+name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Build a composite VRT to resample
    vrtfile = build_composite(data_dir, basename, stat, P['years'])

    # If continuous, provide distribution stats from the resampling, otherwise just give the Mode

    Q = [AGGREGATION_MODES[P['aggregate']]]
    if P['quartiles']:
        Q= Q + [
         ('q0',  'min',     'q0: 0th Quartile (minimum) value of resampled points from original 30 m resolution dataset.'),
         ('q1',  'q1',      'q1: 1st Quartile value of resampled points from original 30 m resolution dataset.'),
         ('q2',  'med',     'q2: 2nd Quartile (median) value of resampled points from original 30 m resolution dataset.'),
         ('q3',  'q3',      'q3: 3rd Quartile value of resampled points from original 30 m resolution dataset.'),
         ('q4',  'max',     'q4: 4th Quartile (maximum) value of resampled points from original 30 m resolution dataset.')
        ]


    # Resample VRT into tifs
    print "Resampling", name
    cmds = []
    for q in Q:
        method_name, method, desc = q
        outfile = '{0}/{1}_{2}_{3}_{4}.tif'.format(out_dir, basename, stat, method_name, name)
        if not os.path.exists(outfile) or P['overwrite']:
            cmds.append(warpcmd.format(r=method, src=vrtfile, dest=outfile))
    par = Pool(P['n_jobs'])
    par.map(callSilent, cmds)

    # Convert tifs to NetCDF
    print "Converting to netCDF"
    for q in Q:
        method_name, method, desc = q
        outfile = '{0}/{1}_{2}_{3}_{4}.tif'.format(out_dir, basename, stat, method_name, name)

        V = [P]
        V[0]['desc'] += ' '+desc
        years = range(P['years'][0], P['years'][1]+1)
        convertToNC4(outfile, years, V)


# Provide the 'PRISM' reprojection -- 0.5 minute pixels over CONUS
def resample_to_prism(P):
    warpcmd = 'gdalwarp -co "TILED=YES" -co "BIGTIFF=YES" -co "INTERLEAVE=BAND" -multi -r {r} -t_srs EPSG:4269 -te -125.0208333 24.0625104 -66.4791900 49.9375000 -tr 0.00833333 -0.00833333 {src} {dest}'
    composite_and_resample(P, warpcmd, 'prism')

# Provide the MsTMIP reprojection -- 0.5 degree pixels over the globe
def resample_to_MsTMIP(P):
    #TODO
    warpcmd = 'gdalwarp -multi -r {r} -t_srs EPSG:37008 -te -180 -90 180 90 -tr 0.5 0.5  {src} {dest}'
    composite_and_resample(P, warpcmd, 'MsTMIP')


def main(params, build_mosiacs=False, overwrite=False):
    with open(params, 'r') as infile:
        M = yaml.safe_load(infile)

    baseparams = M.copy()
    del baseparams['variables']

    for k in M['variables']:
        P = baseparams.copy()
        P.update(M['variables'][k])
        P['stat'] = k
        P['desc'] = P['desc'] +' '+ P['description']

        if build_mosiacs:
            mosaic_tiles_fix.main(P['data_dir'], P['dtype'], 'all', k)

        if not 'overwrite' in P: P['overwrite'] = overwrite
        if not 'n_jobs' in P: P['n_jobs'] = 2

        print k
        zip_to_ftp(P)
        resample_to_prism(P)
        resample_to_MsTMIP(P)



if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
