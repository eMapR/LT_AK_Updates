import sys,os
from multiprocessing import Pool
import subprocess as sp
from glob import glob
from osgeo import gdal

GDAL_DATATYPES = {
  'byte': gdal.GDT_Byte,
  'uint8': gdal.GDT_Byte,
  'uint16': gdal.GDT_UInt16,
  'int16': gdal.GDT_Int16,
  'uint32': gdal.GDT_UInt32,
  'int32': gdal.GDT_Int32,
  'float32': gdal.GDT_Float32,
  'float64': gdal.GDT_Float64
  }

######
#base_dir = "/vol/v3/lt_stem_v3.1/models/biomassfiaald_20180708_0859"
#target_dtype = 'Int16'
#stat = 'mean'
#####

def callSilent(cmd, sout=sp.PIPE):
    s = sp.Popen(cmd, stdout=sout, stderr=sp.STDOUT, shell=True)
    #out = s.communicate()
    s.wait()

def mosaic(tiles_dir, stat, target_dtype, name):

    print '{n}: {s} -> {d}'.format(n=name,s=stat,d=target_dtype)
    sys.stdout.flush()

    STANDARDIZE_TMP = 'gdal_translate -of VRT -ot '+target_dtype+' {fn} {vrt}'
    ASSEMBLE_TMP = 'gdalbuildvrt -overwrite "{tiles_dir}../{name}_{stat}.vrt" {vrt_dir}*{stat}.vrt'
    MOSAIC_TMP = 'gdal_translate -co "TILED=YES" -co "INTERLEAVE=BAND" -co "BIGTIFF=YES" {tiles_dir}../{name}_{stat}.vrt {tiles_dir}/../{name}_{stat}.tif'

    vrt_dir = tiles_dir + 'vrts/'
    if not os.path.exists(vrt_dir):
        os.mkdir(vrt_dir)

    for fn in glob(tiles_dir+'*'+stat+'.tif'):
        basefn = os.path.basename(fn)
        callSilent( STANDARDIZE_TMP.format(fn=fn, vrt=vrt_dir + basefn.replace('.tif', '.vrt')) )

    sys.stdout.write('. Mosaicking: ')
    sys.stdout.flush()
    callSilent(ASSEMBLE_TMP.format(tiles_dir=tiles_dir, vrt_dir=vrt_dir, name=name, stat=stat))
    callSilent(MOSAIC_TMP.format(tiles_dir=tiles_dir, name=name, stat=stat), None)
    print ''

def par_mosaic(args):
    mosaic(*args)

def main(base_dir, target_dtype, year, *stats):

    if year.lower() == 'all': year = '????'

    if base_dir[-1] == '/': base_dir = base_dir[:-1]

    if isinstance(stats, basestring):
        stats = [stats]

    cmds = []
    print base_dir+'/'+year+'/_temp_tiles/'
    for T in glob(base_dir+'/'+year+'/_temp_tiles/'):
        n = '_'.join(T.split('/')[-4:-2])
        for S in stats:
            #mosaic(T, S, target_dtype, n)
            cmds.append((T, S, target_dtype, n))

    par = Pool(4, maxtasksperchild=1)
    par.map(par_mosaic, cmds)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
