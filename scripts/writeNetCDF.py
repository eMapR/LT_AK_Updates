import os, sys
import numpy as np
if 'GDAL_DATA' not in os.environ:
    os.environ['GDAL_DATA'] = r'/usr/lib/anaconda/share/gdal'
from osgeo import gdal
import netCDF4

NP2GDAL_DATATYPE = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
gdal.UseExceptions()


def writeNetCDF(data, src_ds, fn, years, variables=[], nodata=None, color_table=None, legend=None, CO=["FORMAT=NC4", "WRITE_LONLAT=YES"]):
    if data.ndim < 4:
        data.shape = tuple([1]*(4-data.ndim) + list(data.shape))
    num_vars, bands, Ysize, Xsize = data.shape

    if not nodata:
        nodata = src_ds.GetRasterBand(1).GetNoDataValue()

    DataType = NP2GDAL_DATATYPE[data.dtype.name]

    # Use GDAL to set up a temp NetCDF data file from which to extract spatial info
    driver = gdal.GetDriverByName('NetCDF');
    ncds = driver.Create('/tmp/_T'+os.path.basename(fn), Xsize, Ysize, 1, DataType, CO)
    ncds.SetMetadata( src_ds.GetMetadata() )
    ncds.SetProjection( src_ds.GetProjectionRef() )
    ncds.SetGeoTransform( src_ds.GetGeoTransform() )
    del ncds

    # Use the netCDF4 module to create variables (subdatasets) and add the 3d data
    nco = netCDF4.Dataset(fn, 'w')

    ## Copy Lat, Lon, and CRS Data into new dataset
    dsin = netCDF4.Dataset('/tmp/_T'+os.path.basename(fn))
    #Copy dimensions
    for dname, the_dim in dsin.dimensions.iteritems():
        nco.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
    # Copy variables
    for v_name, varin in dsin.variables.iteritems():
        if v_name == 'Band1': continue
        #print dir(varin)
        outVar = nco.createVariable(v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        try:
            outVar[:] = varin[:]
        except TypeError:
            pass #Happens when varin[] is empty
    # close the output file
    dsin.close()
    #Destroy temp file
    os.unlink('/tmp/_T'+os.path.basename(fn))

    # Build the Time Dimension and Variable
    nco.createDimension('time',bands)
    timeo = nco.createVariable('time','i2',('time'))
    timeo.units = 'years since 1970-08-01'
    timeo.standard_name = 'time'
    timeo.long_name = 'time'
    timeo[:] = np.array(years, dtype='int16')-1970

    nco.sync()

    # Add each of the variables to the dataset
    for i in range(num_vars):
        V = variables[i]
        ncvar = nco.createVariable(V['name'], data.dtype, ('time', 'lat', 'lon'), chunksizes=[1,256,256], fill_value=nodata)
        if 'units' in V:
            ncvar.units = V['units']
        if 'desc' in V:
            ncvar.description = V['desc']
        if 'scale' in V:
            ncvar.scale_factor = V['scale']
        #ncvar[:] =  np.flipud(np.squeeze(data[i,:,:,:]))
        ncvar[:] =  np.squeeze(data[i,:,::-1,:])
        nco.sync()

    nco.close()
