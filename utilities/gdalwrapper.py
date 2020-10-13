# Author: Guiming Zhang
# Last update: 8/1/2020
#http://www.paolocorti.net/2012/03/08/gdal_virtual_formats/
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow
#https://gdal.org/development/rfc/rfc26_blockcache.html


import gdal, gdalconst, glob, psutil
import os, sys, time
import random, math
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import conf

class VRTBuilder:
    '''
    '''
    def __init__(self):
        '''
        '''
    def buildVRT(self, srcFilelist, outVrt):
        vrt_options = gdal.BuildVRTOptions(separate=True, VRTNodata=-9999)
        gdal.BuildVRT(outVrt, srcFilelist, options=vrt_options)

class tiledRasterReader:
    '''
    '''
    def __init__(self, srcRasterfile, xoff=0, yoff=0, xsize=None, ysize=None):
        '''
        '''
        #print('Initializing reader...')
        self.srcRasterfile = srcRasterfile
        gdal.SetCacheMax(2**30) # 1 GB
        self.ds = gdal.Open(self.srcRasterfile, gdalconst.GA_ReadOnly)
        #print('self.ds: ', self.ds)

        if '.vrt' in self.srcRasterfile:
            self.fileList = self.ds.GetFileList()[1:]
            #print('self.fileList: ', self.fileList)
            self.measurement_level_ints = []
            for fn in self.fileList:
                # default level of measurement
                msrlevel = conf.MSR_LEVEL_RATIO
                for keyword in conf.NOMINAL_KEYWORD_IN_FN:
                    if keyword in fn:
                        msrlevel = conf.MSR_LEVEL_NOMINAL
                        break
                for key in conf.MSR_LEVELS:
                    if conf.MSR_LEVELS[key] == msrlevel:
                        self.measurement_level_ints.append(int(key))
                        break
            self.measurement_level_ints = np.array(self.measurement_level_ints)

        self.nbands = self.ds.RasterCount
        self.nrows = self.ds.RasterYSize
        self.ncols = self.ds.RasterXSize
        self.geotransform = self.ds.GetGeoTransform()
        self.projection = self.ds.GetProjection()
        print('%s:\n\t%d rows %d columns' % (self.srcRasterfile, self.nrows, self.ncols))

        band = self.ds.GetRasterBand(1)
        self.nodata = band.GetNoDataValue()

        ## each band may have a different nodata value
        nodatas = []
        for b in range(1, self.nbands+1):
            #print('band %d nodata: %.2f' % (b, self.ds.GetRasterBand(b).GetNoDataValue()))
            nodatas.append(self.ds.GetRasterBand(b).GetNoDataValue())
        self.nodatas = np.array(nodatas)

        '''
        for i in range(1, self.nbands + 1):
            b = self.ds.GetRasterBand(1)
            nd = b.GetNoDataValue()
            print('band %d nd %.2f' % (i, nd))
        '''
        self.block_ysize_base = band.GetBlockSize()[0]
        #print('self.fileList', self.fileList)
        if '.vrt' in self.srcRasterfile:
            self.block_xsize_base = gdal.Open(self.fileList[0], gdalconst.GA_ReadOnly).GetRasterBand(1).GetBlockSize()[0]
        else:
            #self.block_xsize_base = self.ds.GetRasterBand(1).GetBlockSize()[1]
            self.block_xsize_base = band.GetBlockSize()[1]

        #print('\t%d x %d' % (self.block_xsize_base, self.block_ysize_base))

        self.__N_TilesRead = 0
        self.xoff, self.yoff = xoff, yoff

        if xsize is None:
            self.xsize = self.block_xsize_base
        elif xsize > self.ncols:
            print('tile xsize exceeds RasterXsize %d' % self.ncols)
            sys.exit(1)
        else:
            self.xsize = xsize

        if ysize is None:
            self.ysize = self.block_ysize_base
        elif ysize > self.nrows:
            print('tile xsize exceeds RasterYsize %d' % self.nrows)
            sys.exit(1)
        else:
            self.ysize = ysize


        ## estimated data size (in MB)
        self.estimate_TotalSize_MB = self.estimateTileSize_MB(self.nrows, self.ncols)
        self.estimate_TileSize_MB = self.estimateTileSize_MB(self.xsize, self.ysize)

        # min, max, mean, stddev
        self.statistics = np.zeros((self.nbands, 4))
        for i in range(self.nbands):
            self.statistics[i] = self.ds.GetRasterBand(i+1).GetStatistics(0, 1)
            #self.statistics[i] = np.array([0, 1, 0, 1])

        self.MP_pool = None

        #print('Done initializing reader...')

    def estimateTileSize_MB(self, xsize=None, ysize=None):
        '''
        '''
        if xsize is None:
            xsize = self.xsize
        if ysize is None:
            ysize = self.ysize
        return np.array([1.0]).astype('float32').nbytes / 1024.0**2 * xsize * ysize  * self.nbands

    def readWholeRaster(self, multithread = conf.MULTITHREAD_READ):
        data  = None
        if multithread:
            def threadReadingByBand(i, rasterfile):
                ''' each thread reads a whole band
                    using multiprocess pool
                '''
                import gdal, gdalconst, psutil, conf
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray()
                return data

            # optimal for multi-thread reading by band
            n_threads = self.nbands
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)

            ## multi-thread reading by band
            band_idx = range(1, n_threads + 1)
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            data = self.MP_pool.map(threadReadingByBand, band_idx, fns)
            data = np.stack(data, axis=0)
            self.MP_pool.clear()
        else:
            data = self.ds.ReadAsArray(xoff=0, yoff=0, xsize=None, ysize=None)

        ## nodatavalues
        #if self.nodata < 0:
        #    data[data < self.nodata] = self.nodata
        return data

    def readNextTile(self, xsize=None, ysize=None, multithread = conf.MULTITHREAD_READ):
        ## update xsize and ysize if needed
        ## PLEASE specify xsize, ysize ONLY ONCE (when reading the first tile)
        if xsize is not None: self.xsize = xsize
        if ysize is not None: self.ysize = ysize

        N_BLOCK_X = int(math.ceil(self.ncols*1.0/self.xsize))
        y = int(self.__N_TilesRead / N_BLOCK_X)
        x = self.__N_TilesRead - y * N_BLOCK_X

        self.xoff = min(x * self.xsize, self.ncols)
        xsize = min(self.xsize, self.ncols - self.xoff)

        self.yoff = min(y * self.ysize, self.nrows)
        ysize = min(self.ysize, self.nrows - self.yoff)

        if self.xoff == self.ncols or self.yoff == self.nrows:
            return (None, self.xoff, self.yoff, 0, 0)

        data = None
        if multithread: ## multi-thread read
            def threadReadingByBand(i, param, rasterfile):
                ''' each thread reads a band, with tile dimension spec in param
                    using multiprocess pool
                '''
                import gdal, gdalconst
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray(xoff=param[0], yoff=param[1], win_xsize=param[2], win_ysize=param[3])
                return data

            # optimal for multi-thread reading by band
            n_threads = self.nbands# - 1
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)

            ## multi-thread reading by band
            params = []
            for i in range(n_threads):
                params.append([self.xoff, self.yoff, xsize, ysize])
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            band_idx = range(1, self.nbands + 1)
            data = self.MP_pool.map(threadReadingByBand, band_idx, params, fns)
            data = np.stack(data, axis=0)
            self.MP_pool.clear()

        else: ## single-thread read
            data = self.ds.ReadAsArray(xoff=self.xoff, yoff=self.yoff, xsize=xsize, ysize=ysize)
        ## nodatavalues
        #if self.nodata < 0:
        #    data[data < self.nodata] = self.nodata

        self.__N_TilesRead += 1

        return (data, self.xoff, self.yoff, xsize, ysize)

    def setNTilesRead(self, N):
        self.__N_TilesRead = N

    def readNextTileOverlap(self, xsize=None, ysize=None, overlap = 2, multithread = conf.MULTITHREAD_READ):
        ## update xsize and ysize if needed
        ## PLEASE specify xsize, ysize ONLY ONCE (when reading the first tile)
        if xsize is not None: self.xsize = xsize
        if ysize is not None: self.ysize = ysize

        N_BLOCK_X = int(math.ceil(self.ncols*1.0/self.xsize))
        y = int(self.__N_TilesRead / N_BLOCK_X)
        x = self.__N_TilesRead - y * N_BLOCK_X

        self.xoff = min(x * self.xsize, self.ncols)
        xsize = min(self.xsize, self.ncols - self.xoff)

        self.yoff = min(y * self.ysize, self.nrows)
        ysize = min(self.ysize, self.nrows - self.yoff)

        if self.xoff == self.ncols or self.yoff == self.nrows:
            return (None, self.xoff, self.yoff, 0, 0, -1, -1)

        data = None
        if multithread: ## multi-thread read
            def threadReadingByBand(i, param, rasterfile):
                ''' each thread reads a band, with tile dimension spec in param
                    using multiprocess pool
                '''
                import gdal, gdalconst
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray(xoff=param[0], yoff=param[1], win_xsize=param[2], win_ysize=param[3])
                return data

            # optimal for multi-thread reading by band
            n_threads = self.nbands# - 1
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)

            ## multi-thread reading by band
            params = []
            for i in range(n_threads):

                _xoff = max(0, self.xoff - overlap)
                _yoff = max(0, self.yoff - overlap)

                if _xoff == 0:
                    _xsize = min(xsize + overlap, self.ncols - self.xoff)
                else:
                    _xsize = min(xsize + 2 * overlap, self.ncols - self.xoff)

                if _yoff == 0:
                    _ysize = min(ysize + overlap, self.nrows - self.yoff)
                else:
                    _ysize = min(ysize + 2 * overlap, self.nrows - self.yoff)
                params.append([_xoff, _yoff, _xsize, _ysize])

                #params.append([self.xoff, self.yoff, xsize, ysize])
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            band_idx = range(1, self.nbands + 1)
            data = self.MP_pool.map(threadReadingByBand, band_idx, params, fns)
            data = np.stack(data, axis=0)
            self.MP_pool.clear()

        else: ## single-thread read
            _xoff = max(0, self.xoff - overlap)
            _yoff = max(0, self.yoff - overlap)

            if _xoff == 0:
                _xsize = min(xsize + overlap, self.ncols - self.xoff)
            else:
                _xsize = min(xsize + 2 * overlap, self.ncols - self.xoff)

            if _yoff == 0:
                _ysize = min(ysize + overlap, self.nrows - self.yoff)
            else:
                _ysize = min(ysize + 2 * overlap, self.nrows - self.yoff)

            #print('inside', self.xoff, self.yoff, self.xsize, self.ysize)
            #print('inside', _xoff, _yoff, _xsize, _ysize)
            data = self.ds.ReadAsArray(xoff=_xoff, yoff=_yoff, xsize=_xsize, ysize=_ysize)

            #data = self.ds.ReadAsArray(xoff=self.xoff, yoff=self.yoff, xsize=xsize, ysize=ysize)

        ## nodatavalues
        #if self.nodata < 0:
        #    data[data < self.nodata] = self.nodata

        self.__N_TilesRead += 1

        #return (data, _xoff, _yoff, _xsize, _ysize)
        return (data, self.xoff, self.yoff, xsize, ysize, _xoff, _yoff)

    def reset(self):
        ''' reset after reading tiles
        '''
        self.xoff, self.yoff = 0, 0
        self.__N_TilesRead = 0

    def extractByXY(self, x, y, xsize=1, ysize=1):
        ''' Extract raster value by x, y coordinates
        '''
        xoff = int((x - self.geotransform[0]) / self.geotransform[1])
        yoff = int((y - self.geotransform[3]) / self.geotransform[5])
        return self.ds.ReadAsArray(xoff, yoff, xsize, ysize)

    def extractByNbrhd(self, centerX, centerY, nbrXsize=1, nbrYsize=1):
        ''' Extract raster value by x, y coordinates
        '''
        xoff = int((x - self.geotransform[0]) / self.geotransform[1])
        yoff = int((y - self.geotransform[3]) / self.geotransform[5])
        return self.ds.ReadAsArray(xoff-int(nbrXsize/2), yoff-int(nbrYsize/2), nbrXsize, nbrYsize)

    def extractByNbrhd_batch(self, centerXs, centerYs, nbrXsize=1, nbrYsize=1):
        ''' Extract raster value by x, y coordinates
        '''
        xoffs = ((centerXs - self.geotransform[0]) / self.geotransform[1]).astype(int) - int(nbrXsize/2)
        yoffs = ((centerYs - self.geotransform[3]) / self.geotransform[5]).astype(int) - int(nbrYsize/2)

        data = None
        for xoff, yoff in zip(xoffs, yoffs):
            #print('Extracting NBRHD (%d, %d)' % (xoff, yoff))
            tmp = self.ds.ReadAsArray(xoff.item(), yoff.item(), nbrXsize, nbrYsize)

            #print(tmp.shape)
            tmp = np.expand_dims(tmp, axis=0)
            #print(tmp.shape)
            if data is None:
                data = tmp
            else:
                data = np.concatenate((data, tmp), axis=0)
            #print('data.shape:', data.shape)
        #print(data.shape)
        return data

    def extractByRC(c, r, xsize=1, ysize=1):
        '''Extract raster value by row, col
        '''
        return self.ds.ReadAsArray(c, r, xsize, ysize)

    def close(self):
        self.ds = None
        if self.MP_pool is not None:
            self.MP_pool.clear()

class tiledRasterWriter:
    '''
    '''
    def __init__(self, outRasterfile, nrows, ncols, nbands, geotransform, projection, nodata, dtype='float'):
        '''
        '''
        self.nrows, self.ncols, self.nbands, self.nodata = nrows, ncols, nbands, nodata
        driver = gdal.GetDriverByName('GTiff')
        if 'int16' in dtype:
            Dtype = gdal.GDT_Int16
        elif 'int32' in dtype:
            Dtype = gdal.GDT_Int32
        else:
            Dtype = gdal.GDT_Float32
        #self.ds = driver.Create(outRasterfile, self.ncols, self.nrows, self.nbands, Dtype, options = [ 'COMPRESS=LZW', 'BIGTIFF=YES' ])
        self.ds = driver.Create(outRasterfile, self.ncols, self.nrows, self.nbands, Dtype, options = [ 'COMPRESS=DEFLATE', 'BIGTIFF=YES' ])
        self.ds.SetGeoTransform(geotransform)
        self.ds.SetProjection(projection)

        self.bands = []
        for i in range(self.nbands):
            band = self.ds.GetRasterBand(i + 1)
            band.SetNoDataValue(self.nodata)
            self.bands.append(band)

    def WriteWholeRaster(self, data):
        if len(data.shape) == 2 and self.nbands == 1:
            self.bands[0].WriteArray(data, xoff=0, yoff=0)
            self.bands[0].FlushCache()
        elif len(data.shape) == 3 and self.nbands > 1 and data.shape[0] == self.nbands:
            for i in range(self.nbands):
                self.bands[i].WriteArray(data[i], xoff=0, yoff=0)
                self.bands[i].FlushCache()
        else:
            print('data dimension does not match raster dimension. exiting...')
            sys.exit(1)

        data = None

    def writeTile(self, data, xoff, yoff):
        if len(data.shape) == 2 and self.nbands == 1:
            self.bands[0].WriteArray(data, xoff=xoff, yoff=yoff)
            self.bands[0].FlushCache()
        elif len(data.shape) == 3 and self.nbands > 1 and data.shape[0] == self.nbands:
            for i in range(self.nbands):
                self.bands[i].WriteArray(data[i], xoff=xoff, yoff=yoff)
                self.bands[i].FlushCache()
        else:
            print('data dimension does not match raster dimension. exiting...')
            sys.exit(1)

        data = None

    ## Have to call this to write to disc
    def close(self):
        for band in self.bands:
            stats = band.GetStatistics(0, 1)
            #SetStatistics(double min, double max, double mean, double stddev)
            band.SetStatistics(stats[0], stats[1], stats[2], stats[3])

        for band in self.bands:
            band = None

        self.ds = None
