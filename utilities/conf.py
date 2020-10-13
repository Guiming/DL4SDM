# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: 10/13/2020
### this is a configuration file specifying various parameters

import sys, os
print('Python version: %s' % sys.version)

MULTITHREAD_READ = False ## read raster with multithreads

TILE_READ = False
## if None, will be determined automatically
TILE_XSIZE = None # 45036 ## setting specific to covariates_10m.vrt (BlockXsize of the underlying geotiff)
TILE_YSIZE = None #128 * 10 # None # 128 * 2 ## multiple of Blocksize in vrt
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow

#### debug mode
DEBUG_FLAG = False

N_INTERVALS = 300
N_HIST_BINS = 50

#### measurement level of environmental covariates
MSR_LEVELS = {'0':'nominal', '1':'ordinal', '2':'interval', '3':'ratio', '4':'count'}
MSR_LEVEL_NOMINAL = MSR_LEVELS['0']
MSR_LEVEL_ORDINAL = MSR_LEVELS['1']
MSR_LEVEL_INTERVAL = MSR_LEVELS['2']
MSR_LEVEL_RATIO = MSR_LEVELS['3']
MSR_LEVEL_COUNT = MSR_LEVELS['4']
NOMINAL_KEYWORD_IN_FN = ['geo', 'geology']
