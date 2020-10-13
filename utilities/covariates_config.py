# Author: Guiming Zhang
# Last update: 8/11/2020

## generate a json file containing layer index and variable names of the covariates
## to help with variable selection in modeling

import os, sys, json
import numpy as np
#sys.path.insert(0, 'utilities')
import gdalwrapper

class covariates_config:
    '''
    '''
    def __init__(self, config_json_fn = None):
        '''
        '''
        if config_json_fn is None: ## generate config json from scratch
            self.__generate_config_json()
        else: ## load config json from file
            with open(config_json_fn, 'r') as infile:
                self.config_json = json.load(infile)
        self.groups = []
        self.variables = []
        _selected = []

        for key in self.config_json.keys():
            group = self.config_json[key]['group']
            variable = self.config_json[key]['variable']
            selected = self.config_json[key]['selected']

            if group not in self.groups:
                self.groups.append(group)
            self.variables.append(variable)
            _selected.append(selected)

        self.selected = np.array(_selected)

    def set_by_group(self, group, selected = True):
        ''' Select (select = True) or de-select (select = False) a group of variables
        '''
        for key in self.config_json.keys():
            if self.config_json[key]['group'] == group:
                self.config_json[key]['selected'] = selected
                self.selected[int(key)] = selected

    def set_by_variable(self, variable, selected = True):
        ''' Select (select = True) or de-select (select = False) a variable
        '''
        for key in self.config_json.keys():
            if self.config_json[key]['variable'] == variable:
                self.config_json[key]['selected'] = selected
                self.selected[int(key)] = selected

    def set_by_index(self, index, selected = True):
        if index < 0 or index > len(self.variables) - 1:
            print('index out of range... there are only %d variables' % len(self.variables))
            sys.exit(1)
        self.config_json[index]['selected'] = selected
        self.selected[index] = selected

    def get_selected_index(self):
        ''' return index of selected variables
        '''
        index = np.array(range(1, len(self.variables)+1))
        return index[self.selected * index > 0] - 1

    def print_groups(self):
        '''
        '''
        print('there are %d groups of variables' % len(self.groups))
        for group in self.groups:
            print('\t%s' % group)

    def print_variables(self):
        '''
        '''
        print('there are %d variables in %d groups' % (len(self.variables), len(self.groups)))
        for idx, variable, selected in zip(range(len(self.variables)), self.variables, self.selected):
            print('\t%d: %s (group=%s) selected=%d' % (idx, variable, self.config_json[idx]['group'], selected))

    def write_config_file(self, config_json_fn = 'covs_config.json'):
        ''' write config to json file
        '''
        with open(config_json_fn, 'w') as outfile:
            json.dump(self.config_json, outfile)

    def __generate_config_json(self):
        ''' generate config json
        '''
        COV_BASE_DIR = 'D:/OneDrive - University of Denver/eBird/covariates/americas'

        COV_GROUP_FNS = ['bioclimatic_variables',\
                   'global_habitat_heterogeneity',\
                   'gpw_pop_density',\
                   'grip4_road_density',\
                   'landcover_prevalence',\
                   'topo_cont_vars_max',\
                   'topo_cont_vars_mean',\
                   'topo_cont_vars_median',\
                   'topo_cont_vars_min',\
                   'topo_cont_vars_std',\
                   'topo_landform_class_pcnt']

        ## Number of layers in each covariates group
        #N_BANDS = [19, 14, 5, 6, 12, 16, 16, 16, 16, 16, 15] ## TOTAL = 151
        N_BANDS = []
        STATS = []
        for COV_GROUP_FN in COV_GROUP_FNS:
            raster = gdalwrapper.tiledRasterReader(COV_BASE_DIR + os.sep + COV_GROUP_FN + '.tif')
            N_BANDS.append(raster.nbands)
            STATS.append(raster.statistics)
            #print('%s: %d bands' % (COV_GROUP_FN, raster.nbands), 'stats: ', raster.statistics)
        #print(N_BANDS)

        VAR_NAMES = [['BIO1 = Annual Mean Temperature',
                      'BIO2 = Mean Diurnal Range (Mean of Monthly (Max Temp - Min Temp))',
                      'BIO3 = Isothermality (BIO2/BIO7)(x100)',
                      'BIO4 = Temperature Seasonality (Standard Deviation ×100)',
                      'BIO5 = Max Temperature of Warmest Month',
                      'BIO6 = Min Temperature of Coldest Month',
                      'BIO7 = Temperature Annual Range (BIO5-BIO6)',
                      'BIO8 = Mean Temperature of Wettest Quarter',
                      'BIO9 = Mean Temperature of Driest Quarter',
                      'BIO10 = Mean Temperature of Warmest Quarter',
                      'BIO11 = Mean Temperature of Coldest Quarter',
                      'BIO12 = Annual Precipitation',
                      'BIO13 = Precipitation of Wettest Month',
                      'BIO14 = Precipitation of Driest Month',
                      'BIO15 = Precipitation Seasonality (Coefficient of Variation)',
                      'BIO16 = Precipitation of Wettest Quarter',
                      'BIO17 = Precipitation of Driest Quarter',
                      'BIO18 = Precipitation of Warmest Quarter',
                      'BIO19 = Precipitation of Coldest Quarter'],
                     ['HAB1 = Coefficient of Variation - Normalized Dispersion of EVI',
                     'HAB2 = Evenness - Evenness of EVI',
                     'HAB3 = Range - Range of EVI',
                     'HAB4 = Shannon - Diversity of EVI',
                     'HAB5 = Simpson - Diversity of EVI',
                     'HAB6 = Standard Deviation - Diversity of EVI',
                     'HAB7 = Contrast - Exponentially Weighted Difference in EVI between Adjacent Pixels',
                     'HAB8 = Correlation - Linear Dependency of EVI on Adjacent Pixels',
                     'HAB9 = Dissimilarity - Difference in EVI between Adjacent Pixels',
                     'HAB10 = Entropy - Disorderliness of EVI',
                     'HAB11 = Homogeneity - Similarity of EVI between Adjacent Pixels',
                     'HAB12 = Maximum - Dominance of EVI Combinations between Adjacent Pixels',
                     'HAB13 = Uniformity - Orderliness of EVI',
                     'HAB14 = Variance - Dispersion of EVI Combinations between Adjacent Pixels'],
                     ['POP1 = Population Density, v4.11 (2000)',
                      'POP2 = Population Density, v4.11 (2005)',
                      'POP3 = Population Density, v4.11 (2010)',
                      'POP4 = Population Density, v4.11 (2015)',
                      'POP5 = Population Density, v4.11 (2020)'],
                     ['ROD1 = Road Density for all roads, equally weighted',
                      'ROD2 = Road Density for GRIP Type 1 - Highways',
                      'ROD3 = Road Density for GRIP Type 2 - Primary Roads',
                      'ROD4 = Road Density for GRIP Type 3 - Secondary Roads',
                      'ROD5 = Road Density for GRIP Type 4 - Tertiary Roads',
                      'ROD6 = Road Density for GRIP Type 5 - Local Roads'],
                     ['LDC1 = Evergreen/Deciduous Needleleaf Trees - Percentage',
                      'LDC2 = Evergreen Broadleaf Trees - Percentage',
                      'LDC3 = Deciduous Broadleaf Trees - Percentage',
                      'LDC4 = Mixed/Other Trees - Percentage',
                      'LDC5 = Shrubs - Percentage',
                      'LDC6 = Herbaceous Vegetation - Percentage',
                      'LDC7 = Cultivated and Managed Vegetation - Percentage',
                      'LDC8 = Regularly Flooded Vegetation - Percentage',
                      'LDC9 = Urban/Built-up - Percentage',
                      'LDC10 = Snow/Ice - Percentage',
                      'LDC11 = Barren - Percentage',
                      'LDC12 = Open Water - Percentage'],
                     ['TOPOMAX1 = Elevation',
                      'TOPOMAX2 = Slope',
                      'TOPOMAX3 = Aspect Cosine',
                      'TOPOMAX4 = Aspect Sine',
                      'TOPOMAX5 = Aspect Eastness',
                      'TOPOMAX6 = Aspect Northness',
                      'TOPOMAX7 = Roughness',
                      'TOPOMAX8 = TPI - Topographic Position Index',
                      'TOPOMAX9 = TRI - Topographic Ruggedness Index',
                      'TOPOMAX10 = VRM - Vector Ruggedness Measure',
                      'TOPOMAX11 = DX - First Order Partial Derivative (E-W slope)',
                      'TOPOMAX12 = DXX - Second Order Partial Derivative (E-W slope)',
                      'TOPOMAX13 = DY - First Order Partial Derivative (N-S slope)',
                      'TOPOMAX14 = DYY - Second Order Partial Derivative (N-S slope)',
                      'TOPOMAX15 = PCURV - Profile Curvature',
                      'TOPOMAX16 = TCURV - Tangential Curvature'],
                     ['TOPOMEAN1 = Elevation',
                      'TOPOMEAN2 = Slope',
                      'TOPOMEAN3 = Aspect Cosine',
                      'TOPOMEAN4 = Aspect Sine',
                      'TOPOMEAN5 = Aspect Eastness',
                      'TOPOMEAN6 = Aspect Northness',
                      'TOPOMEAN7 = Roughness',
                      'TOPOMEAN8 = TPI - Topographic Position Index',
                      'TOPOMEAN9 = TRI - Topographic Ruggedness Index',
                      'TOPOMEAN10 = VRM - Vector Ruggedness Measure',
                      'TOPOMEAN11 = DX - First Order Partial Derivative (E-W slope)',
                      'TOPOMEAN12 = DXX - Second Order Partial Derivative (E-W slope)',
                      'TOPOMEAN13 = DY - First Order Partial Derivative (N-S slope)',
                      'TOPOMEAN14 = DYY - Second Order Partial Derivative (N-S slope)',
                      'TOPOMEAN15 = PCURV - Profile Curvature',
                      'TOPOMEAN16 = TCURV - Tangential Curvature'],
                     ['TOPOMEDIAN1 = Elevation',
                      'TOPOMEDIAN2 = Slope',
                      'TOPOMEDIAN3 = Aspect Cosine',
                      'TOPOMEDIAN4 = Aspect Sine',
                      'TOPOMEDIAN5 = Aspect Eastness',
                      'TOPOMEDIAN6 = Aspect Northness',
                      'TOPOMEDIAN7 = Roughness',
                      'TOPOMEDIAN8 = TPI - Topographic Position Index',
                      'TOPOMEDIAN9 = TRI - Topographic Ruggedness Index',
                      'TOPOMEDIAN10 = VRM - Vector Ruggedness Measure',
                      'TOPOMEDIAN11 = DX - First Order Partial Derivative (E-W slope)',
                      'TOPOMEDIAN12 = DXX - Second Order Partial Derivative (E-W slope)',
                      'TOPOMEDIAN13 = DY - First Order Partial Derivative (N-S slope)',
                      'TOPOMEDIAN14 = DYY - Second Order Partial Derivative (N-S slope)',
                      'TOPOMEDIAN15 = PCURV - Profile Curvature',
                      'TOPOMEDIAN16 = TCURV - Tangential Curvature'],
                     ['TOPOMIN1 = Elevation',
                      'TOPOMIN2 = Slope',
                      'TOPOMIN3 = Aspect Cosine',
                      'TOPOMIN4 = Aspect Sine',
                      'TOPOMIN5 = Aspect Eastness',
                      'TOPOMIN6 = Aspect Northness',
                      'TOPOMIN7 = Roughness',
                      'TOPOMIN8 = TPI - Topographic Position Index',
                      'TOPOMIN9 = TRI - Topographic Ruggedness Index',
                      'TOPOMIN10 = VRM - Vector Ruggedness Measure',
                      'TOPOMIN11 = DX - First Order Partial Derivative (E-W slope)',
                      'TOPOMIN12 = DXX - Second Order Partial Derivative (E-W slope)',
                      'TOPOMIN13 = DY - First Order Partial Derivative (N-S slope)',
                      'TOPOMIN14 = DYY - Second Order Partial Derivative (N-S slope)',
                      'TOPOMIN15 = PCURV - Profile Curvature',
                      'TOPOMIN16 = TCURV - Tangential Curvature'],
                     ['TOPOSTD1 = Elevation',
                      'TOPOSTD2 = Slope',
                      'TOPOSTD3 = Aspect Cosine',
                      'TOPOSTD4 = Aspect Sine',
                      'TOPOSTD5 = Aspect Eastness',
                      'TOPOSTD6 = Aspect Northness',
                      'TOPOSTD7 = Roughness',
                      'TOPOSTD8 = TPI - Topographic Position Index',
                      'TOPOSTD9 = TRI - Topographic Ruggedness Index',
                      'TOPOSTD10 = VRM - Vector Ruggedness Measure',
                      'TOPOSTD11 = DX - First Order Partial Derivative (E-W slope)',
                      'TOPOSTD12 = DXX - Second Order Partial Derivative (E-W slope)',
                      'TOPOSTD13 = DY - First Order Partial Derivative (N-S slope)',
                      'TOPOSTD14 = DYY - Second Order Partial Derivative (N-S slope)',
                      'TOPOSTD15 = PCURV - Profile Curvature',
                      'TOPOSTD16 = TCURV - Tangential Curvature'],
                     ['TOPOCAT1 = Flat Geomorphological Landforms - Percentage',
                      'TOPOCAT2 = Peak Geomorphological Landforms - Percentage',
                      'TOPOCAT3 = Ridge Geomorphological Landforms - Percentage',
                      'TOPOCAT4 = Shoulder Geomorphological Landforms - Percentage',
                      'TOPOCAT5 = Spur Geomorphological Landforms - Percentage',
                      'TOPOCAT6 = Slope Geomorphological Landforms - Percentage',
                      'TOPOCAT7 = Hollow Geomorphological Landforms - Percentage',
                      'TOPOCAT8 = Footslope Geomorphological Landforms - Percentage',
                      'TOPOCAT9 = Valley Geomorphological Landforms - Percentage',
                      'TOPOCAT10 = Pit Geomorphological Landforms - Percentage',
                      'TOPOCAT11 = Majority of Geomorphological Landforms',
                      'TOPOCAT12 = Count of Geomorphological Landforms',
                      'TOPOCAT13 = Shannon Index of Geomorphological Landforms',
                      'TOPOCAT14 = Entropy of Geomorphological Landforms',
                      'TOPOCAT15 = Uniformity of Geomorphological Landforms']]

        self.config_json = {}

        idx_group = 0
        idx_cov = 0
        for COV_GROUP_FN, N_BAND, STAT in zip(COV_GROUP_FNS, N_BANDS, STATS):
            for i in range(N_BAND):
                self.config_json[idx_cov] = {}
                self.config_json[idx_cov]['group'] = COV_GROUP_FN
                self.config_json[idx_cov]['variable'] = VAR_NAMES[idx_group][i]
                self.config_json[idx_cov]['min'] = STAT[i][0]
                self.config_json[idx_cov]['max'] = STAT[i][1]
                self.config_json[idx_cov]['mean'] = STAT[i][2]
                self.config_json[idx_cov]['std'] = STAT[i][3]
                self.config_json[idx_cov]['selected'] = True # include in modeling

                idx_cov += 1 # proceed to next variable

            idx_group += 1 # proceed to next group

def main():

    #config = covariates_config('data' + os.sep + 'covs_config.json')
    config = covariates_config()
    config.write_config_file('covs_config.json')
    config.print_groups()
    config.print_variables()
    #print(config.get_selected_index())

    config.set_by_group('gpw_pop_density', False)
    config.set_by_variable('TOPOCAT11 = Majority of geomorphological landforms', False)
    config.set_by_index(150, False)

    config.print_variables()
    print(config.get_selected_index().size, config.get_selected_index())

    tmp = np.ones(len(config.variables))
    print(tmp.sum())
    print(tmp[config.get_selected_index()].sum())

    config.write_config_file('covs_config.json')

if __name__ == "__main__":
    main()
