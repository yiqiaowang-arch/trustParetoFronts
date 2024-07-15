import geopandas as gpd
import pandas as pd
import numpy as np
import calliope

"""
set a class of building, which contains the information of building, including
building id as string, area as real, building location as a dictionary {lat: real, lon: real}
building emission system temperature as integer
the following are in an array of 8760 elements, each element is a real number:
appliance and demand, heating demand, cooling demand, hot water demand,
PV generation, PVT generation, solar collector generation
I want to specify only the name of building and path of the folder that contains the data
and the class will automatically read the data and store them in the class
the data is stored in multiple subfolders of the path, some are for demand, some are for generation
the data is stored in csv files, each file contains the data of one building
the file name is the building id, or builidng id + PV, or building id + PVT, or building id + sc
"""


class Building:
    def __init__(self, name, scenario_path):
        self.name: str = name
        self.scenario_path: str = scenario_path

        # get type of emission system
        emission_dict = {'HVAC_HEATING_AS1': 80, 'HVAC_HEATING_AS4': 45}
        air_conditioning = gpd.read_file(self.scenario_path + r'\inputs\building-properties'
                                         + '\\' 'air_conditioning.dbf')
        air_conditioning.index = air_conditioning['Name']
        self.emission_type: str = air_conditioning.loc[self.name, 'type_hs']
        self.emission_temp: int = emission_dict[self.emission_type]

        # get building area
        zone = gpd.read_file(self.scenario_path + r'\inputs\building-geometry' + '\\' 'zone.shp')
        zone.index = zone['Name']
        self.area: float = zone.loc[self.name, 'geometry'].area
        self.location: dict = {'lat': zone.loc[self.name, 'geometry'].centroid.y,
                               'lon': zone.loc[self.name, 'geometry'].centroid.x}
        self.get_demand_supply()
        

    def get_demand_supply(self):
        """
        Description:
        This method reads the input scenario_path, following the CEA result file structure, finds the pre-computed 
        result csvs for demand (electricity (E), heating (Qhs), cooling (Qcs) and hot water (Qww)),
        along with supply from PV, PVT and flat-panel solar collectors (SC_FP). 
        Currently, each timeseries is an independent dataframe. TODO: merge all dataframes into one!
        """
        demand_path = self.scenario_path + r'\outputs\data\demand'
        demand_df = pd.read_csv(demand_path + '\\' + self.name + '.csv')
        # set index as date, but the date is not yet in datetime format so it needs to be converted
        demand_df.set_index(pd.to_datetime(demand_df['DATE']), inplace=True)
        supply_path = self.scenario_path + r'\outputs\data\potentials\solar'
        pv_df = pd.read_csv(supply_path + '\\' + self.name + '_PV.csv')
        # delete the last 6 strings of the date, which is unnecessary
        pv_df.set_index(pd.to_datetime(pv_df['Date'].str[:-6]), inplace=True)
        pvt_df = pd.read_csv(supply_path + '\\' + self.name + '_PVT.csv')
        pvt_df.set_index(pd.to_datetime(pvt_df['Date'].str[:-6]), inplace=True)
        sc_df = pd.read_csv(supply_path + '\\' + self.name + '_SC_FP.csv')
        sc_df.set_index(pd.to_datetime(sc_df['Date'].str[:-6]), inplace=True)

        # time series data
        # read demand data
        demand_df = demand_df[['E_sys_kWh', 'Qhs_sys_kWh', 'Qcs_sys_kWh', 'Qww_sys_kWh']]

        self.app: pd.DataFrame = - demand_df[['E_sys_kWh']].astype('float64').rename(columns={'E_sys_kWh': self.name})
        self.sh: pd.DataFrame = - demand_df[['Qhs_sys_kWh']].astype('float64').rename(columns={'Qhs_sys_kWh': self.name})
        self.sc: pd.DataFrame = - demand_df[['Qcs_sys_kWh']].astype('float64').rename(columns={'Qcs_sys_kWh': self.name})
        self.dhw: pd.DataFrame = - demand_df[['Qww_sys_kWh']].astype('float64').rename(columns={'Qww_sys_kWh': self.name})

        # read supply data
        self.pv: pd.DataFrame = pv_df[['E_PV_gen_kWh']].astype('float64').rename(columns={'E_PV_gen_kWh': self.name})
        self.pvt_e: pd.DataFrame = pvt_df[['E_PVT_gen_kWh']].astype('float64').rename(columns={'E_PVT_gen_kWh': self.name})
        self.pvt_h: pd.DataFrame = pvt_df[['Q_PVT_gen_kWh']].astype('float64').rename(columns={'Q_PVT_gen_kWh': self.name})
        self.scfp: pd.DataFrame = sc_df[['Q_SC_gen_kWh']].astype('float64').rename(columns={'Q_SC_gen_kWh': self.name})
        self.pv_intensity: pd.DataFrame = self.pv.astype('float64') / self.area
        self.pvt_e_intensity: pd.DataFrame = self.pvt_e.astype('float64') / self.area
        self.pvt_h_intensity: pd.DataFrame = self.pvt_h.astype('float64') / self.area
        # devide pvt_h with pvt_e element-wise to get relative intensity, which is still a dataframe.
        # replace NaN and inf with 0
        df_pvt_h_relative_intensity = self.pvt_h_intensity.divide(self.pvt_e_intensity[self.name], axis=0).fillna(0)
        df_pvt_h_relative_intensity.replace(np.inf, 0, inplace=True)
        self.pvt_h_relative_intensity: pd.DataFrame = df_pvt_h_relative_intensity.astype('float64')
        self.scfp_intensity: pd.DataFrame = self.scfp.astype('float64') / self.area
        # change all timeseries data's index column name to 't'
        self.app.index.names = ['t']
        self.sh.index.names = ['t']
        self.sc.index.names = ['t']
        self.dhw.index.names = ['t']
        self.pv.index.names = ['t']
        self.pvt_e.index.names = ['t']
        self.pvt_h.index.names = ['t']
        self.scfp.index.names = ['t']
        self.pv_intensity.index.names = ['t']
        self.pvt_e_intensity.index.names = ['t']
        self.pvt_h_intensity.index.names = ['t']
        self.pvt_h_relative_intensity.index.names = ['t']
        self.scfp_intensity.index.names = ['t']


    def set_building_specific_config(self, 
                                     building_specific_config: calliope.AttrDict, 
                                     building_status: pd.Series) -> calliope.AttrDict:
        """
        Description:
        This function sets the building specific configuration for the building model.
        - If the building is not in the district heating area, delete the district heating technologies keys.
        - If the building is only renovated (not rebuilt), set the original GSHP and ASHP costs to 0
            because they are already installed.
            Also, if they want to change to another technology, set the costs higher because 
            changing heating system in an existing building costs more.
        - If the building is rebuilt, set the costs to normal.
        - This function assumes that no ASHP is used for Dhw in any original buildings.

        Inputs:
        - self:                     Building object
        - building_specific_config: calliope.AttrDict
        - building_status:          pd.Series

        Outputs:
        - building_specific_config: calliope.AttrDict (modified)
        """
        name = self.name
        # if building is not in district heating area, delete the district heating technologies keys
        if not building_status['is_disheat']:
            building_specific_config.del_key(f'locations.{name}.techs.DHDC_small_heat')
            building_specific_config.del_key(f'locations.{name}.techs.DHDC_medium_heat')
            building_specific_config.del_key(f'locations.{name}.techs.DHDC_large_heat')
            
        # if building is not rebuilt, set GSHP and ASHP costs higher
        if building_status['is_new']:
            if building_status['is_rebuilt']: # rebuilt, so everything is possible and price is normal
                pass
            else: # renovated, can do GSHP and ASHP but price higher
                if building_status['already_GSHP']: # already has GSHP, only need to set ASHP price higher, and GSHP price to 0
                    building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.purchase', 0)
                    building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.energy_cap', 0)
                    
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.purchase', 18086)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.energy_cap', 1360)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.purchase', 18086)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.energy_cap', 1360)
                elif building_status['already_ASHP']: # ASHP for heating no cost; but ASHP for DHW higher; also GSHP higher
                    building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.purchase', 39934)
                    building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.energy_cap', 1316)
                    
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.purchase', 0)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.energy_cap', 0)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.purchase', 18086)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.energy_cap', 1360)
                else: # no GSHP and no ASHP, set both to higher price
                    building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.purchase', 39934)
                    building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.energy_cap', 1316)
                    
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.purchase', 18086)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.energy_cap', 1360)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.purchase', 18086)
                    building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.energy_cap', 1360)
        else: # not new, so no new GSHP but new ASHP allowed; however if they are already with GSHP or ASHP, then no corresponding cost is applied
            if building_status['already_GSHP']:
                building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.purchase', 0)
                building_specific_config.set_key(f'locations.{name}.techs.GSHP_heat.costs.monetary.energy_cap', 0)
                
                building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.purchase', 18086)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.energy_cap', 1360)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.purchase', 18086)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.energy_cap', 1360)
            elif building_status['already_ASHP']: # no previous GSHP, so delete GSHP keys; 
                building_specific_config.del_key(f'locations.{name}.techs.GSHP_heat')
                building_specific_config.del_key(f'locations.{name}.techs.GSHP_cooling')
                building_specific_config.del_key(f'locations.{name}.techs.geothermal_boreholes')
                
                building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.purchase', 0)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.energy_cap', 0)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.purchase', 18086)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.energy_cap', 1360)
            else: # no previous GSHP and no previous ASHP, so delete GSHP keys and higher ASHP keys
                building_specific_config.del_key(f'locations.{name}.techs.GSHP_heat')
                building_specific_config.del_key(f'locations.{name}.techs.GSHP_cooling')
                building_specific_config.del_key(f'locations.{name}.techs.geothermal_boreholes')
                
                building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.purchase', 18086)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP.costs.monetary.energy_cap', 1360)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.purchase', 18086)
                building_specific_config.set_key(f'locations.{name}.techs.ASHP_DHW.costs.monetary.energy_cap', 1360)
                
        return building_specific_config


    def get_building_model( self, yaml_path: str, store_folder: str,
                                building_status: pd.Series = pd.Series(), flatten_spikes=False, flatten_percentile=0.98, 
                                to_lp=False, to_yaml=False, 
                                obj='cost',
                                emission_constraint=None):
        """
        Description:
        This function gets building parameters and read the scenario files to create a calliope model for the building.

        Input:
        building_name:              str, the name of the building
        building_scenario_folder:   str, the folder that contains the building's scenario files
        yaml_path:                  str, the path to the yaml file that contains the energy hub configuration
        store_folder:               str, the folder that stores the results
        building_status:            pd.Series, the status of the building, including is_new, is_rebuilt, already_GSHP, already_ASHP, is_disheat
        flatten_spikes:             bool, if True, flatten the demand spikes
        flatten_percentile:         float, the percentile to flatten the spikes
        to_lp:                      bool, if True, store the model in lp format
        to_yaml:                    bool, if True, store the model in yaml format
        obj:                        str, the objective function, either 'cost' or 'emission'
        emission_constraint:        float, the emission constraint

        Return:
        Model:                      calliope.Model, the optimized model
        """
        if flatten_spikes:
            self.flatten_spikes_demand(percentile=flatten_percentile) # flatten the demand spikes
        dict_timeseries_df = {'demand_el': self.app,
                            'demand_sh': self.sh,
                            'demand_dhw': self.dhw,
                            'demand_sc': self.sc,
                            'supply_PV': self.pv_intensity,
                            'supply_PVT_e': self.pvt_e_intensity,
                            'supply_PVT_h': self.pvt_h_relative_intensity,
                            'supply_SCFP': self.scfp_intensity
                            }
        # modify the building_specific_config to match the building's status
        building_specific_config: calliope.AttrDict = calliope.AttrDict.from_yaml(yaml_path)
        building_specific_config.set_key(key='locations.Building.available_area', value=self.area)
        print('the area of building '+self.name+' is '+str(self.area)+' m2')
        building_sub_dict = building_specific_config['locations'].pop('Building')
        building_specific_config['locations'][self.name] = building_sub_dict
        
        # update geothermal storage max capacity
        building_specific_config.set_key(key=f'locations.{self.name}.techs.geothermal_boreholes.constraints.energy_cap_max',
                                        value=(self.area+400)*0.1) # assume 100W/m2 max yield
        if building_status is not None:
            building_specific_config = self.set_building_specific_config(building_specific_config=building_specific_config, building_status=building_status)
            # set the wood energy cap max to be 0.5W/m2 times building area +400 m2
            building_specific_config.set_key(key=f'locations.{self.name}.techs.wood_supply.constraints.energy_cap_max', value=(self.area+400)*0.5*0.001)
        # # test: delete ASHP
        # building_specific_config.del_key(f'locations.{self.name}.techs.ASHP')
        
        # if emission constraint is not None, add it to the building_specific_config
        if emission_constraint is not None:
            building_specific_config.set_key(key='group_constraints.systemwide_co2_cap.cost_max.co2', value=emission_constraint)
        
        # if obj is cost, set the objective to be cost; if obj is emission, set the objective to be emission
        if obj == 'cost':
            building_specific_config.set_key(key='run.objective_options.cost_class.monetary', value=1)
            building_specific_config.set_key(key='run.objective_options.cost_class.co2', value=0)
        elif obj == 'emission':
            building_specific_config.set_key(key='run.objective_options.cost_class.monetary', value=0)
            building_specific_config.set_key(key='run.objective_options.cost_class.co2', value=1)
        else:
            raise ValueError('obj must be either cost or emission')
        # print current objective setting
        print(building_specific_config.get_key('run.objective_options.cost_class'))
        model = calliope.Model(building_specific_config, timeseries_dataframes=dict_timeseries_df)
        if to_lp:
            model.to_lp(store_folder+'/'+self.name+'.lp')
        if to_yaml:
            model.save_commented_model_yaml(store_folder+'/'+self.name+'.yaml')
        return model
    

    def get_pareto_front(self, epsilon:int, building_name: str, building_scenario_folder: str, 
                         yaml_path: str, store_folder: str,
                         building_status: pd.Series = pd.Series(), flatten_spikes=False, flatten_percentile=0.98,
                         to_lp=False, to_yaml=False):
        """
        Description:
        This function finds the pareto front of one building regarding cost and emission.
        - First, it finds the emission-optimal solution and store the cost and emission in df_pareto.
        - Then, it finds the cost-optimal solution and store the cost and emission in df_pareto.
        - Then it reads the number of epsilon cuts 
            and evenly distribute the emissions between the cost-optimal and emission-optimal solutions.
        - For each epsilon, it finds the epsilon-optimal solution and store the cost and emission in df_pareto.
        - Finally, it returns the df_pareto, which contains two columns: first cost, second emission. 
            Along with index of number of epsilon cut. 0: emission-optimal, epsilon+1: cost-optimal.
        - It also returns the df_tech_cap_pareto, which contains the technology capacities of each solution.

        Inputs:
        - epsilon:                  int, the number of epsilon cuts between cost-optimal and emission-optimal solutions
        - building_name:            str, the name of the building
        - building_scenario_folder: str, the folder that contains the building's scenario files
        - yaml_path:                str, the path to the yaml file that contains the energy hub configuration
        - store_folder:             str, the folder that stores the results
        - building_status:          pd.Series, the status of the building, including is_new, is_rebuilt, already_GSHP, already_ASHP, is_disheat
        - flatten_spikes:           bool, if True, flatten the demand spikes
        - flatten_percentile:       float, the percentile to flatten the spikes
        - to_lp:                    bool, if True, store the model in lp format
        - to_yaml:                  bool, if True, store the model in yaml format
        
        Outputs:
        - df_pareto:                pd.DataFrame, the pareto front of the building, with cost and emission as columns
        - df_tech_cap_pareto:       pd.DataFrame, the technology capacities of each solution
        """

        df_pareto = pd.DataFrame(columns=['cost', 'emission'], index=range(epsilon+2))
        # read yaml file and get the list of technologies
        tech_list = calliope.AttrDict.from_yaml(yaml_path).get_key(f'locations.Building.techs').keys()
        df_tech_cap_pareto = pd.DataFrame(columns=tech_list, index=range(epsilon+2))
        # first get the emission-optimal solution
        model_emission = self.get_building_model(yaml_path=yaml_path, store_folder=store_folder, 
                                                 building_status=building_status, flatten_spikes=flatten_spikes, 
                                                 flatten_percentile=flatten_percentile, to_lp=to_lp, to_yaml=to_yaml, 
                                                 obj='emission')
        model_emission.run()
        model_emission.to_netcdf(path=store_folder + '/' + building_name+'_emission.nc')
        print('emission is done')
        # store the cost and emission in df_pareto
        df_emission = model_emission.get_formatted_array('cost').sel(locs=building_name).to_pandas().transpose().sum(axis=0)
        # add the cost and emission to df_pareto
        df_pareto.loc[0] = [df_emission['monetary'], df_emission['co2']]
        # store the technology capacities in df_tech_cap_pareto
        df_tech_cap_pareto.loc[0] = model_emission.get_formatted_array('energy_cap').to_pandas().iloc[0]
        
        # then get the cost-optimal solution
        model_cost = self.get_building_model(yaml_path=yaml_path, store_folder=store_folder, 
                                             building_status=building_status, flatten_spikes=flatten_spikes, 
                                             flatten_percentile=flatten_percentile, to_lp=to_lp, to_yaml=to_yaml, 
                                             obj='cost')
        # run model cost, and find both cost and emission of this result
        model_cost.run()
        model_cost.to_netcdf(path=store_folder  + '/' + building_name+'_cost.nc')
        print('cost is done')
        # store the cost and emission in df_pareto
        # add epsilon name as row index, start with epsilon_0
        df_cost: pd.DataFrame = model_cost.get_formatted_array('cost').sel(locs=building_name).to_pandas().transpose().sum(axis=0) # first column co2, second column monetary
        # add the cost and emission to df_pareto
        df_pareto.loc[epsilon+1] = [df_cost['monetary'], df_cost['co2']]
        # store the technology capacities in df_tech_cap_pareto
        df_tech_cap_pareto.loc[epsilon+1] = model_cost.get_formatted_array('energy_cap').to_pandas().iloc[0]
        # based on epsilon numbers, create empty rows in df_pareto for further filling

        # then get the epsilon-optimal solution
        # first find out min and max emission, and epsilon emissions are evenly distributed between them
        # if cost and emission optimal have the same emission, then there's no pareto front
        if df_cost['co2'] <= df_emission['co2']:
            print('cost-optimal and emission-optimal have the same emission, no pareto front')
            self.df_pareto = df_pareto
        else:
            emission_max =df_cost['co2']
            emission_min =df_emission['co2']
            # calculate the interval between two emissions
            interval = (emission_max - emission_min) / (epsilon+1)
            # for each epsilon, get the epsilon-optimal solution
            for i in range(1, epsilon+1):
                print(f'starting epsilon {i}')
                # set the emission constraint to be emission_min + i * interval
                emission_constraint = emission_min + i * interval
                model_epsilon = self.get_building_model(yaml_path=yaml_path, store_folder=store_folder, 
                                                        building_status=building_status, flatten_spikes=flatten_spikes, 
                                                        flatten_percentile=flatten_percentile, to_lp=to_lp, to_yaml=to_yaml, 
                                                        obj='cost', emission_constraint=emission_constraint)
                model_epsilon.run()
                model_epsilon.to_netcdf(path=store_folder  + '/' + building_name + f'_epsilon_{i}.nc')
                print(f'epsilon {i} is done')
                # store the cost and emission in df_pareto
                df_epsilon = model_epsilon.get_formatted_array('cost').sel(locs=building_name).to_pandas().transpose().sum(axis=0)
                # add the cost and emission to df_pareto
                df_pareto.loc[i] = [df_epsilon['monetary'], df_epsilon['co2']]
                # store the technology capacities in df_tech_cap_pareto
                df_tech_cap_pareto.loc[i] = model_epsilon.get_formatted_array('energy_cap').to_pandas().iloc[0]
                
            df_pareto = df_pareto.merge(df_tech_cap_pareto, left_index=True, right_index=True)
            df_pareto = df_pareto.astype({'cost': float, 'emission': float})
            self.df_pareto = df_pareto
            self.df_tech_cap_pareto = df_tech_cap_pareto

        
    def flatten_spikes(self, df: pd.DataFrame, column_name, percentile: float = 0.98, is_positive: bool = False):
        # first fine non-zero values of the given column of the given dataframe
        # then calculate the 95th percentile of the non-zero values
        # then find the index of the values that are greater than the 98th percentile
        # then set the values of the index to the 95th percentile
        # then return the dataframe
        # the input dataframe should have a datetime index
        if not is_positive:
            df = - df

        nonzero_subset = df[df[column_name] != 0]
        percentile_value = nonzero_subset[column_name].quantile(percentile)
        df.loc[df[column_name] > percentile_value, column_name] = percentile_value

        if not is_positive:
            df = - df

        return df

    def flatten_spikes_demand(self, percentile:float = 0.98):
        self.app = self.flatten_spikes(self.app, self.name, percentile, is_positive=False)
        self.sh = self.flatten_spikes(self.sh, self.name, percentile, is_positive=False)
        self.sc = self.flatten_spikes(self.sc, self.name, percentile, is_positive=False)
        self.dhw = self.flatten_spikes(self.dhw, self.name, percentile, is_positive=False)
