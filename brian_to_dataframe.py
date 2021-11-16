'''
Open issues:
- Does hierarchical table construction depend on an equal number of neurons per group?
    - it shouldn't have to
'''
#%%

import pandas as pd


#%%
def volt_monitors_to_hier_df(all_monitors, group_names, neuron_names):
    dfh = hier_df_from_lists(group_names, neuron_names)
    for idx,mon in enumerate(all_monitors):
        data_dict = mon.get_states(['t','v'], units=False)
        dfh = expand_volt_monitor_to_hier_df(data_dict, group_name=group_names[idx], df=dfh)

    dfh['time [ms]'] = data_dict['t']
    return dfh

def hier_df_from_lists(high_level_list, lower_level_list):
    '''
    creates an empty dataframe with nested column indices
    each combination from high_level and lower_level become a column header
    '''
    multi_idx = pd.MultiIndex.from_product( [high_level_list, lower_level_list])
    df = pd.DataFrame(columns = multi_idx)
    return df
    
def expand_numpy_to_hier_df(np_ary, group_names, neuron_names, df=None):
    if df is None:
        df = hier_df_from_lists(group_names, neuron_names)
    for gi,g in enumerate(group_names):
        for ni,n in enumerate(neuron_names):
            df[(g,n)] = np_ary[:, len(neuron_names)*gi + ni]
    return df
    
def expand_volt_monitor_to_hier_df(data_dict, group_name, df=None):
    '''
    expands the voltage variable from (num_time x num_channels) numpy array
    to columns nested under a group name in a pandas dataframe:
    
    | Population_A  | Population_B  |
    | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
    ---------------------------------
    '''
    if df is None:
        df = pd.DataFrame()    
    n_channels = data_dict['v'].shape[1]
    v_names = [i for i in range(n_channels)]
    for idx,v_name in enumerate(v_names):
        df[(group_name,v_name)] = data_dict['v'][:,idx]    
        
    return df

def expand_volt_monitor_to_df_columns(data_dict, df=None, channel_name=''):
    '''
    expands the voltage variable from (num_time x num_channels) numpy array
    to pandas dataframe with column names which represent hierarchy 
    (even though the column indices are in-fact flat):
    
    | vA_1 | vA_2 | vA_3 | vA_4 | vB_1 | vB_2 | vB_4 | vB_4 | 
    ---------------------------------------------------------
    '''
    if df is None:
        df = pd.DataFrame()
    df['t'] = data_dict['t']
    n_channels = data_dict['v'].shape[1]
    v_names = [f'v{channel_name}_{i}' for i in range(n_channels)]
    for idx,v_name in enumerate(v_names):
        df[v_name] = data_dict['v'][:,idx]    
    return df

def melt_hier_df_voltage(df, output_var='voltage'):
    # using hierarchical ID for faceting 
    dfm = df.melt(id_vars='time [ms]',var_name=['population','neuron'],value_name=output_var)
    # Create a combined id column to facet by
    dfm['total_neuron_idx'] = dfm['population']+ dfm['neuron'].astype(str)
    return dfm
#%%
# This code gets the final states, but doesn't unpack the entire timeseries
# df= Ga.get_states(units=False,format='pandas')
# df.head()
if __name__ == "__main__":
    print('To-do: write simple demo script here')