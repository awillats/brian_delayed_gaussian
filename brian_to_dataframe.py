import pandas as pd

# This section creates a "flat" dataframe where nested index 
# of population and channel are encoded simply in the string of the column name
# i.e. v2_5 for population 2, channel 5

def hier_df_from_lists(high_level_list, lower_level_list):
    'creates an empty dataframe with nested column indices'
    multi_idx = pd.MultiIndex.from_product( [high_level_list, lower_level_list])
    df = pd.DataFrame(columns = multi_idx)
    return df
    
def expand_volt_monitor_to_df_columns(data_dict, df=None, channel_name=''):
        '''
    expands the voltage variable from (num_time x num_channels) numpy array
    to pandas dataframe with column names which represent hierarchy 
    (even though the column indices are in-fact flat):
    | vA_1 | vA_2 | vA_3 | vA_4 | vB_1 | vB_2 | vB_4 | vB_4 | 
    -----------------
    '''
    if df is None:
        df = pd.DataFrame()
    df['t'] = data_dict['t']
    n_channels = data_dict['v'].shape[1]
    v_names = [f'v{channel_name}_{i}' for i in range(n_channels)]
    for idx,v_name in enumerate(v_names):
        df[v_name] = data_dict['v'][:,idx]    
    return df

def expand_volt_monitor_to_hier_df(data_dict, group_name, df=None):
    '''
    expands the voltage variable from (num_time x num_channels) numpy array
    to columns nested under a group name in a pandas dataframe:
    | group_name_A |
    | 0 | 1 | 2 | 3 |
    -----------------
    '''
    if df is None:
        df = pd.DataFrame()    
    n_channels = data_dict['v'].shape[1]
    v_names = [i for i in range(n_channels)]
    for idx,v_name in enumerate(v_names):
        df[(group_name,v_name)] = data_dict['v'][:,idx]    
        
    return df


#%%
# This code gets the final states, but doesn't unpack the entire timeseries
# df= Ga.get_states(units=False,format='pandas')
# df.head()
if __name__ == "__main__":
    print('To-do: write simple demo script here')