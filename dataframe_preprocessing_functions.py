import pandas as pd
import numpy as np

#%%
# Additional helper functions
def null_last_row(df):
    '''
    called to break up transitions between series for plotly based on melted dataframe
    '''
    #assumes time column is on the far right side
    df.iloc[-1,:-1]=None
    return df

def time_selection_df(df,time_range, time_name = 'time [ms]'):
    return df[(df[time_name]>time_range[0]) & (df[time_name]<time_range[1])]

def compare_df(df1, df2, tail_amount):
    '''
    return differences in the last <tail_amount> samples of a dataframe 
    '''
    df1_tail = df1.tail(tail_amount).reset_index(drop=True)
    df2_tail = df2.tail(tail_amount).reset_index(drop=True)
    
    return df1_tail.compare(df2_tail)
#%%
zscore_fn = lambda x: (x-np.nanmean(x,axis=0))/np.nanstd(x,axis=0)
# zscore_fn = lambda x: (x-np.nanmin(x,axis=0))/np.ptp(x,axis=0)

def zscore_df_cols(df, cols_to_norm):
    df[cols_to_norm] = df[cols_to_norm].apply(zscore_fn)
    return df
    
def zscore_df_cols_except(df, cols_to_exclude):
    # df.loc[:,df.columns!=cols_to_exclude] = df.loc[:,df.columns!=cols_to_exclude].apply(zscore_fn)
    cols_to_norm = df.drop(cols_to_exclude,axis=1).columns
    return zscore_df_cols(df,cols_to_norm)
    
def cross_function_df(df, col_names, func_ij):
    '''
    applies a function across each of the pairs of columns of a DataFrame 
    - e.g. cross-correlation 
    '''
    N = len(col_names)
    
    #initialize xcorr dataframe from product of group names
    dfx = pd.DataFrame(columns=pd.MultiIndex.from_product( [col_names, col_names]).set_names(['from','to']))

    for i in range(N):
        for j in range(N):
            this_output = func_ij(df, i, j)
            dfx[(col_names[i],col_names[j])] = this_output
            
    dfx['lag [ms]'] = df['time [ms]'] - df['time [ms]'].mean()
    return dfx
    
    
def corr_df(df,i,j):
    '''
    computes the cross correlation between two columns of a dataframe 
    (normalized by the number of samples)
    NOTE: this method assumes (and doesn't check) that all samples in the frame are from a uniform spacing in time
    '''
    return np.correlate(df.iloc[:-1,i], df.iloc[:-1,j], 'same')/len(df.iloc[:-1,i])
    # 
    
    
def xcorr_df_func(df,i,j, do_sub_auto_corr=False):
    xc= corr_df(df,i,j)
    # if do_sub_auto_corr:
    if i==j or do_sub_auto_corr:
        xc -= corr_df(df,i,i)   
    return xc