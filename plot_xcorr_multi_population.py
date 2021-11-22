import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brian_to_dataframe import *
from ring_buffers import *
from circuit_helpers import *

%matplotlib inline
%load_ext autoreload
%autoreload 2

'''
rename XCORR dataframe!
- then can send to Matt
- print other params to title!!

ERROR: NotImplementedError: Multiple "summed variables" target the variable "I_in" in group "neurongroup_2". Use multiple variables in the target group instead.

- stop dash app from executing from notebook
------
tau blurs xcorr peaks out to wider lags!

longer sims means easier xcorr decoding 
'''

# %%
start_scope()
duration = 2000*ms
dt = defaultclock.dt;
def time2index(t):
    return np.round(t/dt).astype(int)

    
min_buffer_len = time2index(0*ms)

N_groups = 5 
N_neurons = 1
N_total = N_groups*N_neurons
neuron_names = range(N_neurons)
tau = 0.5*10*ms
base_sigma = 1

def ij_to_flat_index(gi, ni, N_high=N_groups, N_low=N_neurons):
    return gi*N_low + ni
# def flat_index_to_ij(fi, N_high=N_groups, N_low=N_neurons):
    # return 

#%%
base_weight = 2 / N_neurons; #0.001 = null, 1=medium-strong, 3=strong, 0.9 for reciprocal
base_delay = 20*ms #tau * 0.5
base_delay_samp = time2index(base_delay)


Weights = np.zeros((N_groups,N_groups))
# Weights[0][1] = base_weight 
Weights[0][1] = base_weight 
Weights[1][2] = base_weight 
Weights[4][3] = base_weight
# Weights[][4] = base_weight

def is_ij_valid(i,j):
    return Weights[i][j] != 0
Delays = np.ones((N_groups, N_groups)) * base_delay
Delays[Weights==0] = 0
# Delays[1][0] = 3.5*base_delay
# Delays[0][1] = 50*ms
# Delays[1][2] = 50*ms
Delays_samp = time2index(Delays)

buffer_len = int(max(np.max(Delays_samp[:])+1, min_buffer_len))

# history_buffer = DQRingBuffer(buffer_len = buffer_len, n_channels = N_total, initial_val=0)
history_buffer = RingBuffer_2D(buffer_len = buffer_len, n_channels = N_total, initial_val=0)







#%%

#linear gaussian (autoregressive) equations
eqs = '''
dv/dt = (v0 - v + I_in)/tau + sigma*xi*tau**-0.5 :1
v0 : 1
I_in : 1 # input current from other nodes
sigma : 1
'''

# simplest gap synapse
simple_gap_eq = '''
            w : 1 #weight
            I_in_post = w * (v_pre) : 1 (summed)
            '''
            
delayed_gap_eq = '''
            v_delayed: 1 #set in network_operation
            w : 1 #weight
            I_in_post = w * (v_delayed) : 1 (summed)
            '''
DO_NOTHING = 'v=v'
FAKE_THRESHOLD = 'v>999999'

#option 1: store delay inside synapse. (requires fake threshold, reset)
# all_groups = [NeuronGroup(N_neurons, eqs, method='euler', threshold=FAKE_THRESHOLD,reset='') for i in range(N_groups)]

#option 2: store delay in custom attribute
all_groups = [NeuronGroup(N_neurons, eqs, method='euler') for i in range(N_groups)]
for g in all_groups:
    g.sigma=base_sigma
# all_groups[2].sigma=0

#%%

circuit_str= adj_to_str(Weights, line_joiner =', ', node_name_f=lambda i: group_names[i])
print(circuit_str)
param_str = f'w={base_weight} sigma={base_sigma} tau={tau/ms:.1f}ms delay={base_delay/ms:.1f}ms'
param_str

#%%
group_names = [chr(i+97).upper() for i in range(N_groups)]

# N_nodes = len(all_groups)

def get_current_v():
    return np.concatenate( [G.v[:] for G in all_groups] ).T
        
#%%
all_synapses = []

# constrcut synapses based on weight matrix.
# - assumes all-to-all connectivity within group-group connections
for i in range(N_groups):
    for j in range(N_groups):
        if is_ij_valid(i,j):
            
            #option 1: store delay inside synapse. (requires fake threshold, reset)
            # ij_syn = Synapses(all_groups[i], all_groups[j], model=delayed_gap_eq, on_pre=DO_NOTHING, delay=Delays[i,j])
            #option 2: store delay in custom attribute
            ij_syn = Synapses(all_groups[i], all_groups[j], model=delayed_gap_eq)

            ij_syn.connect()
            ij_syn.w = Weights[i,j];
            
            ij_syn.add_attribute('delay_samp')
            ij_syn.delay_samp = Delays_samp[i,j]*np.ones(N_neurons*N_neurons).astype(int) #could this be made into a per-neuron list?
        
            ij_syn.add_attribute('group_i')
            # ij_syn.add_attribute('group_j')
            ij_syn.group_i = i
            # ij_syn.group_j = j
            
            all_synapses.append(ij_syn)


all_monitors = [StateMonitor(g,'v', record=True) for g in all_groups]

#%%
# following: https://brian.discourse.group/t/delay-for-summed-variables-in-synapses/424/2


    #%%

@network_operation
def record_v_to_buffer():
    global history_buffer
        
    history_buffer.append( get_current_v() )
    # --------------------------
    for a_syn in all_synapses:        
        #option 1: access delay from delay(time) variable,
        #    - then convert to samples
        # this_delay_samp = time2index(a_syn.delay) 
    
        #option 2: access delay from custom delay_samp attribute
        this_delay_samp = a_syn.delay_samp[:]
        buffer_from_idx = ij_to_flat_index(a_syn.group_i, a_syn.i[:])
        a_syn.v_delayed = history_buffer[buffer_from_idx, -this_delay_samp-1]
        
        # for debugging timing ONLY:
        # _ = history_buffer.to_np()
        # _ = history_buffer[buffer_from_idx, -this_delay_samp-1]
        # a_syn.v_delayed = np.zeros(buffer_from_idx.shape)
        # print(this_delay_samp)
        # a_syn.v_delayed = history_buffer.get_delayed(this_delay_samp,True)[buffer_from_idx]
        pass

history_buffer.to_np().shape
net = Network()
net.add(all_groups, all_synapses, all_monitors)
net.add(record_v_to_buffer)
# %%
t0 = time.time()
net.run(duration)
t1 = time.time()
run_walltime = t1-t0  
type(history_buffer)
print(f'{duration} second simulation took\n {run_walltime:.3f} seconds to simulate\n\
 with {type(history_buffer)},\n\
 buffer len: {buffer_len}, {N_total} neurons')
# %%


# #%% markdown
# 5x10 neurons, 500 buffer, DQ Buffer:
# 
# 1.5 sec without writing  
# -4 seconds writing 0 to v_delayed  
# 2-5 seconds with writting to v_delayed  (from Numpy buffer)
# 6 seconds with writting to v_delayed  (from DQ buffer, with get_delay then indexing by neuron)
# 27 seconds with writting to v_delayed  (from DQ buffer, with clunky custom 2D indexing)
# 
# 
# 
# reading out of history_buffer is the bottleneck !



#%%
#add simple offset for plotting
np_history_buffer = history_buffer.to_np().T
np_history_buffer.shape 

dfhist = expand_numpy_to_hier_df(np_history_buffer, group_names, neuron_names)
dfhist['time [ms]'] = all_monitors[0].t[-buffer_len:]*1000/second
# dfhist.tail(10)

#%%
df = volt_monitors_to_hier_df(all_monitors, group_names, neuron_names)
# df.tail(10)

#%%
print('any differences between buffer and monitor output?')
print(compare_df(df, dfhist, buffer_len))

#%%
history_group_names = ['history of '+n for n in group_names]

rename_groups = dict(zip(group_names, history_group_names))

df_m = melt_hier_df_timeseries(null_last_row(df))
dfhist_m = melt_hier_df_timeseries(null_last_row(dfhist))

df_m['compare population'] = df_m['population']
dfhist_m['compare population'] = dfhist_m['population'] 
dfhist_m.replace({'compare population': rename_groups}, inplace=True)

nudge_y = 0
dfhist_m['voltage'] = dfhist_m['voltage'] + nudge_y
#%%


#%%


fig = px.line(df_m, x='time [ms]', y='voltage', color='population')
fig.update_layout(width=500, height=300)
fig

#%%
'plots each channel as a row'
if N_neurons > 1:
    # fig = px.line(df_m,x='time [ms]',y='voltage',facet_row='flat_hier_idx',color='population')
    # fig.update_layout(width=500, height=80*N_nodes*N_neurons)
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))
    ' collapses into a row per population '
    fig = px.line(df_m, x='time [ms]', y='voltage', facet_row='population', color='neuron')
    fig.update_layout(width=500, height=400)
    fig.update_traces(line=dict(width=1))
    fig
#%% 


#%%
# figh = px.line(pd.concat([df_m, dfhist_m]), x='time [ms]', y='voltage', facet_row='flat_hier_idx',color='compare population',
#     title=f'last {buffer_len} samples of history saved into buffer')
# figh.update_layout(width=500, height=80*N_nodes*N_neurons)
# figh.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))
df_m
figh = px.line(pd.concat([df_m, dfhist_m ] ), x='time [ms]', y='voltage', facet_row='population',color='compare population',
    title=f'last {buffer_len} samples of history saved into buffer')
# figh.update_traces(marker=dict(size=1,opacity=.9))    
figh.update_layout(width=500, height=150*N_groups)

# figh.update_xaxes(range=[0,duration/second])

figh.update_traces(line=dict(width=1))
figh
#%%

# Construct cross-correlation matrix (as dataframe) from time-series data-frames
df.head(2)
# first average across neurons
df_avg = df.groupby(axis='columns', level=0).mean()
df_avg
#%%
figt = px.line(melt_group_df_timeseries(df_avg), x='time [ms]', y='voltage', 
    facet_row='population',color='population',labels={'population':'pop'})
figt.update_layout(width=600, height=500)
# figt.write_html('figs/gaussian_timeseries.html')
figt.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1], ))
figt.for_each_annotation(lambda a: a.update(x=-.08, textangle=-90) )

# figt.for_each_annotation(lambda a: print(a) )

figt.update_layout(showlegend=False)
figt




#%%
#%%


#%%
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
#%%
do_norm_outputs = True # seems like generally a good idea
do_sub_auto_corr = False 
# in some cases, NOT subtracting the auto-corr can clarify things
    # subtracted peak depends strongly on auto-corr width 
    # often subtracting the autocorr of the input carves a valley from the center of the xcorr, which is at least distracting 
do_norm_xcorr = False #seems 

df_avg_norm = df_avg.copy()

if do_norm_outputs:
    df_avg_norm = zscore_df_cols(df_avg_norm, group_names)

# create "closure" around our choice of whether to subtract autocorrelation
xcorr_df_func_norm = lambda df,i,j: xcorr_df_func(df, i, j, do_sub_auto_corr)
dfx = cross_function_df(df_avg_norm, group_names, xcorr_df_func_norm)

lag_key = 'lag [ms]'
nested_lag_key = ('lag [ms]','')


dfx.drop([nested_lag_key],axis=1)
if do_norm_xcorr:
    dfx = zscore_df_cols_except(dfx, [nested_lag_key])


time_range = np.array([-1,1])*250
dfx_m = melt_hier_df_timeseries(dfx,'from','to','xcorr',lag_key)
# dfx_m['to'] = dfx_m['to'].apply(lambda A: 'to '+A)

figx = px.line(time_selection_df(dfx_m, time_range, lag_key) , x=lag_key, y='xcorr',
    facet_row='from',color='to')
    
# figx.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]+'→'))
figx.for_each_annotation(lambda a: a.update(text='from '+a.text.split("=")[-1]))
# figx.for_each_trace(lambda a: a.update(text=a.text.split("=")[-1]+'→'))

figx.update_layout(width=350, height=500)
figx.layout.legend.x = 1.15

# figx.update_yaxes(range=[-5000,15000])
# figx.write_html('figs/gaussian_xcorr.html')
figx
#%%

fig = make_subplots(N_groups,2, column_widths=[0.8, 0.2],
    shared_xaxes=True,shared_yaxes='columns', 
    y_title='voltages',
    row_titles=[g+'→' for g in group_names], column_titles=['outputs','xcorr'])
fig.update_layout(width=850,height=500)


q_colors = px.colors.qualitative.Plotly   
def go_line(df,x,y,color,legendgroup):
    return go.Scatter(x=df[x],y=df[y],mode='lines',line = dict(color=color),legendgroup=legendgroup)

time_key = 'time [ms]'
df_avg
df_m = melt_group_df_timeseries(df_avg)

#see https://plotly.com/python/legend/#grouped-legend-items for linking legend toggling across groups


for i in range(N_groups):
    ip = i+1
    df_i = df_m[df_m['population']==group_names[i]]
    gl = go_line(df_i,x=time_key, y='voltage',color=q_colors[i],legendgroup=f'group{i}')
    gl.name='→'+group_names[i]
    if i==0:
        gl.legendgrouptitle.text="First Group Title"

    fig.add_trace(gl,row=ip,col=1)
    
    for j in range(N_groups):
        
        ij_mask = (dfx_m['from']==group_names[i]) & (dfx_m['to']==group_names[j])
        # time_mask = 
        df_ij = time_selection_df(dfx_m[ij_mask], time_range, lag_key)
        gl = go_line(df_ij,x=lag_key, y='xcorr',color=q_colors[j],legendgroup=f'group{j}')
        gl.showlegend=False
        gl.name=f'{group_names[i]}→{group_names[j]}'
        fig.add_trace(gl,row=ip,col=2)
        
fig.update_xaxes(title_text=time_key, row=ip, col=1)
fig.update_xaxes(title_text=lag_key, row=ip, col=2)

fig.update_layout(
    title=go.layout.Title(
        text = f'Cross-correlations from a gaussian network: {circuit_str} <br><sup>{param_str}</sup>',
        xref="paper",
        x=0
    ))
fig.write_html('figs/gaussian_combo.html')
fig

#%%      
#%%
# import dash_functions as my_dash
# my_dash.dash_app_from_figs([figt,figx], col_widths=None, title='Cross-correlations',subtitle='from a delayed gaussian network')

#%%
# '''
# -color by time (whether in or out of candidate window)
# -add annotation rectangle
# -stack multple weight values
# '''

# fig.write_html('figs/gaussian_xcorr.html')
#%%
# df_avg.mean()
# mi2 = pd.MultiIndex.from_product( [list(mi), list(mi)])




















#%%

