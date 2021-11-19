import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

import pandas as pd
import plotly.express as px

from brian_to_dataframe import *
from ring_buffers import *

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
do_norm_outputs = True # seems like generally a good idea
do_sub_auto_corr = False 
# in some cases, NOT subtracting the auto-corr can clarify things
    # subtracted peak depends strongly on auto-corr width 
    # often subtracting the autocorr of the input carves a valley from the center of the xcorr, which is at least distracting 
do_norm_xcorr = False



X = df_avg.to_numpy()
X.shape
# xstd=np.nanstd(X,axis=0)
# (X/xstd).shape

if do_norm_outputs:
    X = (X-np.nanmean(X,axis=0))/np.nanstd(X,axis=0)

# assumes time is last column, timeseries is contiguous
# XCORRS = {}

' initialize xcorr dataframe from product of group names '
dfx = pd.DataFrame(columns=pd.MultiIndex.from_product( [group_names, group_names]).set_names(['from','to']))
dfx.head()

def corr_col(i,j):
    return np.correlate(X[:-1,i], X[:-1,j],'same')
    
for i in range(X.shape[1]-1):
    for j in range(X.shape[1]-1):
        this_xcorr = corr_col(i,j)  
        if do_sub_auto_corr or i==j:
            norm_i = i
            this_xcorr -= corr_col(norm_i,norm_i)
        this_xcorr /= len(this_xcorr)
            
        # xcorr_df(df[i], df[j]
        dfx[(group_names[i],group_names[j])] = this_xcorr
        
#NOTE: this should be specified in terms of df_avg.time!        
dfx['lag [ms]'] = (dfx.index - len(dfx.index)/2)*dt/ms


if do_norm_xcorr:
    norm_dfx=(dfx-dfx.mean())/dfx.std()
    # norm_dfx=dfx/dfx.std()
    # norm_dfx=(dfx-dfx.mean())
else:
    norm_dfx = dfx
    

time_range = np.array([-1,1])*200

dfx_m = melt_hier_df_timeseries(norm_dfx,'from','to','xcorr','lag [ms]')
dfx_m['to'] = dfx_m['to'].apply(lambda A: 'to '+A)

figx = px.line(time_selection_df(dfx_m,time_range, 'lag [ms]') , x='lag [ms]',y='xcorr',
    facet_row='from',color='to')
    
# figx.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]+'→'))
figx.for_each_annotation(lambda a: a.update(text='from '+a.text.split("=")[-1]))
# figx.for_each_trace(lambda a: a.update(text=a.text.split("=")[-1]+'→'))

figx.update_layout(width=350,height=500)
figx.layout.legend.x = 1.15

# figx.update_yaxes(range=[-5000,15000])
# figx.write_html('figs/gaussian_xcorr.html')
figx
#%%
      
#%%
import dash_functions as my_dash
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

