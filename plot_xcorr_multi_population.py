import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_functions import *

import plotly.express as px
import plotly.colors

from brian_to_dataframe import *
from ring_buffers import *
from circuit_helpers import *
from dataframe_preprocessing_functions import *


%matplotlib inline
%load_ext autoreload
%autoreload 2

'''
To-do:
ERROR: NotImplementedError: Multiple "summed variables" target the variable "I_in" in group "neurongroup_2". Use multiple variables in the target group instead.

'''

# %%
start_scope()
duration = 2000*ms
dt = defaultclock.dt;
def time2index(t):
    return np.round(t/dt).astype(int)

    
min_buffer_len = time2index(0*ms)

N_groups = 5
N_neurons = 10
N_total = N_groups*N_neurons
neuron_names = range(N_neurons)
tau = 10*ms
base_sigma = 1

def ij_to_flat_index(high_idx, low_idx, N_high=N_groups, N_low=N_neurons):
    return high_idx*N_low + low_idx
    
def flat_index_to_ij(fi, N_high=N_groups, N_low=N_neurons):
    low_idx = fi%N_low
    high_idx = floor(fi/N_low)
    return (high_idx, low_idx)

#%%
base_weight = 2 / N_neurons; #0.001 = null, 1=medium-strong, 3=strong, 0.9 for reciprocal
base_delay = 20*ms #tau * 0.5
base_delay_samp = time2index(base_delay)


'''
This section specifies connectivity
'''
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
# Annotate parameters
group_names = [chr(i+97).upper() for i in range(N_groups)]

peak_window = [-(base_delay+tau)*3/ms, 0]

circuit_str= adj_to_str(Weights, line_joiner =', ', node_name_f=lambda i: group_names[i])
param_str = f'w={base_weight} sigma={base_sigma} tau={tau/ms:.1f}ms delay={base_delay/ms:.1f}ms {N_neurons} neurons per pop.'

def gaussian_title(circuit_str, param_str):
    return f'Cross-correlations from a gaussian network: {circuit_str} <br><sup>{param_str}</sup>'

fig_title = gaussian_title(circuit_str, param_str)
nested_colors = gen_nested_color_scheme(N_groups, N_neurons)
#%%

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
# %% markdown
## Simulation notes:

#%% code
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

# add extra annotation to color by nested variables
df_m['compare population'] = df_m['population']
dfhist_m['compare population'] = dfhist_m['population'] 
dfhist_m.replace({'compare population': rename_groups}, inplace=True)

# nudge_y = 0
# dfhist_m['voltage'] = dfhist_m['voltage'] + nudge_y

#%%

#%%
fig = []
fig = px.line(df_m, x='time [ms]', y='voltage', facet_row='population', 
    color='flat_hier_idx', color_discrete_sequence=nested_colors)
fig.update_layout(width=800, height=500)
fig.update_traces(line=dict(width=1))
fig.update_layout(showlegend=False)
# fig.write_image("figs/nested_color.png",scale=2) #hangs?
# scope._shutdown_kaleido()
# see issue: https://github.com/plotly/Kaleido/issues/42
# fig.write_image("figs/nested_color_.svg",scale=2)
fig 
#%%
# Construct cross-correlation matrix (as dataframe) from time-series data-frames
# first average across neurons
df_avg = df.groupby(axis='columns', level=0).mean()
df_avg

 
#Options for cross-correlation analysis
do_norm_outputs = True # seems like generally a good idea
do_sub_auto_corr = False 
# in some cases, NOT subtracting the auto-corr can clarify things
    # subtracted peak depends strongly on auto-corr width 
    # often subtracting the autocorr of the input carves a valley from the center of the xcorr, which is at least distracting 
do_norm_xcorr = False #seems 

dfx = extract_xcorr_from_df(df_avg, do_norm_outputs, do_sub_auto_corr, do_norm_xcorr)

#%%

figtx = df_plot_timeseries_and_xcorr(df_avg, dfx, group_names, xcorr_plot_window=[-250,250],highlight_window=peak_window,
    fig_title=fig_title, html_file=None)
    # 'figs/gaussian_combo.html')
figtx


















#%%

