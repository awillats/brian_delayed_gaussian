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

# %% markdown
## meta-tasks

----------
## within a loop
- store current value
    - with net_op append current val to short buffer of past values to X_past 
    
- grab from past -> (write to history buffer) -> write to stim-current
    - with net_op apply stimulation according to X_past
    - how to store the past?
    - Do I need a queue implementation?
    - TimedArray would allow flexible relative time [discussion](https://github.com/brian-team/brian2/issues/467)
- access stim-current for effect 
    
    - handled automatically by `w * I_past_stim` in voltage equations
----------

how to handle multiple delays?
- tensor? n_neurons x n_neurons x n_delay
    - with weight as entry
    - collapse result?

----------
# %%
# should be computed from the max delay_samples
buffer_len = 500

# %%
start_scope()
duration = 1000*ms
dt = defaultclock.dt;

N_groups = 2 # gets re-written later
N_neurons = 2
N_total = N_groups*N_neurons
neuron_names = range(N_neurons)
tau = 3*10*ms
sigma = 50#50
weight = 1 / N_neurons;

delay = tau*.4
delay_samp = int(round(delay / dt))

buffer_len = max(delay_samp+1, buffer_len)

history_buffer = DQRingBuffer(buffer_len = buffer_len, n_channels = N_total, initial_val=0)
# history_buffer = RingBuffer_2D(buffer_len = buffer_len, n_channels = N_total, initial_val=0)

#%%

#linear gaussian (autoregressive) equations
eqs = '''
dv/dt = (v0 - v + I_in)/tau + sigma*xi*tau**-0.5 :1
v0 : 1
I_in : 1 # input current from other nodes
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
Ga = NeuronGroup(N_neurons, eqs, method='euler')
Gb = NeuronGroup(N_neurons, eqs, method='euler')
# Ga.sigma = sigma
# Gb.sigma=sigma/2
#%%
all_groups = [Ga,Gb]
group_names = ['A','B']
N_nodes = len(all_groups)

def get_current_v():
    return np.concatenate( [G.v[:] for G in all_groups] ).T
    # return Ga.v[:]
        
# get_current_v()
#%%
all_synapses = []

ab_syn = Synapses(Ga, Gb, model=delayed_gap_eq)
ab_syn.connect()
ab_syn.w = weight;

all_synapses.append(ab_syn)
all_monitors = [StateMonitor(g,'v', record=True) for g in all_groups]

#%%
# following: https://brian.discourse.group/t/delay-for-summed-variables-in-synapses/424/2
@network_operation
def record_v_to_buffer():
    global history_buffer
    
    history_buffer.append( get_current_v() )
    # --------------------------
    # one-liner, only works for buffers which store values as 2D numpy:
    ab_syn.v_delayed = history_buffer[ab_syn.i[:], -delay_samp-1]
        
    # explicit, multi-line version:
    # selected_delayed_vals = history_buffer.get_delayed(delay_samp,True)
    # mapped_delayed_vals = selected_delayed_vals[ab_syn.i[:]]
    # ab_syn.v_delayed = mapped_delayed_vals
    
    # one-liner, works for either buffer type
    # ab_syn.v_delayed = history_buffer.to_np()[ab_syn.i[:], -delay_samp-1]
    # ab_syn.v_delayed = np.array(history_buffer.get_delayed(delay_samp))[ab_syn.i[:]]


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


#%% markdown
I think all these runtimes are for access only,
1.0 second simulation took
 0.560, 0.642  seconds to simulate
 with <class 'ring_buffers.DQRingBuffer'>,
 buffer len: 500

1.0 second simulation took
 0.873, .817 seconds to simulate
 with <class 'ring_buffers.RingBuffer_2D'>,
 buffer len: 500
 
----------
 5.0 second simulation took
  2.791, 2.845 seconds to simulate
  with <class 'ring_buffers.DQRingBuffer'>,
  buffer len: 500
  
 5.0 second simulation took
  3.711, 4.006 seconds to simulate
  with <class 'ring_buffers.RingBuffer_2D'>,
  buffer len: 500
 ----------



#%%
#add simple offset for plotting
np_history_buffer = history_buffer.to_np().T
np_history_buffer.shape 

dfhist = expand_numpy_to_hier_df(np_history_buffer, group_names, neuron_names)
dfhist['time [ms]'] = all_monitors[0].t[-buffer_len:]*1000/second
# dfhist.tail(10)

#%%
dfh = volt_monitors_to_hier_df(all_monitors, group_names, neuron_names)
# dfh.tail(10)

#%%
dfhist_tail = dfhist.tail(buffer_len).reset_index(drop=True)
df_tail = dfh.tail(buffer_len).reset_index(drop=True)

print('any differences between buffer and monitor output?')
print(df_tail.compare(dfhist_tail))
#%%
history_group_names = ['buffer A','buffer B']
rename_groups = dict(zip(group_names, history_group_names))

df_m = melt_hier_df_timeseries(null_last_row(dfh))
dfhist_m = melt_hier_df_timeseries(null_last_row(dfhist))

df_m['compare population'] = df_m['population']
dfhist_m['compare population'] = dfhist_m['population'] 
dfhist_m.replace({'compare population': rename_groups}, inplace=True)

nudge_y = 0
dfhist_m['voltage'] = dfhist_m['voltage'] + nudge_y
#%%

if N_total > 50:
    print('DANGER, plots are going to take a long time')

#%%


fig = px.line(df_m, x='time [ms]', y='voltage', color='population')
fig.update_layout(width=500, height=200)
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

figh = px.line(pd.concat([df_m, dfhist_m ] ), x='time [ms]', y='voltage', facet_row='population',color='compare population',
    title=f'last {buffer_len} samples of history saved into buffer')
# figh.update_traces(marker=dict(size=1,opacity=.9))    
figh.update_layout(width=500, height=400)

# figh.update_xaxes(range=[0,duration/second])

figh.update_traces(line=dict(width=1))
figh
