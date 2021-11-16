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


 - [ ] just record data to buffer with network_operation 

----------
## within a loop
- store current value
    - with net_op append current val to short buffer of past values to X_past 
    
- grab from past -> (write to history buffer) -> write to stim-current
    ```python
    buffer = ring_buffers.RingBuffer(buffer_len = max_delays, n_channels=len(delay_recorded_group) )
    
    @network_operation()
    def propagate_delayed_x():
        global buffer
        buffer.append( delay_recorded_group.x[:] )
        synapses.x_delayed = buffer[synapses.i[:], synapses.delay_in_steps[:]-1]
        #or
        synapses.x_delayed = buffer.get_delayed(synapses.delay_in_steps[:])[synapses.i[:]]
    ```
    - with net_op apply stimulation according to X_past
    - how to store the past?
    - Do I need a queue implementation?
    - TimedArray would allow flexible relative time [discussion](https://github.com/brian-team/brian2/issues/467)
- access stim-current for effect 
    
    - handled automatically by `w * I_past_stim` in voltage equations
----------
### Useful resources

- syn.i gets presynaptic index, syn.j post-
----------
# %%
buffer_len = 500
# %%
start_scope()
duration = 5000*ms
dt = defaultclock.dt;

N_groups = 2 # gets re-written later
N_neurons = 5
N_total = N_groups*N_neurons
neuron_names = range(N_neurons)
tau = 10*ms
sigma = 50#50
weight = -1;

delay = 10*ms
delay_samp = int(round(delay / dt))

history_buffer = DQRingBuffer(buffer_len = buffer_len, n_channels = N_total)
# history_buffer = RingBuffer_2D(buffer_len = buffer_len, n_channels = N_total)
history_buffer.fill(0*np.ones(N_total))

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

@network_operation
def record_v_to_buffer():
    global history_buffer
    history_buffer.append( get_current_v() )

Ga = NeuronGroup(N_neurons, eqs, method='euler')
Gb = NeuronGroup(N_neurons, eqs, method='euler')
#%%
def get_current_v():
    return np.concatenate( [Ga.v[:],Gb.v[:]] )
    
get_current_v()
#%%
all_groups = [Ga,Gb]
group_names = ['A','B']
N_nodes = len(all_groups)

all_synapses = []

ab_syn = Synapses(Ga, Gb, model=simple_gap_eq)
ab_syn.connect()
ab_syn.w = weight;

all_synapses.append(ab_syn)
all_monitors = [StateMonitor(g,'v', record=True) for g in all_groups]

#%%

net = Network()
net.add(all_groups, all_synapses, all_monitors)
net.add(record_v_to_buffer)


# %%
t0 = time.time()
net.run(duration)
t1 = time.time()
run_walltime = t1-t0  
type(history_buffer)
print(f'{duration} second simulation took\n {run_walltime:.3f} seconds to simulate\n with {type(history_buffer)},\n buffer len: {buffer_len}')
#%% markdown

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
np_history_buffer = history_buffer.to_np()
np_history_buffer.shape

dfhist = expand_numpy_to_hier_df(np_history_buffer, group_names, neuron_names)
dfhist['time [ms]'] = all_monitors[0].t[-buffer_len:]/second
dfhist.tail(10)

#%%
dfh = volt_monitors_to_hier_df(all_monitors, group_names, neuron_names)
dfh.tail(10)
#%%
dfhist_tail = dfhist.tail(buffer_len).reset_index(drop=True)
df_tail = dfh.tail(buffer_len).reset_index(drop=True)

print('any differences between buffer and monitor output?')
print(df_tail.compare(dfhist_tail))
#%%
history_group_names = ['hA','hB']
rename_groups = dict(zip(group_names, history_group_names))

df_m = melt_hier_df_voltage(dfh)
dfhist_m = melt_hier_df_voltage(dfhist)
# df_m[df_m['time [ms]'] >= 0.05]
df_m['compare population'] = df_m['population']
dfhist_m['compare population'] = dfhist_m['population'] 
dfhist_m.replace({'compare population': rename_groups}, inplace=True)


#%%
'plots each channel as a row'
fig = px.line(df_m,x='time [ms]',y='voltage',facet_row='total_neuron_idx',color='population')
fig.update_layout(width=500, height=80*N_nodes*N_neurons)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))

' collapses into a row per population '
# fig = px.line(df_m, x='time [ms]', y='voltage', facet_row='population', color='neuron')
# fig.update_layout(width=500, height=400)
# fig.update_traces(line=dict(width=1))
fig
#%% 


#%%
figh = px.line(pd.concat([df_m, dfhist_m]), x='time [ms]', y='voltage', facet_row='total_neuron_idx',color='compare population',
    title=f'last {buffer_len} samples of history saved into buffer')
figh.update_xaxes(range=[0,duration/second])
figh.update_layout(width=500, height=80*N_nodes*N_neurons)
figh.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))

# figh.update_traces(line=dict(width=1))

