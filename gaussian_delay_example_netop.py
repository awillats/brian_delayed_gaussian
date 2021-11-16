import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

import pandas as pd
import plotly.express as px

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
# %%
#%%

# %%
start_scope()
duration = 5000*ms
dt = defaultclock.dt;

N_neurons = 5
neuron_names = range(N_neurons)
tau = 10*ms
sigma = 50#50
weight = -1;

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

Ga = NeuronGroup(N_neurons, eqs, method='euler')
Gb = NeuronGroup(N_neurons, eqs, method='euler')

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



# %%
net.run(duration)
#%%

#%%
# This code gets the final states, but doesn't unpack the entire timeseries
# df= Ga.get_states(units=False,format='pandas')
# df.head()
#%%
dfh = hier_df_from_lists(group_names, neuron_names)
for idx,mon in enumerate(all_monitors):
    data_dict = mon.get_states(['t','v'],units=False)
    dfh = expand_volt_monitor_to_hier_df(data_dict, group_name=group_names[idx], df=dfh)

dfh['time [ms]'] = data_dict['t']
dfh.head(10)
#%%
# Collapses dataframe hierarchcial index so we can plot the dataframe 
# using hierarchical ID for faceting 
dfm = dfh.melt(id_vars='time [ms]',var_name=['population','neuron'],value_name='voltage')
# Create a combined id column to facet by
dfm['total_neuron_idx'] = dfm['population']+ dfm['neuron'].astype(str)
# dfm.head()
#%%
'plots each channel as a row'
fig = px.line(dfm,x='time [ms]',y='voltage',facet_row='total_neuron_idx',color='population')
fig.update_layout(width=500, height=80*N_nodes*N_neurons)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))

' collapses into a row per population '
# fig = px.line(dfm, x='time [ms]', y='voltage', facet_row='population', color='neuron')
# fig.update_layout(width=500, height=400)

fig.update_traces(line=dict(width=1))


#%%
