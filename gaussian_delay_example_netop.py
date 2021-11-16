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
- check my for-loop implementation from cross-correlation visualizer
- this is very close 
    - https://brian.discourse.group/t/delay-for-summed-variables-in-synapses/424/2?u=adam
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
so... no synapses used?
# %%
#%%

# 
# print(q.to_np())
# print(q.get_delayed(1))
# print(q[-1])
# print(m.get_delayed(1))

# print(r.get_delayed(2))


# %%
start_scope()
duration = 5000*ms
dt = defaultclock.dt;

N_neurons = 5
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

m_idx = pd.MultiIndex.from_product( [group_names, range(N_neurons)])
dfh = pd.DataFrame(columns=m_idx)

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
# fig = px.line(dfm,x='time [ms]',y='voltage',facet_row='total_neuron_idx',color='population')
fig = px.line(dfm,x='time [ms]',y='voltage',facet_row='population',color='neuron')
fig.update_traces(line=dict(width=1))
# fig.update_layout(width=500,height=80*N_nodes*N_neurons)
fig.update_layout(width=500,height=400)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))


#%% ARCHIVE
# class RingBuffer:
#     '''
#     loose implementation of a ring-buffer with numpy arrays
#     - consider something like collections.deque for a stricter implementation
#     - see also Implementing a Ring Buffer - Python Cookbook: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
# 
# 
#     use cycle_in() to insert values while maintaining the length of the buffer
#     '''
#     def __init__(self, buffer_len = 100, initial_val=None):
#         self.size = buffer_len
#         self.values = np.full(self.size, initial_val)
# 
#     def enqueue(self, new_val):
#         self.values = np.append(self.values, new_val)
# 
#     def dequeue(self):
#         self.values = self.values[1:]
# 
#     def cycle_in(self, new_val):
#         self.enqueue(new_val)
#         self.dequeue()
# 
#     def last(self):
#         return self.values[-1]
# 
#     def get_delayed(self, delay=1):
#         # if invalid delay delay is requested:
#         # return None or NaN 
# 
#         if not 0 < delay < self.size:
#             return None
#         # alternate implementation would
#         # cap the delay, so that delay = -Inf returns the oldest value in the array.
#         # delay = max(min(delay, self.size),1)
# 
#         return self.values[-delay]
#     def __str__(self):
#         return f'buffer with len: {self.size}, \nvals: {str(self.values)}\n'
