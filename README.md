# Goals

See ["Delay in continuous connections"](https://brian.discourse.group/t/delay-in-continuous-connections/509) for more context

Looking to simulate a circuit with the Brian2 simulator framework which represents a rate model of a network with delay in synapses


# Components of a solution
1. store current value in history buffer
  - see `ring_buffers`
2. translate buffer to brian-accessible value
  - using [`network_operation` [docs]](https://brian2.readthedocs.io/en/stable/reference/brian2.core.operations.network_operation.html) or [`run_regularly` [docs]](https://brian2.readthedocs.io/en/stable/reference/brian2.core.operations.network_operation.html)
3. access & apply influence of delayed variable

---
# Directory of files
## Scripts
- `gaussian_example_base.py`
  - constructs a 2-node network *without* synaptic delay of any kind
  - intended as a sanity check and point of comparison to the delayed version
  
- `gaussian_delay_example_netop.py`
  - uses `network_operation` to store values
  
- ~~`gaussian_delay_example_runreg.py`~~
- ~~`gaussian_delay_example_cpp.py`~~

## Helper functions 
- `ring_buffers`
  - `class DQRingBuffer`
    - uses collections.deque 
    - can be made 2D by passing columns in (get converted to tuples)
  - `class RingBuffer` 
    - 1D, using numpy
  - `class RingBuffer_2D`
    - 2D, using numpy
- `brian_to_dataframe`
  - `hier_df_from_lists()`
  - `expand_volt_monitor_to_df_columns()`
  - `expand_volt_monitor_to_hier_df()`