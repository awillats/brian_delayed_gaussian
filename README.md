# Goals


- Looking to simulate a circuit with the Brian2 simulator framework which represents a rate model of a network with delay in synapses
  - See ["Delay in continuous connections"](https://brian.discourse.group/t/delay-in-continuous-connections/509) for more context

- **secondary goal:** export data from StateMonitors to pandas DataFrames
  - format suitable for df-based plotting libraries like plotly
  
# Components of a solution
1. Extract current state of network (ideally as a column) 
1. store current value in history buffer
    - see `ring_buffers`
2. translate buffer to brian-accessible value
    - using [`network_operation` [docs]](https://brian2.readthedocs.io/en/stable/reference/brian2.core.operations.network_operation.html) or [`run_regularly` [docs]](https://brian2.readthedocs.io/en/stable/reference/brian2.core.operations.network_operation.html)
3. access & apply influence of delayed variable

## Indexing conventions (for cylindrical history buffer)
- [time x neuron] index
  - comes from Morrison et al. 2007 via Vectorized Algos.
  - consistent with dataframe view 
  - `np.concatenate([ G.v[:] for G in all_groups ])` returns a row-vector
- [neuron x time] 
  - marcel and RTH's solutions both use this
  - allows current state to be inserted as a column 
  - consistent with plotting time on x-axis 
---
## Advanced features 
- [ ] generalize beyond two populations 
  - [?] multiple inputs to each output population 
    - sum effect in `I_from_delayed`?
      ```
        for i, this_delay in enumerate(delayed_connections):
          I_from_delayed += w[i] * history_buffer.get_delayed(this_delay)
      ```
      - handled for us with `(summed)` keyword and equations being stored in synapse?
      
  - [ ] !!! different delays for each synapse 
    - [ ] start by implementing different weights for each synapse 
      - see my implementation of a weight matrix + delay matrix (single delay per connection)
      
    - [ ] store per-synapse delay as synapse variable 
    - [ ] access per-synapse delay in network_op
    - may require `for .. in` to loop across 
    - does this require hard-coded connections?
      - i'm currently using `v_delayed` as container for presynaptic effect 
         - does this assume each 
  
- [ ] multiple delays per synapse 
  - is this taken care of us by multi-synapses?
    - see [creating multi-synapses](https://brian2.readthedocs.io/en/stable/user/synapses.html#creating-multi-synapses)
      - > This is useful for example if one wants to have multiple synapses with different delays. To distinguish multiple variables connecting the same pair of neurons in synaptic expressions and statements,
  - handle as a N x N x N_delay tensor of weights?
    - could lead to simple, efficient linear algebra for updates
    - could be stored sparsely
    - connections would typically depend on a sparse selection of delays
    - could be implemented as nested list / matrix of lists 
    
- [ ] faster solutions 
  - [ ] instead of rolling buffer, move the index
    - see cylindrical array in Vectorized Algos
    - see also implementation by RTH: https://brian.discourse.group/t/delay-in-continuous-connections/509/6

  - [ ] C++ implementation
    - [see current SpikeQueue implementation](https://github.com/brian-team/brian2/blob/master/brian2/synapses/cspikequeue.cpp)
    - needs cylindrical array implementation 
      - std::deque is available
    - needs access to brian's internal arrays 
      - [example from github issue](https://github.com/brian-team/brian2genn/issues/123#issuecomment-720425213)   
        - note this is for Brian2GeNN
      - https://brian.discourse.group/t/user-defined-functions/271
  - conflict between sparse list-based solution and vectorized indexing / insertion
    - something like matlab's sparse matrix representation could bridge the gap

- [ ] only store history from variables which influence downstream targets
  - currently storing history of entire network
  - alternate list + loop based approach
    ```python
    def get_current_state( delay_sources):
      current_state = []
      for a_src in delay_sources:
        current_state.append( get_current_val(a_src,'v') )
      return current_state 
    ```
    - store "keys" for sources somewhere?
    
- [ ] friendly synatax for expressing delayed relationships 
  - ideally, consistent with [delay for spiking synapses](https://brian2.readthedocs.io/en/stable/user/synapses.html#delays)
    - `synapses = Synapses(sources, targets, '...', on_pre='...', delay=1*ms)`
    - would be especially convenient to be able to switch between spiking and rate-based implementations with minimal model-specification code differences
    
    - can do this for now, pull property back out inside network operations 
      - but this requires converting from time to samples every step 
    - can also `add_attribute('delay_samples')` 
    
  - or consistent with delay differential equation expression
    - `eqs = 'dr/dt = w * r(t - delay_t) '`
    - [Adding delay differential equation solver](https://brian.discourse.group/t/adding-delay-differential-equation-solver/191)
- [ ] flexible time access - e.g. `TimedArray()`
  - see [github discussion](https://github.com/brian-team/brian2/issues/467#issuecomment-119234615)

---
# Directory of files
## Scripts
- `gaussian_example_base.py`
  - constructs a 2-node network *without* synaptic delay of any kind
  - intended as a sanity check and point of comparison to the delayed version
  
- `gaussian_delay_example_record_history.py`
  - uses `network_operation` to store values in global ring buffer

- `gaussian_delay_example_netop_stim.py`
  - uses `network_operation` to:
    - store values in ring buffer 
    - map history buffer to corresponding inputs 
    - apply synaptic effect from delayed voltages
  
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
  - has "if __name__ == __main__" to test buffers
- `brian_to_dataframe`
  - `null_last_row()`
    - niche utility function 
    - adds a row of None values so that melted dataframes plot nicely with Plotly
  - `volt_monitors_to_hier_df()`
    - creates multi-index dataframe via `hier_df_from_lists()`
    - calls `expand_volt_monitor_to_hier_df()`
    - appends time column
  - `expand_volt_monitor_to_df_columns()`
    - makes non-nested dataframe, embedding hierarchy in column names instead

    