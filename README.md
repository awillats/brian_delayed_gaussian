# Goals


- Looking to simulate a circuit with the Brian2 simulator framework which represents a rate model of a network with delay in synapses
  - See ["Delay in continuous connections"](https://brian.discourse.group/t/delay-in-continuous-connections/509) for more context
  - incrementally progressing from simplest solutions, to more efficient ones

- **secondary goal:** export data from StateMonitors to pandas DataFrames
  - format suitable for df-based plotting libraries like plotly

- ~~**tertiary goal:** deploy interactive cross-correlation figures as a Dash app (through Heroku + GitHub)~~
  - after trying this, I would prefer a static html export
---
# Adam's current development priorities 
- [ ] wrap up cross-correlation & plotting
  - [ ] clean up labels on cross-correlation dataframe
    
    - nodes x time to
    - from node, to node, lag [ms]
    
    - can I normalize without casting to numpy?
      - why cast to numpy?
        - for use in functions 
        - to freeze time index sorting
        - to use :-1 slicing 
      - pass columns instead OR use pandas functions
        - this operation works well enough: https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe/28577480
    
    
  - [ ] robustify normalization code
    - ?? hoist options to the top of the script
    - [ ] encapsulate `timeseries_df_to_xcorr`
      - [ ]  bundle for loop into function 
     
    - [ ] encapsulate encoding 2 columns down to one
      - look into plotly combining columns
    
  - [ ] translate dash app to combined figures for static export
  - [ ] debug plotly xrange issue
  - [ ] add nicer cross-correlation annotation

- [~] partition plotting-related and cross-correlation code from delay-buffer-related code 
  - [ ] update script directory
  - [ ] perhaps have separate branches?

- [ ] debug multiple input synapses to same current

- [ ] cleanup setting values for sigma
  - "sigma" is an internal variable of group "neurongroup_1", but also exists in the run namespace with the value 1. The internal variable will be used.
  
- [ ] clean up indexing convention

- [ ] implement, demonstrate "rolling index" 

---

  
# Components of a solution
1. Extract current state of network (ideally as a column) 
1. store current value in history buffer
    - see `ring_buffers`
2. translate buffer to brian-accessible value
    - using [`network_operation` [docs]](https://brian2.readthedocs.io/en/stable/reference/brian2.core.operations.network_operation.html) or [`run_regularly` [docs]](https://brian2.readthedocs.io/en/stable/reference/brian2.core.operations.network_operation.html)
3. access & apply influence of delayed variable


  
---
## Advanced features 
- [x] generalize beyond two populations 
  - [?] multiple inputs to each output population 
    - sum effect in `I_from_delayed`?
      ```
        for i, this_delay in enumerate(delayed_connections):
          I_from_delayed += w[i] * history_buffer.get_delayed(this_delay)
      ```
      - handled for us with `(summed)` keyword and equations being stored in synapse?
      
  - [x] !!! different delays for each synapse 
    - [x] start by implementing different weights for each synapse 
      - see my implementation of a weight matrix + delay matrix (single delay per connection)
      
    - [!] store per-synapse delay as synapse variable 
    - [x] access per-synapse delay in network_op
      - may require `for .. in` to loop across 
      - for now looping across synapses is enough
    - does this require hard-coded connections?
      - i'm currently using `v_delayed` as container for presynaptic effect 
  
- [ ] multiple delays per synapse 
  - is this taken care of us by multi-synapses?
    - see [creating multi-synapses](https://brian2.readthedocs.io/en/stable/user/synapses.html#creating-multi-synapses)
      - > This is useful for example if one wants to have multiple synapses with different delays. To distinguish multiple variables connecting the same pair of neurons in synaptic expressions and statements,
  - handle as a N x N x N_delay tensor of weights?
    - could lead to simple, efficient linear algebra for updates
    - could be stored sparsely
    - connections would typically depend on a sparse selection of delays
    - could be implemented as nested list / matrix of lists 
## Speed & efficiency:    
### Faster indexing / appending
  - [ ] instead of rolling buffer, move the index
    - see cylindrical array in "Vectorized Algos"
    - see also implementation by RTH: https://brian.discourse.group/t/delay-in-continuous-connections/509/6

  - [ ] **Indexing conventions (for cylindrical history buffer)**

    - [neuron x time] 
      - Marcel and RTH's solutions both use this
      - allows current state to be inserted as a column 
      - also allows memory-efficient access of loading a single time-slice (if the number of time-samples stored is less than the number of groups)
        - see [row and column order - wiki](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
      - consistent with plotting time on x-axis 
      
    - [time x neuron] index
      - comes from Morrison et al. 2007 via "Vectorized Algos." paper
      - consistent with dataframe view 
      - `np.concatenate([ G.v[:] for G in all_groups ])` returns a row-vector


  - [ ] C++ implementation
    - [see current SpikeQueue implementation](https://github.com/brian-team/brian2/blob/master/brian2/synapses/cspikequeue.cpp)
    - needs cylindrical array implementation 
      - std::deque is available
    - needs access to brian's internal arrays 
      - [example from github issue](https://github.com/brian-team/brian2genn/issues/123#issuecomment-720425213)   
        - note this is for Brian2GeNN
      - https://brian.discourse.group/t/user-defined-functions/271
### Storage
  - tension between sparse list-based solution and vectorized indexing / insertion
    - [Storing a Sparse Matrix - wiki page](https://en.wikipedia.org/w/index.php?title=Sparse_matrix#Storing_a_sparse_matrix)
    - something like matlab's sparse matrix representation could bridge the gap
      - scipy has a 
      - so does PyTorch: [torch.sparse](https://pytorch.org/docs/stable/sparse.html)
      - discussion and benchmarking of [sparse vs dense matrices on stack overflow](https://stackoverflow.com/questions/36969886/using-a-sparse-matrix-versus-numpy-array)
    - "Vectorized Algos" discusses:
      - dense matrix storage 
      - list of lists 
      - compressed spare row


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
    
- [ ] friendly syntax for expressing delayed relationships 
  - ideally, consistent with [delay for spiking synapses](https://brian2.readthedocs.io/en/stable/user/synapses.html#delays)
    - `synapses = Synapses(sources, targets, '...', on_pre='...', delay=1*ms)`
    - would be especially convenient to be able to switch between spiking and rate-based implementations with minimal model-specification code differences
    
    - can do this for now, pull property back out inside network operations 
      - but this requires converting from time to samples every step 
    - can also `add_attribute('delay_samples')` 
    - see `example_netop_multi_population` for implementations of this
    
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

- `demo_netop_stim.py`
  - uses `network_operation` to:
    - store values in ring buffer 
    - map history buffer to corresponding inputs 
    - apply synaptic effect from delayed voltages

- `demo_netop_multi_population.py`
  - generalized delayed stimulation to more than two groups
  - exports voltage monitors to dataframes for plotting
  
- 🚧 `plot_xcorr_multi_population.py` 🚧
  - largely mimics the `multi_population` script
  - adds interactive plotly plots of cross-correlations 
    - very helpful to verify delayed interactions are implemented as intended 
    - but also this means a lot of extraneous code
  - Adam currently using it for some ad-hoc parameter sweeps


<details><summary> archived scripts </summary>
  
  - `gaussian_delay_example_record_history.py`
    - just demonstrates recording to buffer
    - uses `network_operation` to store values in global ring buffer
</details>

<details><summary> possible future scripts </summary>
  
- `gaussian_delay_example_runreg.py`
  - try run_regularly rather than network_operation
  
- `gaussian_delay_example_cpp.py`
  - after testing storage structures in python, convert to c++ standalone mode
  
</details>


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
    
    
    
    