from collections import deque 
import numpy as np

import itertools
#%%
# This code gets the final states, but doesn't unpack the entire timeseries
# df= Ga.get_states(units=False,format='pandas')
# df.head()


'''
beware, if self.values ends up as a ([...],dtype=object) array, 
the pointers will lead to unintended consequences later
'''
#%%
class DQRingBuffer(deque):
    '''
    Assumes last value in buffer is current! i.e. 0-delay
    - this is compatible with calling
        buff.append(current_val)
        current_val_also = buff.get_delayed(0) 
    - see more discussion at https://stackoverflow.com/questions/4151320/efficient-circular-buffer
    - becuase of storage of columns as tuples, indexing is less flexible than numpy version
    '''
    from collections import deque
    def __init__(self, buffer_len = 100, n_channels=1, initial_val=None):
        super().__init__([],maxlen = buffer_len)
        self.n_channels = n_channels
        if initial_val is not None:
            self.fill(np.ones(n_channels)*initial_val)
    #inherits append method
    def append(self, new_val):
        if self.n_channels > 1 and len(new_val) is not self.n_channels:
            raise IndexError(f'ERROR, wrong size to append\n expected size {self.n_channels} got {len(new_val)}')
            return None
        if isinstance(new_val, np.ndarray):
            new_val = tuple(new_val)
        return super().append(new_val)
    
    def fill(self, fill_val):
        for i in range(self.maxlen):
            self.append(fill_val)
    def to_np(self):
        return np.asarray(self).T
    def get_delayed(self, delay=1, to_np=False):
        if not 0 <= delay < self.maxlen:
            # return 0
            # won't let you return get_delayed(0),
            raise IndexError(f'delay {delay} out of bounds for buffer length {self.maxlen}')
        # print(-delay-1)
        if to_np:
            return np.array(self[-delay-1])
        else:
            return self[-delay-1]
    
    def __getitem__(self, idx, jdx=None):
        '''
        wrapper to extend indexing to deque to be multidimensional
        - ideally this indexing mirrors numpy's
            - can't handle advanced numpy indexing flexibly... should just use numpy
        - slicing deques in python: https://discuss.python.org/t/slicing-notation-with-deque-and-multi-dimensional-slicing-on-data-structures/2791
        
        '''
        if isinstance(idx, slice):
            # 1D slicing
            # print(idx.start)
            return list(itertools.islice(self, idx.start, idx.stop, idx.step))
        elif isinstance(idx, tuple):
            # if it's more complicated than that, need to conver to numpy
            # convert to numpy then index
            v = self.to_np()
            return v[idx[0], idx[1]]
        else:
            # print(type(idx))
            return super().__getitem__(idx)
        
        # if jdx is not None:
        #     print(jdx)
        # return super().__getitem__(idx)
        # if isinstance(jdx, int):
        #     return super().__getitem__(idx)
        # elif isinstance(jdx, slice):
        #     print(slice)
        #     return list(itertools.islice(self, 0, 1))
        # else:
        #     print(type(idx))

        # if isinstance(idx, slice):
        #     pass

class RingBuffer:
    '''
    loose implementation of a ring-buffer with numpy arrays
    - only buffers along axis=1
    - consider something like collections.deque for a stricter implementation
    - see also Implementing a Ring Buffer - Python Cookbook: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
    - speed test for roll implementations: https://gist.github.com/cchwala/dea03fb55d9a50660bd52e00f5691db5

    '''
    def __init__(self,buffer_len = 100, initial_val=0):
        if initial_val is None:
            raise ValueError('Error: filling with None results in a dtype=object array\n\
            which is bad for using the buffer later')
        self.n_channels = n_channels 
        self.buffer_len = buffer_len
        if initial_val is not None:
            self.values = np.full(self.buffer_len, initial_val)
    
    def append(self, new_val):
        self.values = np.append(self.values, new_column)
        self.values = self.values[:, 1:]
        # self.values = np.roll(self.values, -1, axis=0)
        # self.values[-1] = new_val

    def last(self):
        return self.values[-1]

    def get_delayed(self, delay=1):
         
        if not 0 <= delay < self.buffer_len:
            raise IndexError(f'delay {delay} out of bounds for buffer length {self.buffer_len}')
        # alternate implementation would
        # cap the delay, so that delay = -Inf returns the oldest value in the array.
        # delay = max(min(delay, self.size),1)
        return self.values[-delay-1]
    def to_np(self):
        'this transpose shouldnt be necessary'
        return self.values
    def fill(self, fill_val):
        self.values.fill(fill_val)
    def __getitem__(self, idx):
        return self.values[idx]
    def __str__(self):
        return f'buffer with len: {self.buffer_len}, \nvals: {str(self.values)}\n'
        
class RingBuffer_2D:
    '''
    (could be made simpler by extending numpy.ndarray)
    loose implementation of a ring-buffer with numpy arrays
    - only buffers along axis=1
    - consider something like collections.deque for a stricter implementation
    - see also Implementing a Ring Buffer - Python Cookbook: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html

    '''
    def __init__(self, n_channels, buffer_len = 100, initial_val=None):
        self.n_channels = n_channels 
        self.buffer_len = buffer_len
        if initial_val is not None:
            self.values = np.full((self.n_channels, self.buffer_len), initial_val)
    
    def append(self, new_column):
        #expands column to 2D 
        self.values = np.append(self.values, new_column[:,None], axis=1)
        self.values = self.values[:, 1:]
        
        # np.roll is about 2x slower than method above for medium-sized arrays
        # self.values = np.roll(self.values, -1, axis=1)
        # self.values[:,-1] = new_column

    def last(self):
        return self.values[:,-1]

    def get_delayed(self, delay=1):
        if not 0 <= delay < self.buffer_len:
            raise IndexError(f'delay {delay} out of bounds for buffer length {self.buffer_len}')
        # alternate implementation would
        # cap the delay, so that delay = -Inf returns the oldest value in the array.
        # delay = max(min(delay, self.size),1)
        return self.values[:,-delay-1]
    def to_np(self):
        'this transpose shouldnt be necessary'
        return self.values
    def fill(self, fill_val):
        self.values.fill(fill_val)
    def __getitem__(self, idx):
        return self.values[idx]
    def __str__(self):
        return f'buffer with {self.n_channels} columns, len: {self.buffer_len}, \nvals: {str(self.values)}\n'
# %%
if __name__ == "__main__":
    import time 
    n_channels = 10
    buff_len = 10000

    np_buff = RingBuffer_2D(n_channels = n_channels, buffer_len=buff_len)
    dq_buff = DQRingBuffer(buff_len, n_channels)
    dq_buff2 = DQRingBuffer(buff_len, n_channels)
    '''
    DeQue ring buffer is generally faster than numpy 2D version
    '''

    ntest = 1000 #10000
    t0 = time.time()

    vec = np.arange(n_channels)
    vec.shape

    dq_buff.fill(vec*0)
    dq_buff2.fill(vec*0)

    # Test numpy 2D array as buffer
    for i in range(ntest):
        np_buff.append( vec+i )
        _ = np_buff.get_delayed(1)

    t1 = time.time()
    num_t = t1-t0
    print('numpy 2D test')
    print(f'{num_t:.3f} seconds for {ntest} iterations\n')

    # Test deque of tuples as buffer
    t0 = time.time()
    for i in range(ntest):
        dq_buff.append( vec+i )
        _ = dq_buff.get_delayed(1)
    t1 = time.time()
    dq_t = t1-t0
    print('deque test')
    print(f'{dq_t:.3f} seconds for {ntest} iterations\n')


    # Test deque of tuples as buffer + time to convert to numpy
    t0 = time.time()
    for i in range(ntest):
        dq_buff2.append( vec+i )
        _ = np.array(dq_buff2.get_delayed(1))
    t1 = time.time()
    dq_conv_t = t1-t0
    print('deque test w/ conversion to numpy')
    print(f'{dq_conv_t:.3f} seconds for {ntest} iterations\n')



    print( f'deque {num_t/dq_t:.2f} times faster')
    print( f'deque {num_t/dq_conv_t:.2f} times faster with conversion back to numpy array') 