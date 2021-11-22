import numpy as np
'''
would like to be able to detect certain patterns like chains, see circuit-visualizer-p5 for a start
'''

def int2char(idx):
    return chr(int(idx)+97);

def flatidx_to_str(L, N, node_name_f = int2char):
    i = np.floor(L/N)
    j = L%N
    return f'{node_name_f(i)}→{node_name_f(j)}'

def set_yticks_flatadj(ax, adj):
    n = adj.shape[0]
    nn = adj.size
    ax.set_yticks(np.arange(nn))
    ax.set_yticklabels( [ flatidx_to_str(i, n) for i in np.arange(nn) ] ,fontsize=15)
        
def adj_to_str(adj, line_joiner = '_', node_name_f = None):
    # ported from js: https://github.com/awillats/xcorr-visualizer-p5/blob/main/network-simulation.js
    if (node_name_f == None):
        node_name_f = int2char
    
    mat_str = '' 
    n = adj.shape[0]
    
    for i in range(adj.shape[0]):
        col_str_from = f'{node_name_f(i)}→'
        col_str_to = ''
        
        for j in range(adj.shape[1]):
            if (adj[i][j] != 0):
                col_str_to += f'{node_name_f(j)},'
        if col_str_to:
            mat_str += f'{col_str_from}{col_str_to[:-1]}{line_joiner}'
    
    return mat_str
