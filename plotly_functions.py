import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc
from dash import html

from brian_to_dataframe import * 
from dataframe_preprocessing_functions import *

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    https://github.com/plotly/plotly.js/issues/2684#issuecomment-641023041
    https://stackoverflow.com/questions/50488894/plotly-py-change-line-opacity-leave-markers-opaque
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])
    
def fade_color(hex_color, alpha):
    return 'rgba'+str(hex_to_rgba(h=hex_color,alpha=alpha))
    
def go_line(df,x,y,color,legendgroup):
    return go.Scatter(x=df[x],y=df[y],mode='lines',line = dict(color=color),legendgroup=legendgroup)

# %% markdown
# ARCHIVE:

# each channel (neuron of a population) as a row:
'''
fig = px.line(df_m, x='time [ms]', y='voltage', facet_row='population', color='neuron')
fig.update_layout(width=500, height=400)
fig.update_traces(line=dict(width=1))
fig
'''

#Plot history:
'''
figh = px.line(pd.concat([df_m, dfhist_m ] ), x='time [ms]', y='voltage', facet_row='population',color='compare population',
    title=f'last {buffer_len} samples of history saved into buffer')
# figh.update_traces(marker=dict(size=1,opacity=.9))    
figh.update_layout(width=500, height=150*N_groups)

# figh.update_xaxes(range=[0,duration/second])

figh.update_traces(line=dict(width=1))
figh
'''

# timeseries plotly
'''
figt = px.line(melt_group_df_timeseries(df_avg), x='time [ms]', y='voltage', 
    facet_row='population',color='population',labels={'population':'pop'})
figt.update_layout(width=600, height=500)
# figt.write_html('figs/gaussian_timeseries.html')
figt.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1], ))
figt.for_each_annotation(lambda a: a.update(x=-.08, textangle=-90) )
figt.update_layout(showlegend=False)
figt
'''
#cross-correlation plotly
'''
figx = px.line(time_selection_df(dfx_m, time_range, lag_key) , x=lag_key, y='xcorr',
    facet_row='from',color='to')
figx.for_each_annotation(lambda a: a.update(text='from '+a.text.split("=")[-1]))
figx.update_layout(width=350, height=500)
figx.layout.legend.x = 1.15
figx
'''

# %%
LAG_KEY = 'lag [ms]'
TIME_KEY = 'time [ms]'
def df_plot_xcorr(df, dfx, group_names, highlight_window=None, fig_title=None, html_file=None, xcorr_plot_window=[-250,250]):
    #setup dataframe
    df_m = melt_group_df_timeseries(df)
    dfx_m = melt_hier_df_timeseries(dfx,'from','to','xcorr',LAG_KEY)
    N_groups = len(group_names)
    #setup figure
    fig = make_subplots(N_groups,2, column_widths=[0.8, 0.2],
        shared_xaxes=True, shared_yaxes='columns', 
        y_title='voltages',
        row_titles=[g+'→' for g in group_names], column_titles=['outputs','xcorr'])
    
    fig.update_layout(width=850,height=500)
    #setup colors
    q_colors = px.colors.qualitative.Plotly   

    def go_line_ij_xcorr(df_ij, color, legendgroup):
        '''
        convenience function for plotting xcorr from i->j
        '''
        gl = go_line(df_ij, x=LAG_KEY, y='xcorr',color=color,legendgroup=legendgroup)
        gl.showlegend=False
        gl.line.width=1
        gl.name=f'{group_names[i]}→{group_names[j]}'
        gl.hovertemplate='lag: %{x:.2f}ms<br>xcorr: %{y:.2f}'
        return gl
    
    #loop 
    for i, group_i in enumerate(group_names):
        ip = i+1
        df_i = df_m[df_m['population']==group_i]
        
        #Plot timeseries output:
        # - using legendgroup ties together interactive toggling of traces based on the legend entry
        #see https://plotly.com/python/legend/#grouped-legend-items for linking legend toggling across groups
        gl = go_line(df_i, x=TIME_KEY, y='voltage', color=q_colors[i], legendgroup=f'group{i}')
        gl.name='→'+group_names[i]
        gl.hovertemplate='t: %{x:.2f}ms<br>V: %{y:.2f}'
        if i==0:
            gl.legendgrouptitle.text="Population"
            pass
        fig.add_trace(gl, row=ip, col=1)
        

        #Plot cross-correlations
        for j, group_j in enumerate(group_names):
            color_j = q_colors[j]
            faded_color_j = fade_color(q_colors[j],.5)
            ij_mask = (dfx_m['from']==group_names[i]) & (dfx_m['to']==group_names[j])

            df_ij = time_selection_df(dfx_m[ij_mask], xcorr_plot_window, LAG_KEY)
            
            if highlight_window is None:
                gl = go_line_ij_xcorr(df_ij, color = color_j,legendgroup=f'group{j}')
                fig.add_trace(gl, row=ip, col=2)
            else:
                df_ij_center = time_selection_df(dfx_m[ij_mask], highlight_window, LAG_KEY)
                gl_fade   = go_line_ij_xcorr(df_ij, color = faded_color_j, legendgroup=f'group{j}')
                gl_center = go_line_ij_xcorr(df_ij_center, color = color_j,legendgroup=f'group{j}')
                gl_center.line.width=2
            
                fig.add_trace(gl_fade,row=ip,col=2)
                fig.add_trace(gl_center,row=ip,col=2)
    fig.update_xaxes(dtick=50,col=2)        
    fig.update_xaxes(title_text=TIME_KEY, row=len(group_names), col=1)
    fig.update_xaxes(title_text=LAG_KEY, row=len(group_names), col=2)

    fig.update_layout(
        title=go.layout.Title(
            text = fig_title,
            xref="paper",
            x=0
        ))
    if html_file is not None:
        fig.write_html(html_file)
    return fig

#%%
# https://stackoverflow.com/questions/63459424/how-to-add-multiple-graphs-to-dash-app-on-a-single-browser-page


'''
if port is in use, try:
> sudo lsof -i:8050                                                                                                                                                                                              🐍:(neuroenv) 
then:
> kill <id of what came up>
https://stackoverflow.com/questions/19071512/socket-error-errno-48-address-already-in-use
'''

def dash_app_from_figs(fig_list, col_widths=None, title='',subtitle='', port=8050, do_debug=True):
    '''
    horizontally concatenates several 
    '''
    n_fig = len(fig_list)
    
    int2word = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
          6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
          11 : 'eleven', 12 : 'twelve'}
    
    if col_widths is None:
        col_widths = [ int2word[round(12/n_fig)] for i in range(n_fig) ]
    
    # app = dash.Dash(__name__)
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    # https://stackoverflow.com/questions/63459424/how-to-add-multiple-graphs-to-dash-app-on-a-single-browser-page

    fig_divs = [html.Div([dcc.Graph( id=f'graph{i}', figure=fig)], className=f'{col_widths[i]} columns') for i,fig in enumerate(fig_list)]
    
    app.layout = html.Div(children=[
        html.Div([
            html.Div([
                html.H1(children=title),
                html.Div(children=subtitle),
                html.Div(children = fig_divs),
            ], className='row'),
        ], className='row'),
    ])

    try:
        # if __name__ == '__main__':
        app.run_server(debug=do_debug, port=port)
    except ValueError:
        print(err)