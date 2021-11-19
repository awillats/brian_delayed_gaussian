# https://stackoverflow.com/questions/63459424/how-to-add-multiple-graphs-to-dash-app-on-a-single-browser-page
import dash
from dash import dcc
from dash import html

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