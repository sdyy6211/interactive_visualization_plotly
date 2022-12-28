import dash
from dash import html,dcc
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
app.layout = html.Div(children = [
    html.Div(children = dcc.Graph(id = 'fig3')),
    html.Div(children = dcc.Slider(id = 'slider3',value = 1,min=1,max=15,step=1))])

@app.callback(
    Output("fig3", "figure"), #the output is the scatterchart
    [Input("slider3", "value")], #the input is the year-filter
)
def update_chart(D):
    fig3 = go.FigureWidget()
    fig3.add_scatter(x=[],y = [],name = 'In sample error') #0
    fig3.add_scatter(x=[],y = [],name = 'Out of sample error') #1

    fig3.update_yaxes(title = 'Error',range=[0,3])
    fig3.update_xaxes(title = 'Number of samples')
    
    fig3.update_layout(
    autosize=False,
    width=1000,
    height=400)
    
    np.random.seed(0)
    N_range = np.arange(100,1000,10)
    
    train_loss = []
    test_loss = []
    
    for N in N_range:
        ratio = 0.8
        cut = int(N*ratio)
        X = np.linspace(0,15*np.pi/4,N)
        Y = np.sin(X)

        index = range(N)
        train_index = np.random.choice(index,size = cut,replace = False)
        test_index = np.array(list(set(index)-set(train_index)))
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = Y[train_index]+0.5*np.random.randn(cut,)
        y_test = Y[test_index]+0.5*np.random.randn(N-cut,)
        x_train_ = np.concatenate([(x_train**i)[:,None] for i in range(0,D)],axis = 1)
        x_test_ = np.concatenate([(x_test**i)[:,None] for i in range(0,D)],axis = 1)

        model = np.linalg.inv(x_train_.T@x_train_).T@x_train_.T@y_train
        pred_train = x_train_@model
        pred_test = x_test_@model
        
        train_loss.append(np.mean((y_train-pred_train)**2))
        test_loss.append(np.mean((y_test-pred_test)**2))
        
    with fig3.batch_update():
        fig3.data[0].x = N_range*ratio
        fig3.data[0].y = train_loss
        fig3.data[1].x = N_range*ratio
        fig3.data[1].y = test_loss
        
    return fig3

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)