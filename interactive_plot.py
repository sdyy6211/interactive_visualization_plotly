import dash

import plotly.graph_objs as go
import numpy as np
import pandas as pd

import plotly

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import plotly.express as px
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions = True)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("BENV0115", className="display-4",style = {'font-size':'35px'}),
        html.Hr(),
        html.P(
            "Visual demostrations of machine learning algorithms", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Gradient Descent", href="/page-1", active="exact"),
                dbc.NavLink("Polynomial Regression", href="/page-2", active="exact"),
                dbc.NavLink("Learning Curve", href="/page-3", active="exact"),
                dbc.NavLink("Logistic Regression - Fitting", href="/page-4", active="exact"),
                dbc.NavLink("Logistic Regression - Tuning", href="/page-5", active="exact"),
                dbc.NavLink("Logistic Regression - Non-linear", href="/page-6", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

fig2 = html.Div(children = dcc.Graph(id = 'fig2'))

mode = html.Div(children = [html.Label(['Mode:']),
                            dcc.Dropdown(id = 'mode', options = ['Interpolate','Extrapolate'],
                                         value = 'Interpolate')])
    
slider_N = html.Div(children = [html.Label(['N:']),
                                dcc.Slider(id = 'slider_N',
                                            value = 10,
                                            min=10,
                                            max=1000,
                                            step=1,
                                          marks = {10:'10',
                                                  1000:'1000'})])

slider_D = html.Div(children = [html.Label(['D:']),
                                dcc.Slider(id = 'slider_D',
                                            value = 1,
                                            min=1,
                                            max=15,
                                            step=1)])

slider_r = html.Div(children = [html.Label(['r:']),
                                dcc.Slider(id = 'slider_r',
                                            value = 0.6,
                                            min=0.5,
                                            max=0.9,
                                            step=0.1)])

slider_V = html.Div(children = [html.Label(['V:']),
                                dcc.Slider(id = 'slider_V',
                                            value = 0.5,
                                            min=0.1,
                                            max=1.5,
                                            step=0.1)])

fig2components = html.Div(children = [fig2,
                                      mode,
                                      slider_N,
                                     slider_D,
                                     slider_r,
                                     slider_V])

@app.callback(
    Output("fig2", "figure"), #the output is the scatterchart
    [Input("mode", "value"),
    Input("slider_N", "value"),
    Input("slider_D", "value"),
    Input("slider_r", "value"),
    Input("slider_V", "value")], #the input is the year-filter
)
def update_chart2(mode, slider_N, slider_D, slider_r, slider_V):
    fig2 = go.FigureWidget()
    fig2.add_scatter(x=[],y = [],mode='markers',name = 'In sample points') #0
    fig2.add_scatter(x=[],y = [],mode='markers',name = 'Out of sample points') #1
    fig2.add_scatter(x=[],y = []) #2
    fig2.add_scatter(x=[],y = [],mode='markers',marker_symbol = 'cross',marker = dict(color='orange')) #3

    fig2.update_yaxes(title = 'y',range=[-3,3])
    fig2.update_xaxes(title = 'x')
    
    fig2.update_layout(
    autosize=False,
    width=1000,
    height=400)
    
    np.random.seed(0)
    D = slider_D
    N = slider_N
    ratio = slider_r
    V = slider_V
    cut = int(N*ratio)
    X = np.linspace(0,15*np.pi/4,N)
    Y = np.sin(X)
    if mode == 'Interpolate':
        index = range(N)
        train_index = np.random.choice(index,size = cut,replace = False)
        test_index = np.array(list(set(index)-set(train_index)))
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = Y[train_index]+V*np.random.randn(cut,)
        y_test = Y[test_index]+V*np.random.randn(N-cut,)
    elif mode == 'Extrapolate':
        index = range(N)
        train_index = index[:cut]
        test_index = index[cut:]
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = Y[train_index]+V*np.random.randn(cut,)
        y_test = Y[test_index]+V*np.random.randn(N-cut,)
    x_train_ = np.concatenate([(x_train**i)[:,None] for i in range(0,D)],axis = 1)
    x_test_ = np.concatenate([(x_test**i)[:,None] for i in range(0,D)],axis = 1)
    model = np.linalg.inv(x_train_.T@x_train_).T@x_train_.T@y_train
    pred_train = x_train_@model
    pred_test = x_test_@model
    all_pred = np.zeros((N,))
    all_pred[train_index] = pred_train
    all_pred[test_index] = pred_test
    with fig2.batch_update():
        fig2.data[0].x = x_train
        fig2.data[0].y = y_train
        fig2.data[1].x = x_test
        fig2.data[1].y = y_test
        fig2.data[2].x = X
        fig2.data[2].y = all_pred
        fig2.data[2].name = 'Prediction ( in sample error = {0})'.format(round(np.mean((y_train-pred_train)**2),4))
        fig2.data[3].x = x_test
        fig2.data[3].y = pred_test
        fig2.data[3].name = 'Out of sample predictions (error = {0})'.format(round(np.mean((y_test-pred_test)**2),4))
        
    return fig2

fig3 = html.Div(children = dcc.Graph(id = 'fig3'))

slider3 = html.Div(children = [html.Label(['D:']),
                                dcc.Slider(id = 'slider3',value = 1,min=1,max=15,step=1)])

fig3components = html.Div(children = [fig3,
                                      slider3])

@app.callback(
    Output("fig3", "figure"), #the output is the scatterchart
    [Input("slider3", "value")], #the input is the year-filter
)
def update_chart3(D):
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
    N_range = np.arange(50,500,50)
    
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

fig3 = html.Div(children = dcc.Graph(id = 'fig3'))

slider3 = html.Div(children = [html.Label(['D:']),
                                dcc.Slider(id = 'slider3',value = 1,min=1,max=15,step=1)])

fig3components = html.Div(children = [fig3,
                                      slider3])

fig4 = html.Div(children = dcc.Graph(id = 'fig4'))

slider_nepoch = html.Div(children = [html.Label(['NEPOCH:']),
                                     dcc.RangeSlider(id = 'nepoch',
                                                value = [0,500],
                                                min=0,
                                                max=3000,
                                                step=1,
                                                marks = {0:'0',
                                                        1000:'1000',
                                                        2000:'2000',
                                                        3000:'3000'})])

slider_weight = html.Div(children = [html.Label(['Weight:']),
                                     dcc.Slider(id = 'weight',
                                                value = -5,
                                                min=-7,
                                                max=7,
                                                step=0.15,
                                               marks = {-7:'-7',
                                                       0:'0',
                                                       7:'7'})])

slider_learning_rate = html.Div(children = [html.Label(['Learning Rate:']),
                                            dcc.Slider(id = 'learning_rate',
                                                value = -1,
                                                min=-3,
                                                max=1,
                                                step=0.5)])

fig4components = html.Div(children = [fig4,
                                      slider_nepoch,
                                     slider_weight,
                                     slider_learning_rate])

@app.callback(
    Output("fig4", "figure"), #the output is the scatterchart
    [Input("nepoch", "value"),
    Input("weight", "value"),
    Input("learning_rate", "value")], #the input is the year-filter
)
def update_chart4(nepoch, weight, learning_rate):
    np.random.seed(0)
    x_,y_ = make_regression(n_features = 1,random_state = 0,noise = 50)
    x_ = x_.reshape(-1,)
    x_ = (x_ - x_.mean())/x_.std()
    y_ = (y_ - y_.mean())/y_.std()

    lr = LinearRegression(fit_intercept = False)

    _ = lr.fit(x_.reshape(-1, 1),y_.reshape(-1, 1))

    optimal_value = lr.coef_[0][0]

    optimal_loss = ((y_-x_*optimal_value)**2).mean()

    optimal_derivative = -2*(y_-x_*optimal_value)@x_

    beta_range = np.arange(-10,10,0.1)
    loss_list = ((np.outer(x_,beta_range)-np.repeat(y_[:,None],axis = 1,repeats = beta_range.shape[0]))**2).mean(axis = 0)

    def train(beta_init,alpha,nepoch):
        current_beta = beta_init
        alpha = alpha
        nepoch = nepoch
        loss_list_train = []
        beta_list = []
        next_beta_list = []
        beta_gradient_list = []
        for i in range(nepoch):
            beta_list.append(current_beta)
            loss = ((y_-x_*current_beta)**2).mean()
            loss_list_train.append(loss)
            beta_gradient = (-2*(y_-x_*current_beta)@x_)/x_.shape[0]
            beta_gradient_list.append(beta_gradient)
            current_beta = current_beta-alpha*beta_gradient
            next_beta_list.append(current_beta)
        return loss_list_train,beta_list,next_beta_list,beta_gradient_list

    fig4 = go.FigureWidget(plotly.subplots.make_subplots(1,2))
    fig4.add_scatter(x=x_,y = y_,mode='markers',row = 1,col = 1,name = 'Datapoints') #0
    fig4.add_scatter(x = [],y = [],row = 1,col = 1,name = 'Regression Line') #1

    fig4.add_scatter(x=beta_range,y = loss_list,row = 1,col = 2,name = 'Loss') #2
    fig4.add_scatter(x = [],y = [],row = 1,col = 2) #3
    fig4.add_scatter(x = [],y = [],row = 1,col = 2) #4
    fig4.add_scatter(x = [],y = [],row = 1,col = 2) #5
    fig4.add_scatter(x = [],y = [],row = 1,col = 2,mode='markers') #6
    fig4.add_scatter(x = [],y = [],row = 1,col = 2,mode='markers') #7
    fig4.add_scatter(x = [optimal_value],y = [optimal_loss],row = 1,col = 2,mode='markers',
                    name = 'Optimal point (weight = {0}, loss = {1})'.format(round(optimal_value,4),round(optimal_loss,4))) #7

    fig4.update_xaxes(title = 'x',range=[min(x_)*0.9,max(x_)*0.9], row=1, col=1)
    fig4.update_yaxes(title = 'y',range=[-3,3], row=1, col=1)

    fig4.update_xaxes(title = 'Weight',range=[-7,7], row=1, col=2)
    fig4.update_yaxes(title = 'Error',range=[-5,55], row=1, col=2)
    
    fig4.update_layout(
    autosize=False,
    width=1000,
    height=400)
    
    beta = weight
    alpha = 10**learning_rate
    total_epoch = nepoch[1]
    nepoch = min(nepoch[0],nepoch[1]-1)
    loss_list_train,beta_list,next_beta_list,beta_gradient_list = train(beta_init = beta,alpha = alpha,nepoch = total_epoch)
    xa = np.arange(min(x_)*0.9,max(x_)*0.9,0.1)
    xb = np.arange(min(beta_range)*0.9,max(beta_range)*0.9,0.1)
    with fig4.batch_update():
        fig4.data[1].x = xa
        fig4.data[1].y = xa*beta_list[nepoch]
        fig4.data[1].name = 'Regression Line (slope/weight = {0})'.format(round(beta_list[nepoch],4))
        
        fig4.data[2].name = 'Current error = {0}'.format(round(loss_list_train[nepoch],4))
        
        fig4.data[3].x = beta_list[nepoch]+xb
        fig4.data[3].y = ((y_-x_*beta_list[nepoch])**2).mean()+xb*beta_gradient_list[nepoch]
        fig4.data[3].name = 'Derivative/gradient = {0}'.format(round(beta_gradient_list[nepoch],4))
        
        fig4.data[4].x = [beta_list[nepoch],next_beta_list[nepoch]]
        fig4.data[4].y = [((y_-x_*beta_list[nepoch])**2).mean(),((y_-x_*beta_list[nepoch])**2).mean()]
        fig4.data[4].name = 'Update value (lr*derivative) = {0}'.format(round(alpha*beta_gradient_list[nepoch],4))
        
        fig4.data[5].x = [next_beta_list[nepoch],next_beta_list[nepoch]]
        fig4.data[5].y = [((y_-x_*beta_list[nepoch])**2).mean(),((y_-x_*next_beta_list[nepoch])**2).mean()]
        fig4.data[5].name = 'Error reduction = {0}'.format(round(((y_-x_*beta_list[nepoch])**2).mean()-((y_-x_*next_beta_list[nepoch])**2).mean(),4))
        
        fig4.data[6].x = [beta_list[nepoch]]
        fig4.data[6].y = [((y_-x_*beta_list[nepoch])**2).mean()]
        fig4.data[6].name = 'Current weight = {0}'.format(round(beta_list[nepoch],4))
        
        fig4.data[7].x = [next_beta_list[nepoch]]
        fig4.data[7].y = [((y_-x_*beta_list[nepoch])**2).mean()]
        fig4.data[7].name = 'Next weight = {0} = {1}{3}{2}'.format(round(next_beta_list[nepoch],4),
                                                                round(beta_list[nepoch],4),
                                                                abs(round(alpha*beta_gradient_list[nepoch],4)),
                                                                '-' if beta_gradient_list[nepoch] >0 else '+')

    return fig4

fig5 = html.Div(children = dcc.Graph(id = 'fig5'))

slider_fig5_nepoch = html.Div(children = [html.Label(['NEpoch:']),
                                     dcc.Slider(id = 'nepoch_fig5',
                                                value = 0,
                                                min=0,
                                                max=100,
                                                step=1,
                                               marks = {0:'0',
                                                       25:'25',
                                                       50:'50',
                                                       75:'75',
                                                       100:'100'})])


fig5components = html.Div(children = [fig5,
                                      slider_fig5_nepoch])

@app.callback(
    Output("fig5", "figure"), #the output is the scatterchart
    [Input("nepoch_fig5", "value")], #the input is the year-filter
)
def update_chart5(nepoch_fig5):
    
    epoch = nepoch_fig5
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def decision_boundary(x,intercept,coef1,coef2):
        return -intercept/(coef2+1e-4)+(-coef1/(coef2+1e-4))*x

    def lgmodel(x1,x2,intercept,coef1,coef2):
        return sigmoid(intercept+x1*coef1+x2*coef2)

    def loss_func(y_,pred):
        return -(y_*np.log(sigmoid(pred))+(1-y_)*np.log(1-sigmoid(pred))).mean()

    def loss_func2(y,pred):
        return np.log(1+np.exp(-y*pred)).mean()

    def loss_func3(y,pred):
        return (-y*pred+np.log(1+np.exp(pred))).mean()
    
    def train_logit(x_,y_,beta_init,alpha,nepoch):
        current_beta = beta_init
        alpha = alpha
        nepoch = nepoch
        loss_list_train = []
        beta_list = []
        next_beta_list = []
        beta_gradient_list = []
        for i in range(nepoch):
            beta_list.append(current_beta)
            if len(x_.shape)<2:
                pred_ = sigmoid(x_*current_beta)
            else:
                pred_ = sigmoid(x_@current_beta)
            loss = -(y_*np.log(pred_)+(1-y_)*np.log(1-pred_)).mean()
            loss_list_train.append(loss)
            beta_gradient = ((pred_-y_).T@x_)/x_.shape[0]
            beta_gradient_list.append(beta_gradient)
            current_beta = current_beta-alpha*beta_gradient
            next_beta_list.append(current_beta)
        return loss_list_train,beta_list,next_beta_list,beta_gradient_list

    np.random.seed(0)
    x_,y_ = make_classification(n_features = 1,random_state = 0,n_redundant = 0,n_informative = 1,n_clusters_per_class = 1,flip_y = 0.1,class_sep  = 0.3,weights = [0.8,0.2])
    x_ = x_.reshape(-1,)
    x_ = (x_ - x_.mean())/x_.std()

    lr = LogisticRegression(fit_intercept = False,penalty = 'none')

    _ = lr.fit(x_.reshape(-1, 1),y_.reshape(-1, 1))

    optimal_value = lr.coef_[0][0]

    pred_ = x_*optimal_value

    optimal_loss = loss_func(y_,pred_)

    optimal_derivative = (x_*optimal_value-y_).T@x_

    beta_range = np.arange(-10,10,0.1)
    y_whole = np.repeat(y_[:,None],axis = 1,repeats = beta_range.shape[0])
    pred_whole = sigmoid(np.outer(x_,beta_range))
    loss_list = -(y_whole*np.log(pred_whole)+(1-y_whole)*np.log(1-pred_whole)).mean(axis = 0)

    x_1,y_1 = make_classification(n_features = 2,random_state = 0,n_redundant = 0,n_informative = 1,n_clusters_per_class = 1,flip_y = 0.1,class_sep  = 0.4,weights = [0.65,0.35])
    x_1 = (x_1 - x_1.mean(axis = 0))/x_.std(axis = 0)
    x_1_ = np.c_[np.ones((100,1)),x_1]

    nepoch = 101
    loss_list_train,beta_list,next_beta_list,beta_gradient_list = train_logit(x_1_,y_1,[0,0,0],3,nepoch)

    fig5 = go.FigureWidget(plotly.subplots.make_subplots(1,2))
    fig5.add_scatter(x=x_1[np.where(y_1==0)[0],0],y = x_1[np.where(y_1==0)[0],1],mode='markers',row = 1,col = 1,name = 'negative samples',line_color = 'blue') #0
    fig5.add_scatter(x=x_1[np.where(y_1==1)[0],0],y = x_1[np.where(y_1==1)[0],1],mode='markers',row = 1,col = 1,name = 'positive samples',line_color = 'red') #1
    fig5.add_scatter(x=[],y = [],row = 1,col = 1,name = 'Decision Boundary',line_color = 'black',opacity = 0.5) #2
    colorscale = [[0, 'blue'], [1, 'red']]
    xx1_ = np.arange(-3,3,0.1)
    xx2_ = np.arange(-3,3,0.1)
    fig5.add_contour(x = xx1_,y = xx2_,z=[],colorscale=colorscale,opacity = 0.5,row = 1,col = 1,showscale=False,contours=dict(showlabels = True,labelfont = dict( # label font properties
                    size = 8,
                    color = 'white'))) #3

    fig5.add_scatter(x=np.arange(0,nepoch-1,1),y = loss_list_train,row = 1,col = 2,name = 'Error Curve') #4
    fig5.add_scatter(x = [],y = [],row = 1,col = 2) #5

    fig5.update_xaxes(title = 'x1',range=[min(x_1[:,0])*1.2,max(x_1[:,0])*1.2], row=1, col=1)
    fig5.update_yaxes(title = 'x2',range=[min(x_1[:,1])*1.2,max(x_1[:,1])*1.2], row=1, col=1)

    fig5.update_xaxes(title = 'Epoch',range=[-1,nepoch], row=1, col=2)
    fig5.update_yaxes(title = 'Error',range=[min(loss_list_train)*0.9,max(loss_list_train)*1.1], row=1, col=2)
    
    fig5.update_layout(
    autosize=False,
    width=1000,
    height=400)
    
    intercept = beta_list[epoch][0]
    coef1 = beta_list[epoch][1]
    coef2 = beta_list[epoch][2]
    
    x1 = np.arange(-3,3,0.01)
    y1 = decision_boundary(x1,intercept,coef1, coef2)
    
    xx1 = np.arange(-3,3,0.1)
    xx2 = np.arange(-3,3,0.1)
    
    xx_1,xx_2 = np.meshgrid(xx1,xx2)
    
    zz = sigmoid(intercept+xx_1*coef1+xx_2*coef2)
    
    with fig5.batch_update():
        fig5.data[2].x = x1
        fig5.data[2].y = y1
        fig5.data[3].z = zz
        
        fig5.data[5].x = [epoch]
        fig5.data[5].y = [loss_list_train[epoch]]
        fig5.data[5].name = 'Error: {0}'.format(round(loss_list_train[epoch],4))
        
    return fig5

fig6 = html.Div(children = dcc.Graph(id = 'fig6'))

slider_fig6_intercept = html.Div(children = [html.Label(['Intercept:']),
                                     dcc.Slider(id = 'slider_fig6_intercept',
                                            value = 0,
                                            min=-5,
                                            max=5,
                                            step=0.1,
                                               marks = {-5:'-5',
                                                       0:'0',
                                                       5:'5'})])

slider_fig6_coef1 = html.Div(children = [html.Label(['Coef1:']),
                                     dcc.Slider(id = 'slider_fig6_coef1',
                                            value = 0,
                                            min=-7,
                                            max=7,
                                            step=0.1,
                                            marks = {-7:'-7',
                                                    0:'0',
                                                    7:'7'})])

slider_fig6_coef2 = html.Div(children = [html.Label(['Coef2:']),
                                     dcc.Slider(id = 'slider_fig6_coef2',
                                            value = 0,
                                            min=-7,
                                            max=7,
                                            step=0.1,
                                               marks = {-7:'-7',
                                                       0:'0',
                                                       7:'7'})])


fig6components = html.Div(children = [fig6,
                                      slider_fig6_intercept,
                                     slider_fig6_coef1,
                                     slider_fig6_coef2])

@app.callback(
    Output("fig6", "figure"), #the output is the scatterchart
    [Input("slider_fig6_intercept", "value"),
    Input("slider_fig6_coef1", "value"),
    Input("slider_fig6_coef2", "value")], #the input is the year-filter
)
def update_chart6(slider_fig6_intercept,slider_fig6_coef1, slider_fig6_coef2):
    
    intercept = slider_fig6_intercept
    coef1 = slider_fig6_coef1
    coef2 = slider_fig6_coef2
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def decision_boundary(x,intercept,coef1,coef2):
        return -intercept/(coef2+1e-4)+(-coef1/(coef2+1e-4))*x

    def lgmodel(x1,x2,intercept,coef1,coef2):
        return sigmoid(intercept+x1*coef1+x2*coef2)

    def loss_func(y_,pred):
        return -(y_*np.log(sigmoid(pred))+(1-y_)*np.log(1-sigmoid(pred))).mean()

    def loss_func2(y,pred):
        return np.log(1+np.exp(-y*pred)).mean()

    def loss_func3(y,pred):
        return (-y*pred+np.log(1+np.exp(pred))).mean()
    
    def train_logit(x_,y_,beta_init,alpha,nepoch):
        current_beta = beta_init
        alpha = alpha
        nepoch = nepoch
        loss_list_train = []
        beta_list = []
        next_beta_list = []
        beta_gradient_list = []
        for i in range(nepoch):
            beta_list.append(current_beta)
            if len(x_.shape)<2:
                pred_ = sigmoid(x_*current_beta)
            else:
                pred_ = sigmoid(x_@current_beta)
            loss = -(y_*np.log(pred_)+(1-y_)*np.log(1-pred_)).mean()
            loss_list_train.append(loss)
            beta_gradient = ((pred_-y_).T@x_)/x_.shape[0]
            beta_gradient_list.append(beta_gradient)
            current_beta = current_beta-alpha*beta_gradient
            next_beta_list.append(current_beta)
        return loss_list_train,beta_list,next_beta_list,beta_gradient_list

    np.random.seed(0)
    x_,y_ = make_classification(n_features = 1,random_state = 0,n_redundant = 0,n_informative = 1,n_clusters_per_class = 1,flip_y = 0.1,class_sep  = 0.3,weights = [0.8,0.2])
    x_ = x_.reshape(-1,)
    x_ = (x_ - x_.mean())/x_.std()

    lr = LogisticRegression(fit_intercept = False,penalty = 'none')

    _ = lr.fit(x_.reshape(-1, 1),y_.reshape(-1, 1))

    optimal_value = lr.coef_[0][0]

    pred_ = x_*optimal_value

    optimal_loss = loss_func(y_,pred_)

    optimal_derivative = (x_*optimal_value-y_).T@x_

    beta_range = np.arange(-10,10,0.1)
    y_whole = np.repeat(y_[:,None],axis = 1,repeats = beta_range.shape[0])
    pred_whole = sigmoid(np.outer(x_,beta_range))
    loss_list = -(y_whole*np.log(pred_whole)+(1-y_whole)*np.log(1-pred_whole)).mean(axis = 0)

    x_1,y_1 = make_classification(n_features = 2,random_state = 0,n_redundant = 0,n_informative = 1,n_clusters_per_class = 1,flip_y = 0.1,class_sep  = 0.4,weights = [0.65,0.35])
    x_1 = (x_1 - x_1.mean(axis = 0))/x_.std(axis = 0)
    x_1_ = np.c_[np.ones((100,1)),x_1]

    nepoch = 101
    loss_list_train,beta_list,next_beta_list,beta_gradient_list = train_logit(x_1_,y_1,[0,0,0],3,nepoch)

    fig6 = go.FigureWidget(plotly.subplots.make_subplots(1,2))
    fig6.add_scatter(x=x_1[np.where(y_1==0)[0],0],y = x_1[np.where(y_1==0)[0],1],mode='markers',row = 1,col = 1,name = 'negative samples',line_color = 'blue') #0
    fig6.add_scatter(x=x_1[np.where(y_1==1)[0],0],y = x_1[np.where(y_1==1)[0],1],mode='markers',row = 1,col = 1,name = 'positive samples',line_color = 'red') #1
    fig6.add_scatter(x=[],y = [],row = 1,col = 1,name = 'Decision Boundary',line_color = 'black',opacity = 0.5) #2
    colorscale = [[0, 'blue'], [1, 'red']]
    xx1 = np.arange(-3,3,0.1)
    xx2 = np.arange(-3,3,0.1)
    fig6.add_contour(x = xx1,y = xx2,z=[],colorscale=colorscale,opacity = 0.5,row = 1,col = 1,showscale=False,contours=dict(showlabels = True,labelfont = dict( # label font properties
                    size = 8,
                    color = 'white'))) #3

    fig6.update_xaxes(title = 'x1',range=[min(x_1[:,0])*1.2,max(x_1[:,0])*1.2], row=1, col=1)
    fig6.update_yaxes(title = 'x2',range=[min(x_1[:,1])*1.2,max(x_1[:,1])*1.2], row=1, col=1)
    
    fig6.update_layout(
    autosize=False,
    width=1000,
    height=400)
    
    x1 = np.arange(-3,3,0.01)
    y1 = decision_boundary(x1,intercept,coef1, coef2)
    
    xx1 = np.arange(-3,3,0.1)
    xx2 = np.arange(-3,3,0.1)
    
    xx_1,xx_2 = np.meshgrid(xx1,xx2)
    
    zz = sigmoid(intercept+xx_1*coef1+xx_2*coef2)
    
    with fig6.batch_update():
        fig6.data[2].x = x1
        fig6.data[2].y = y1
        fig6.data[3].z = zz
        
    return fig6

fig7 = html.Div(children = dcc.Graph(id = 'fig7'))

slider_fig7_degree = html.Div(children = [html.Label(['Degree:']),
                                     dcc.Slider(id = 'slider_fig7_degree',
                                                value = 2,
                                                min=2,
                                                max=8,
                                                step=1)])


fig7components = html.Div(children = [fig7,
                                      slider_fig7_degree])

@app.callback(
    Output("fig7", "figure"), #the output is the scatterchart
    [Input("slider_fig7_degree", "value")], #the input is the year-filter
)
def update_chart7(slider_fig7_degree):
    
    degree = slider_fig7_degree
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def decision_boundary(x,intercept,coef1,coef2):
        return -intercept/(coef2+1e-4)+(-coef1/(coef2+1e-4))*x

    def lgmodel(x1,x2,intercept,coef1,coef2):
        return sigmoid(intercept+x1*coef1+x2*coef2)

    def loss_func(y_,pred):
        return -(y_*np.log(sigmoid(pred))+(1-y_)*np.log(1-sigmoid(pred))).mean()

    def loss_func2(y,pred):
        return np.log(1+np.exp(-y*pred)).mean()

    def loss_func3(y,pred):
        return (-y*pred+np.log(1+np.exp(pred))).mean()
    
    def train_logit(x_,y_,beta_init,alpha,nepoch):
        current_beta = beta_init
        alpha = alpha
        nepoch = nepoch
        loss_list_train = []
        beta_list = []
        next_beta_list = []
        beta_gradient_list = []
        for i in range(nepoch):
            beta_list.append(current_beta)
            if len(x_.shape)<2:
                pred_ = sigmoid(x_*current_beta)
            else:
                pred_ = sigmoid(x_@current_beta)
            loss = -(y_*np.log(pred_)+(1-y_)*np.log(1-pred_)).mean()
            loss_list_train.append(loss)
            beta_gradient = ((pred_-y_).T@x_)/x_.shape[0]
            beta_gradient_list.append(beta_gradient)
            current_beta = current_beta-alpha*beta_gradient
            next_beta_list.append(current_beta)
        return loss_list_train,beta_list,next_beta_list,beta_gradient_list

    np.random.seed(0)
    x_,y_ = make_classification(n_features = 1,random_state = 0,n_redundant = 0,n_informative = 1,n_clusters_per_class = 1,flip_y = 0.1,class_sep  = 0.3,weights = [0.8,0.2])
    x_ = x_.reshape(-1,)
    x_ = (x_ - x_.mean())/x_.std()

    lr = LogisticRegression(fit_intercept = False,penalty = 'none')

    _ = lr.fit(x_.reshape(-1, 1),y_.reshape(-1, 1))

    optimal_value = lr.coef_[0][0]

    pred_ = x_*optimal_value

    optimal_loss = loss_func(y_,pred_)

    optimal_derivative = (x_*optimal_value-y_).T@x_

    beta_range = np.arange(-10,10,0.1)
    y_whole = np.repeat(y_[:,None],axis = 1,repeats = beta_range.shape[0])
    pred_whole = sigmoid(np.outer(x_,beta_range))
    loss_list = -(y_whole*np.log(pred_whole)+(1-y_whole)*np.log(1-pred_whole)).mean(axis = 0)

    x_1,y_1 = make_classification(n_features = 2,random_state = 0,n_redundant = 0,n_informative = 1,n_clusters_per_class = 1,flip_y = 0.1,class_sep  = 0.4,weights = [0.65,0.35])
    x_1 = (x_1 - x_1.mean(axis = 0))/x_.std(axis = 0)
    x_1_ = np.c_[np.ones((100,1)),x_1]

    nepoch = 101
    loss_list_train,beta_list,next_beta_list,beta_gradient_list = train_logit(x_1_,y_1,[0,0,0],3,nepoch)

    fig7 = go.FigureWidget(plotly.subplots.make_subplots(1,2))
    fig7.add_scatter(x=x_1[np.where(y_1==0)[0],0],y = x_1[np.where(y_1==0)[0],1],mode='markers',row = 1,col = 1,name = 'negative samples',line_color = 'blue') #0
    fig7.add_scatter(x=x_1[np.where(y_1==1)[0],0],y = x_1[np.where(y_1==1)[0],1],mode='markers',row = 1,col = 1,name = 'positive samples',line_color = 'red') #1
    colorscale = [[0, 'blue'], [1, 'red']]
    xx1__ = np.arange(-3,3,0.1)
    xx2__ = np.arange(-3,3,0.1)
    fig7.add_contour(x = xx1__,y = xx2__,z=[],colorscale=colorscale,opacity = 0.5,row = 1,col = 1,showscale=False,contours=dict(showlabels = True,labelfont = dict( # label font properties
                    size = 8,
                    color = 'white'))) #3

    fig7.update_xaxes(title = 'x1',range=[min(x_1[:,0])*1.2,max(x_1[:,0])*1.2], row=1, col=1)
    fig7.update_yaxes(title = 'x2',range=[min(x_1[:,1])*1.2,max(x_1[:,1])*1.2], row=1, col=1)
    
    fig7.update_layout(
    autosize=False,
    width=1000,
    height=400)
    
    newx_ = np.hstack([np.ones((x_1.shape[0],1))]+[x_1**i for i in range(1,degree)])
    
    lg = LogisticRegression(fit_intercept=False,penalty = 'none',max_iter=100)
    lg.fit(newx_,y_1.reshape(-1,1))
    
    xx1 = np.arange(-3,3,0.1)
    xx2 = np.arange(-3,3,0.1)

    xx_1,xx_2 = np.meshgrid(xx1,xx2)
    
    newxx_ = np.hstack([xx_1.reshape(-1,1),xx_2.reshape(-1,1)])
    newxx__ = np.hstack([np.ones((newxx_.shape[0],1))]+[newxx_**i for i in range(1,degree)])

    zz = lg.predict_proba(newxx__)[:,1].reshape(xx_1.shape)

    with fig7.batch_update():
        fig7.data[2].z = zz
        
    return fig7

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("Please select one of the visual demonstrations from the left sidebar and play with it")
    elif pathname == "/page-1":
        return fig4components
    elif pathname == "/page-2":
        return fig2components
    elif pathname == "/page-3":
        return fig3components
    elif pathname == "/page-4":
        return fig5components
    elif pathname == "/page-5":
        return fig6components
    elif pathname == "/page-6":
        return fig7components
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    app.run_server()
