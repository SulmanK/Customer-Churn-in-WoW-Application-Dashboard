# Import packages
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import colorlover as cl
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")


#--------- Pandas Dataframe
url = "https://raw.githubusercontent.com/SulmanK/Customer-Churn-in-World-of-Warcraft/master/data/churn.csv"
churn = pd.read_csv(url, index_col = 0, nrows = 20000)
#churn = pd.read_csv('data/churn.csv')

## Create train and testing sets, and include stratification (0.80 / 0/20)
X = churn[['guild', 'max_level', 'Average_Hour', 'Average_Playing_density']]
y = churn['Playing_after_6_months']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)

## Scale the data
col_names = ['guild','max_level', 'Average_Hour', 'Average_Playing_density']
features = X_train[col_names]
features_test = X_test[col_names]
ct = ColumnTransformer([
        ('somename', StandardScaler(), ['max_level', 'Average_Hour', 'Average_Playing_density'])
    ], remainder = 'passthrough')
X_train_scaled = ct.fit_transform(features)
X_test_scaled = ct.transform(features_test)


#--------------- Models
"""All models are instantiated for binary classification."""

## Logistic Regresion
clf_LR = LogisticRegression(random_state = 0, solver = 'liblinear',
                            C = 25, penalty = 'l2')

## Support Vector Machine
clf_SVM = svm.SVC(kernel = 'linear', C = 0.009)


## K-Nearest Neighbors
clf_KNN = KNeighborsClassifier(n_neighbors = 24, leaf_size = 2,
                               p = 1)

## Random Forest
clf_RF = RandomForestClassifier(n_estimators = 300, min_samples_split = 15,
                                max_depth = None, criterion = 'entropy',
                                bootstrap = True, random_state = 10)


#--------------------------- ROC Curve
def roc_auc_curve(model, X_train, X_test, y_train, y_test ):
    """Given a classification algorithm, it returns the ROC curve with the AUC score,
    inspired from plotly-dash plots. """
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Decide on whether model has a decision function to get the prediction probabilities
    if hasattr(model, "decision_function"):
        decision_test = model.decision_function(X_test)
    else:
        decision_test = model.predict_proba(X_test)[:,1]

    # Obtain the false-positive rate, true-positive rate, and threshold from sci-kit learn roc curve
    fpr, tpr, threshold = roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = roc_auc_score(y_true = y_test, y_score = decision_test)

    # Create our trace using a plotly go object
    trace0 = go.Scatter(
        x = fpr, y = tpr,
        mode = "lines", name = "Test Data",
        marker = {"color": "#13c6e9"}
    )

    # Adjust the figure parameters 
    layout = go.Layout(
        title = dict(
           # text = f"ROC Curve (AUC = {auc_score:.3f})"
            text = f"AUC = {auc_score:.3f}", y = 0.95,
            x = 0.5, xanchor = 'center', yanchor = 'top'),
        
        xaxis = dict(
            title = "False Positive Rate", gridcolor = "#2f3445",
            title_font = dict(size = 22), range = (-0.05, 1.05) ),
        
        yaxis = dict(
            title = "True Positive Rate", gridcolor = "#2f3445",
            title_font = dict(size = 22), range = (-0.05, 1.05) ),
        
        legend = dict(
            x = 0, y = 1.05,
            orientation = "h"),
        
        margin = dict(
            l = 100, r = 10,
            t = 25, b = 40),

        plot_bgcolor = "#282b38",
        paper_bgcolor = "#282b38",
        font = dict(color = "#a5b1cd", size = 18),
        title_font  =  dict(size  =  22),
    )
    
    # Plug in our parameters above to the plotly go figure objects to create our plots
    data = [trace0]
    roc_auc_curve_fig = go.Figure(data = data, layout = layout)
    return roc_auc_curve_fig

# Instances
LR_roc = roc_auc_curve(clf_LR, X_train_scaled, X_test_scaled, y_train, y_test)
SVM_roc = roc_auc_curve(clf_SVM, X_train_scaled, X_test_scaled, y_train, y_test)
KNN_roc = roc_auc_curve(clf_KNN, X_train_scaled, X_test_scaled, y_train, y_test)
RF_roc = roc_auc_curve(clf_RF, X_train_scaled, X_test_scaled, y_train, y_test)

#--------------------------- Precision-Recall Curve
def pr_curve(model, X_train, X_test, y_train, y_test ):
    """ Given a classification algorithm,
    it returns the precision-recall curve with average precision score,
    inspired from plotly-dash plots. """
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Decide on whether model has a decision function to get the prediction probabilities
    if hasattr(model, "decision_function"):
        decision_test = model.decision_function(X_test)
    else:
        decision_test = model.predict_proba(X_test)[:,1]
    
    # Obtain the precision, recall, and threshold values
    precision, recall, thresholds = precision_recall_curve(y_test, decision_test)

    # AP score
    average_precision = average_precision_score(y_test, decision_test)

    # Create our trace using a plotly go object
    trace0 = go.Scatter(
        x = recall, y = precision,
        mode = "lines", name = "Test Data",
        marker = {"color": "#ffb6d0"}
    )

    # Adjust the figure parameters  
    layout = go.Layout(
        title = dict(
            #text = f"Precision-Recall Curve (AP = {average_precision:.3f})",
            text = f"AP = {average_precision:.3f}", y = 0.95,
            x = 0.5, xanchor = 'center', yanchor = 'top'),
        
        xaxis = dict(
            title = "Recall", gridcolor = "#2f3445",
            title_font = dict(size = 22), range = (-0.05, 1.05) ), 
        
        yaxis = dict(
            title = "Precision", gridcolor = "#2f3445",
            title_font = dict(size = 22), range = (-0.05, 1.05),
            constrain = 'domain'),
            
        legend = dict(
            x = 0, y = 1.05,
            orientation = "h"),
        
        margin = dict(
            l = 100, r = 10,
            t = 25, b = 40),

        plot_bgcolor = "#282b38",
        paper_bgcolor = "#282b38",
        font = dict(color = "#a5b1cd", size = 18),
        title_font  =  dict(size  =  22),
    )
    
    # Plug in our parameters above to the plotly go figure objects to create our plots
    data = [trace0]
    pr_curve_fig = go.Figure(data = data, layout = layout)
    
    return pr_curve_fig

# Instances
LR_pr = pr_curve(clf_LR, X_train_scaled, X_test_scaled, y_train, y_test)
SVM_pr = pr_curve(clf_SVM, X_train_scaled, X_test_scaled, y_train, y_test)
KNN_pr = pr_curve(clf_KNN, X_train_scaled, X_test_scaled, y_train, y_test)
RF_pr = pr_curve(clf_RF, X_train_scaled, X_test_scaled, y_train, y_test)

#----------------------- Confusion Matrix
def confusion_matrix_fig(model, X_train, X_test, y_train, y_test):
    """ Given a classification algorithm, it returns the confusion matrix,
    inspired from plotly-dash plots. """
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Get the predictions
    y_pred_test = model.predict(X_test_scaled) 
    
    # Create the confusion matrix
    matrix = confusion_matrix(y_true = y_test, y_pred = y_pred_test)
    
    # Label the true negative, false positive, false negative, and true positive
    tn, fp, fn, tp = matrix.ravel()
    
    # Plot parameters
    values = [tp, fn,
              fp, tn]
    
    label_text = ["True Positive", "False Negative",
                  "False Positive", "True Negative"]
    
    labels = ["TP", "FN",
              "FP", "TN"]
    
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    
    colors = ["#13c6e9", blue[1],
              "#ff916d", "#ff744c"]
    
    # Create the trace of the pie chart
    trace0 = go.Pie(
        labels = label_text,
        values = values,
        hoverinfo = "label+value+percent",
        textinfo = "text+value",
        text = labels,
        sort = False,
        marker = dict(colors = colors),
        insidetextfont = {"color": "white"},
        rotation = 90,
    )
    # Layout parameters
    layout = go.Layout(
       # title = "Confusion Matrix",
        margin = dict(
            l = 50, r = 50,
            t = 100, b = 10),
        
        legend = dict(
            bgcolor = "#282b38", font = {"color": "#a5b1cd"},
            orientation = "h"),
        
        plot_bgcolor = "#282b38",
        paper_bgcolor = "#282b38",
        font = dict(color = "#a5b1cd", size = 18),
        title_font  =  dict(size  =  22),
        width = 500,
        height = 500,
    )
    # Plug in our parameters above to the plotly go figure objects to create our plots
    data = [trace0]
    confusion_matrix_figure = go.Figure(data = data, layout = layout)
    return confusion_matrix_figure

# Instances
LR_cm = confusion_matrix_fig(clf_LR, X_train_scaled, X_test_scaled, y_train, y_test)
SVM_cm = confusion_matrix_fig(clf_SVM, X_train_scaled, X_test_scaled, y_train, y_test)
KNN_cm = confusion_matrix_fig(clf_KNN, X_train_scaled, X_test_scaled, y_train, y_test)
RF_cm = confusion_matrix_fig(clf_RF, X_train_scaled, X_test_scaled, y_train, y_test)



#----------------------------- Comparison Plot
def comparison_plot(input_array):
    """Function that compares the inputted user data to the average subscribed user"""
    
    # X axis categories
    features = ['Max Level', 'Average Playing Hours',
                'Average Playing Density', 'Guild']
    tmp = X_train
    tmp['Playing_after_six_months'] = y_train.values
    
    # Create the average subscriber array with values
    average_array = np.array([46,
                       tmp[tmp['Playing_after_six_months'] == 1].mean().values[2], 
                       tmp[tmp['Playing_after_six_months'] == 1].mean().values[3],
                       1])
    # Plot parameters
    comparison = make_subplots(rows = 1, cols = 4, horizontal_spacing = 0.15)
    
    
    # Plots the input traces
    for i, j in zip([0, 1, 2, 3], ['Input (Max Level)',
                                   'Input (Average Playing Hours)',
                                   'Input (Average Playing Density)',
                                   'Input (Guild)']
                   ):
        comparison.add_trace(go.Bar(
        x = np.array(features[i]),
        y = np.array(input_array[i]),
        name = j,
        marker_color='indianred'), 
                             1, i+1)

    # Plots the average subscribers
    for i, j in zip([0, 1, 2, 3], ['Subscriber (Max Level)',
                                   'Subscriber (Average Playing Hours)',
                                   'Subscriber (Average Playing Density)',
                                   'Subscriber (Guild)']
                   ):
        comparison.add_trace(go.Bar(
        x = np.array(features[i]),
        y = np.array(average_array[i]),
        name = j,
        marker_color='lightsalmon',),
                             1, i+1)


    # Update xaxis properties
    comparison.update_xaxes(title_text = "Max Level", showticklabels = False,
                            row = 1, col = 1,
                            showgrid = False)
    
    comparison.update_xaxes(title_text = "APH", showticklabels = False,
                            row = 1, col = 2,
                            showgrid = False)
    
    comparison.update_xaxes(title_text = "APD", showticklabels = False,
                            row = 1, col = 3,
                            showgrid = False)
    
    comparison.update_xaxes(title_text = "Guild", showticklabels = False,
                            row = 1, col = 4,
                            showgrid = False)
    
    # Update yaxis properties
    comparison.update_yaxes(row = 1, col = 1,
                            showgrid = False, ticks = 'outside',
                            tickwidth = 2,  ticklen = 15,
                            tickcolor = "#a5b1cd" )
    
    comparison.update_yaxes(row = 1, col = 2,
                            showgrid = False, ticks = 'outside',
                            tickwidth = 2,  ticklen = 15,
                            tickcolor = "#a5b1cd")
    
    comparison.update_yaxes(row = 1, col = 3,
                            showgrid = False, ticks = 'outside',
                            tickwidth = 2, ticklen = 15,
                            tickcolor = "#a5b1cd")
    
    comparison.update_yaxes(row = 1, col = 4,
                            showgrid = False, ticks = 'outside',
                            tickwidth = 2, ticklen = 15,
                            tickcolor = "#a5b1cd")
    # Update layout
    comparison.update_layout(barmode = 'group', title = 'Input and Average Subscriber Comparison',
                             title_font = dict(size  =  26), plot_bgcolor = "#282b38",
                             paper_bgcolor = "#282b38", font = dict(color = "#a5b1cd", size = 16))
    
    return comparison




#---------------------------- Dashboard


## Instantiating the dashboard application
app = dash.Dash(__name__)

    
server = app.server
app.config['suppress_callback_exceptions'] = True


## Create dashboard layout
app.layout = html.Div(
    [
        html.Div(
            
            # Create Banner Layout (Title and Logos)
            
            className = "banner",
            children = [
                html.Div(
                    className = "row",
                    children = [
            
                        # Input Title of Dashboard, Include title and href link
                        
                        html.H2(
                            id = "banner-title",
                            children = [
                                html.A(
                                    "Predicting Customer Churn in WoW",
                                    href = "https://github.com/SulmanK/Customer-Churn-in-World-of-Warcraft",
                                    style = {"text-decoration": "none", "color": "inherit"},
                                )
                            ], style = {'padding-left': '4rem'}
                        ), 
                       
                        # Insert Github Logo with href link
                        
                        html.A(
                            [
                                html.Img(src = app.get_asset_url("github_banner.png"))
                            ], href = "https://github.com/SulmanK/Customer-Churn-in-World-of-Warcraft",
                        ),
                        
                        # Insert WoW Logo with href link
                        
                        html.A(
                            [
                                html.Img(src = app.get_asset_url("wow_logo_banner.png"))
                            ], href = "https://worldofwarcraft.com/en-us/",
                        ),
                        
                        # Insert Dash logo with href link
                        
                        html.A( 
                            [
                                html.Img(src=app.get_asset_url("dash_banner.png")) 
                            ], href = "https://dash.plotly.com/",
                        ), 
                    ],
                )
            ],
        ),
        
        # Text description of the project
        
        html.Div(
            [
                dcc.Markdown(
                    ''' 
World of Warcraft is a massively multiplayer online video game released on November 23, 2004. Before this era, MMORPG’s catered to a small segment of video gamers. But with the massive success of WoW, various video game companies decided to invest resources into developing large-scale titles. Traditional video games provided movie-like experiences, where you would follow a single protagonist on an adventure. However, WoW does not follow a single protagonist, but all of the users playing the video game. Not only was the main objective different from single-player games, but the pricing model. Traditional games followed a single upfront fee. WoW has a monthly subscription to play the game in addition to the upfront fee.   

In regards to user subscriptions, predicting customer churn is a valuable tool in examining user playing behavior to obtain insight into churned and subscribed users.  In this study, we will predict whether a user will unsubscribe to the service after six months of playtime.

Additionally, a user can provide their inputs into the tool for churn predictions and recommendations for their playing behavior to reduce the risk of churn. 

In terms of performance: RF > KNN > LR > SVM, more information on dataset details and assumptions are in the project report and notebook [1](https://github.com/SulmanK/Customer-Churn-in-World-of-Warcraft/blob/master/Customer%20Churn%20in%20World%20of%20Warcraft_Report.pdf), [2](https://github.com/SulmanK/Customer-Churn-in-World-of-Warcraft/blob/master/Customer%20Churn%20in%20World%20of%20Warcraft.ipynb), respectively.
                    '''
                )
            ], style = {'padding': '2rem 4rem 2rem 4rem',
                        'border-top': '20px solid #424B6B',
                        'border-bottom': '20px solid #424B6B',
                        'fontSize' : 28, 'font-family': "Myriad Pro"}
        ),
        

                
        # Setup three column layout
        
        html.Div(
            [
                # Insert Header for Classsification Algorithm

                html.H4('Classification Algorithm'),

                # Insert dropdown menu for Classification Algorithms

                dcc.Dropdown(
                    id = 'Classification_Algorithms',
                    options = [
                        {'label': 'Logistic Regression', 'value': 'LR'},
                        {'label': 'Support Vector Machines (SVM)', 'value': 'SVM'},
                        {'label': 'K-Nearest Neighbors (KNN)', 'value': 'KNN'},
                        {'label': 'Random Foest (RF)', 'value': 'RF'},
                    ],
                    placeholder = "Select an algorithm",
                    value = 'MTL', style = {'width': '80%', 'display': 'inline-block', 'color': '#000000'},
                ), 

        # Insert inputs header and fields         
                # Max Level
                html.H4('Max Level [1 - 80]'),
                dcc.Input(id = 'ML', value = '1',
                          type = 'number', min = 1,
                          max = 80, style = {'background-color': "#282b38" ,
                                             'border-color': "#a5b1cd",
                                             'color': "#a5b1cd"}
                         ), 

                html.Div(id = 'ML-div'),

                # Insert APH

                html.H4('Average Playing Hours per day [0 - 24.0]'),
                dcc.Input(id = 'APH', value = '0',
                          type = 'number', min = 0,
                          max = 24, step = 0.1, 
                          style = {'background-color': "#282b38" ,
                                   'border-color': "#a5b1cd",
                                   'color': "#a5b1cd" }
                         ),

                html.Div(id = 'APH-div'),

                # Insert APD

                html.H4('Average Playing Density per month [0 - 1.0]'),
                dcc.Input(id = 'APD', value = '0',
                          type = 'number',
                          min = 0, max = 1,
                          step= 0.1, style = {'background-color': "#282b38" ,
                                              'border-color': "#a5b1cd",
                                              'color': "#a5b1cd" }
                         ),

                html.Div(id = 'APD-div'), 

                # Insert Guild

                html.H4('Guild [0 or 1]'),
                dcc.Input(id = 'G', value = '0',
                          type = 'number', min = 0,
                          max = 1, step = 1, 
                          style = {'background-color': "#282b38" ,
                                   'border-color': "#a5b1cd",
                                   'color': "#a5b1cd"}
                         ),

                html.Div(id = 'G-div'), 

                # Insert LED indicator red: churn, green: subscribed

                html.Div(
                    [
                        html.Div(id = 'my-indicator'),
                    ], style = {'display' : 'flex',
                                'justifyContent' : 'center',
                                'padding-top' : '4rem'}
                )
            ], className='three columns',

            style = {
                'min-width': '24.5%',
                'max-height': 'calc(100vh - 85px)',
                'overflow-y': 'auto',
                'overflow-x': 'hidden', 
                'box-sizing': 'border-box',
                'padding-left': '2rem',
                'font-family': 'Helvetica',}
        ),

        # Insert Comparison Chart and Recommendation Section
        
        html.Div(
            [
                # Insert Comparison chart (user input and average subscriber)

                html.Div(
                    [
                        dcc.Graph(id = 'Recommendation'), 
                    ], style = {'width': '90%'}
                ),

                # Insert Recommendation header and markdown for tips
                
                html.Div(
                    [
                        # Insert Header
                        
                        html.H4('Recommendations'), 

                        # Insert Tips
                        
                        dcc.Markdown(id = 'Tips')

                    ],  style={'font-family': 'Myriad Pro',
                               'fontSize': 26,
                               'padding-left': '4rem'}
                ), 
            ], className = 'six columns', style = {'box-sizing': 'border-box',
                                                   'border-left': '20px solid #424B6B',
                                                   'border-right': '20px solid #424B6B',
                                                   #'max-height': 'calc(100vh - 85px)',
                                                   #'height': 'calc(85vh - 90px)'
                                                   'height': 'calc(100vh - 120px)'
                                                  }
        ),

        # Insert Classification Metrics section
       
        html.Div(
            [
                html.Div(
                    [
                        # Insert Header

                        html.H4('Classification Metric'),

                        # Insert Dropdown for various classification metrics

                        dcc.Dropdown(
                            id = 'Graphs',
                            options = [
                                {'label': 'Receiver Operating Characteristic', 'value': 'ROC'},
                                {'label': 'Precision - Recall', 'value': 'PR'},
                                {'label': 'Confusion Matrix', 'value': 'CM'},
                            ],
                            placeholder = "Select a plot",
                            value = 'G12', style = {'width': '70%',
                                                    'display': 'inline-block',
                                                    'color' : '#000000'}
                        ),
                        
                        # Insert ROC/PR/CM figure
                        
                        html.Div(
                            [
                                dcc.Graph(id = 'ROC/PR'), 
                            ],  style = { 'width': '100%',
                                         'padding-top': '4rem',
                                         'font-family': 'Helvetica'}
                        ),
            ],          style = {
                        'min-width': '60.5%',
                        # Remove possibility to select the text for better UX
                        'user-select': 'none',
                        '-moz-user-select': 'none',
                        '-webkit-user-select': 'none',
                        '-ms-user-select': 'none', 
                    },
                ),
            ],  className='three columns',      
    ),
        
        # Insert Churn Prediction callback, it's invisible, the LED indicator is getting the value from this
       
        html.Div(className = 'row', 
                 children = [
                     html.Div(
                         [
                             html.Pre(id = 'Churn-Prediction')
                         ], style = {'fontSize': 0 }
                     ), 
                 ],
                ),
    ], style = {'color' : "#a5b1cd", 'backgroundColor' : "#282b38" }
)


#------------------- Function Callbacks

# ROC/PR/CM Graph Callback
@app.callback(
    Output('ROC/PR', 'figure'),
    
    # Classification Algorithm
    [Input(component_id = 'Classification_Algorithms', component_property = 'value'),
    
    # Graphs
    Input(component_id = 'Graphs', component_property = 'value'),]
)

    
def callback_rocprcm(Classification_Algorithms, Graphs):
    """Returns 3D Plots of the Clusters based on which method was selected"""
    # Logistic Regression
    if Classification_Algorithms == 'LR' :

        # Plots of ROC / PR / CM
        if Graphs == 'ROC':
            
            return LR_roc
        
        elif Graphs == 'PR':

            return LR_pr
        
        elif Graphs == 'CM':
            
            return LR_cm
        
        else:
         
            return 'Select a plot!'

    # SVM    
    elif Classification_Algorithms == 'SVM' :

        # Plots of ROC / PR / CM
        if Graphs == 'ROC':
            
            return SVM_roc
        
        elif Graphs == 'PR':

            return SVM_pr
        
        elif Graphs == 'CM':
            
            return SVM_cm
        
        else:
         
            return 'Select a plot!'
    
    # KNN
    elif Classification_Algorithms == 'KNN' :

        # Plots of ROC / PR / CM
        if Graphs == 'ROC':
            
            return KNN_roc
        
        elif Graphs == 'PR':

            return KNN_pr
        
        elif Graphs == 'CM':
            
            return KNN_cm        
        
        else:
         
            return 'Select a plot!'
    
    # RF
    elif Classification_Algorithms == 'RF' :

        # Plots of ROC / PR / CM
        if Graphs == 'ROC':
            
            return RF_roc
        
        elif Graphs == 'PR':

            return RF_pr
        
        elif Graphs == 'CM':
            
            return RF_cm
        
        else:
         
            return 'Select a plot!'
        
    else:
     
        return 'Choose between LR, SVM, KNN, or RF'


# Churn Prediction using the user inputs
@app.callback(
    Output(component_id = 'Churn-Prediction', component_property = 'children'),   
    
    # Classification Algorithm
    [Input(component_id = 'Classification_Algorithms', component_property = 'value'),
    
    # Max Level
    Input(component_id = 'ML', component_property = 'value'),

    # Average Playing Hours
    Input(component_id = 'APH', component_property = 'value'),

    # Average Playing Density
    Input(component_id = 'APD', component_property = 'value'),

    # Guild
    Input(component_id = 'G', component_property = 'value')]
)    
    

def callback_Churn_Prediction(Classification_Algorithms, ML, APH, APD, G):
    """Function for Callback Cluster Detection to identify if someone is a hacker"""
    
    # Take the input data and scale it
    input_array = np.array([ML, APH, APD, G])
    
    data =  {'guild': input_array[3],
             'max_level': input_array[0],
             'Average_Hour': input_array[1],
             'Average_Playing_density': input_array[2],
            }

    predict_scaled = pd.DataFrame(data = data, index=[0])
    predict_scaled = ct.transform(predict_scaled)

    
    # Check the callbacks for whichever clustering algorithm was selected and then predict for hackers
    ## Logistic Regression
    if Classification_Algorithms == 'LR':
        
        predict_labels = clf_LR.predict(predict_scaled)
        
        if predict_labels[0] == 0:
        
            Decision = 'Churned'
        
        else:
        
            Decision = 'Subscribed'
        
        return Decision
    
    ## SVM
    elif Classification_Algorithms == 'SVM':
        
        predict_labels = clf_SVM.predict(predict_scaled)
        
        if predict_labels[0] == 0:
        
            Decision = 'Churned'
        
        else:
        
            Decision = 'Subscribed'
        
        return Decision
    
    # KNN
    elif Classification_Algorithms == 'KNN':
        
        predict_labels = clf_KNN.predict(predict_scaled)
        
        if predict_labels[0] == 0:
        
            Decision = 'Churned'
        
        else:
        
            Decision = 'Subscribed'
        
        return Decision
    
    ## RF
    elif Classification_Algorithms == 'RF':
        
        predict_labels = clf_RF.predict(predict_scaled)
        
        if predict_labels[0] == 0:
        
            Decision = 'Churned'
        
        else:
        
            Decision = 'Subscribed'
            
        return Decision
    
    else:
        
        return 'Choose between LR, SVM, KNN, or RF.'

    
# Churn Indicator Callback
@app.callback(
    Output('my-indicator', 'children'),
    
    # Output from previous callback
    [Input(component_id = 'Churn-Prediction', component_property = 'children')]
)

def update_output(Churn_Prediction):
    """Updates LED indicator based on if a user is churned or subscribed"""
    
    # Churn option
    if Churn_Prediction == 'Churned':
        return html.Div(
            [ 
                daq.Indicator(
                    color = 'red',
                    label = {'label': 'Churned',
                             'style' : {'fontSize': 40}}, 
                    labelPosition = 'bottom',
                    value = True,
                    size = 100
                )
            ]
        )
    
    else: 
        return html.Div(
            [
                daq.Indicator(
                    color = 'green',
                    label = {'label': 'Subscribed',
                             'style' : {'fontSize': 40}}, 
                    labelPosition = 'bottom',
                    value = True,
                    size = 100,
                )
            ]
        )

# Comparison Callback
@app.callback(
    Output('Recommendation', 'figure'),
    
    # Max Level
    [Input(component_id = 'ML', component_property = 'value'),

    # Average Playing Hours
    Input(component_id = 'APH', component_property = 'value'),

    # Average Playing Density
    Input(component_id = 'APD', component_property = 'value'),

    # Guild
    Input(component_id = 'G', component_property = 'value')]
)

    
def callback_recomendation_plot(ML, APH, APD, G):
    """Returns a comparison between the inputted user and average subscriber using a barplot"""
    input_array = np.array([ML, APH, APD, G])
    return comparison_plot(input_array)
    

# Recomenndation Tips Callback
@app.callback(
    Output('Tips', component_property = 'children'),
    
    # Max Level
    [Input(component_id = 'ML', component_property = 'value'),

    # Average Playing Hours
    Input(component_id = 'APH', component_property = 'value'),

    # Average Playing Density
    Input(component_id = 'APD', component_property = 'value'),

    # Guild
    Input(component_id = 'G', component_property = 'value')]
)

    
def callback_recomendation_tips(ML, APH, APD, G):
    """Returns statements depending on the inputted user value compared to the average subscriber values"""
    # Set all inputs as integers
    ML = int(ML)
    APH = int(APH)
    APD = int(APD)
    G = int(G)
    
    # Create arrays for inputs and subscribers
    input_array = np.array([ML, APH, APD, G])
    subscriber_array = [46, 1.7428953, 0.21423, 1]
    
    # Create suggestions lists for markdown
    suggestions = []
    for x, y, name, tips in zip(input_array, subscriber_array,
                                ['max level', 'average playing hours',
                                 'average playing Density', 'guild'],
                                ['Increase your level to reduce the risk of churn.',
                                 'Increase your playing time to reduce the risk of churn.', 
                                 'Increase your number of unique days played to reduce the risk of churn.',
                                 'Join a guild to engage in a social community to reduce the risk of churn.']):
        # If the user input is less than subscriber, insert a statement from above depending on what feature
        if x < y:
            tmp = "* " + str(tips)
            suggestions.append(tmp)
            
    return suggestions
        
        
        
# CSS styleheets used in the dashboard      
external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})    

if __name__ == '__main__':
    app.run_server(debug = True)