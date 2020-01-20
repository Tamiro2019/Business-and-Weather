import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, date

from scipy import stats

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as dates

from plotly.tools import mpl_to_plotly
from plotly.subplots import make_subplots

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


### Load Data ###
yb50 = pd.read_csv('yb50.csv')
yb50 = yb50.loc[:, ~yb50.columns.str.contains('^Unnamed')]
yb50.index.name = 'b_id'
#print(yb50.head())

ts50 = pd.read_csv('ts50.csv')
ts50.set_index('Date',inplace=True)
ts50.index = pd.to_datetime(ts50.index)
#print(ts50.head())

# Cut down long names
replace_dict = dict({'Giant Eagle Market District':'Giant Eagle Market Dist','Winghart\'s Burger & Whiskey Bar':'Winghart\'s',
                     'Carnegie Mellon University':'Carnegie Mellon','iLoveKickboxing- Pittsburgh':'iLoveKickboxing',
                    'Gaucho Parrilla Argentina':'Gaucho','Pittsburgh International Airport':'International Airport'})

yb50['name'] = yb50['name'].replace(replace_dict)      

# Useful dictionary for graphs: 
Wdict = dict({'T': 'Temperature (\u00b0F)','P': 'Precipitation (in)','WS':'Wind Speed (mi/h)'})


## Graph 1 Generator ##
# Make time-trace/scatter plot

def timetrace(data_frame,date_i = dt.date(2016,1,1),date_f= dt.date(2018,1,1), fq = 30, bidx = 2, WQ = 'T'):
    '''
    Returns mpl plot for the chosen business, time frame, and sampling rate.
    - The first subplot shows check-in count vs time.
    - The second subplot shows the chosen weather indicator vs time. 
    
    Inputs
        - data_frame : time-indexed dataframe for business and weather.
        - date_i : initial date / lower bound to plot
        - date_f : final date / upper bound to plot
        - fq : sampling frequency (in days) for time domain
        - bidx : index of the business of interest
        - WQ : weather quantity of interest, can be 'T' (temperature), 'P' (precipitation), or 'WS' (wind speed)
        
    '''
    
    df = data_frame
    
    Freq = str(fq)+'D' # make frequency string
    
    # Figure
    fig0 , (ax1,ax2) = plt.subplots(2,1,figsize=(4.0,3.0),sharex=True) #figsize=(5.5,4)

    # pick up data
    xdata = pd.to_datetime(df.loc[date_i:date_f].resample(Freq).sum().index)
    
    ydata_B = df.loc[date_i:date_f].resample(Freq).sum()[str(bidx)]
    ydata_W = df.loc[date_i:date_f].resample(Freq).mean()[WQ]
    
    # make plot
    ax1.plot(xdata,ydata_B, color='tab:blue')
    ax2.plot(xdata,ydata_W, color='tab:green')
    
    # make plot nice
    ax1.xaxis.set_visible(False)
    ax1.xaxis.set_ticklabels([])

    ax1.set_title('Check-in Count',fontsize= '15',fontweight="bold",y=0.86)
    ax2.set_title(Wdict[WQ],fontsize= '15',fontweight="bold",y=0.86)

    ax1.grid(b=True)
    ax2.grid(b=True)
       
    ax1.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    
    fig0.tight_layout(pad=3.0)
    
    plotly_obj = mpl_to_plotly(fig0)
    plotly_obj.data[0]['x'] = [ dates.num2date(x, tz=None).date() for x in list(plotly_obj.data[0]['x'])]
    plotly_obj.data[1]['x'] = [ dates.num2date(x, tz=None).date() for x in list(plotly_obj.data[1]['x'])]
    plotly_obj.update_layout(margin = dict(t=25,b=10),autosize= True,xaxis=dict(type='date',showticklabels=False,range=None),xaxis2=dict(type='date',range=None,tickangle = 35),template="simple_white") 
   
    # return plotly subplots
    return plotly_obj 


## Graph 2 Generator ##
# Make scatter plot

def BW_scatter(data_frame,date_i = dt.date(2016,1,1),date_f= dt.date(2018,1,1), fq = 30, bidx = 0, WQ = 'T'):
    '''
    Returns mpl scatter plot of check-in count vs weather indicator for a chosen business, time frame, and sampling rate.
    
    Inputs
        - data_frame : time-indexed dataframe for business and weather.
        - date_i : initial date / lower bound to plot
        - date_f : final date / upper bound to plot
        - fq : sampling frequency (in days) for time domain
        - bidx : index of the business of interest
        - WQ : weather quantity of interest, can be 'T' (temperature), 'P' (precipitation), or 'WS' (wind speed)
    '''
    
    df = data_frame
    
    Freq = str(fq)+'D' # make frequency string
    
    # Figure
    fig1 = plt.figure(figsize=(3.8,3.0)) #,figsize=(5.5,4)

    ydata_B = df.loc[date_i:date_f].resample(Freq).sum()[str(bidx)]
    ydata_W = df.loc[date_i:date_f].resample(Freq).mean()[WQ]
    
    # linear regression 
    m, b, r_value, p_value, std_err = stats.linregress(ydata_W,ydata_B)

    # make scatter plot
    plt.scatter(ydata_W,ydata_B)
    plt.plot(ydata_W,m*ydata_W+b, '-r')
    
    # make plot nice
    plt.xlabel(Wdict[WQ],fontsize= '13',fontweight="bold")
    plt.ylabel('Check-in Count',fontsize= '13',fontweight="bold")
    
    plt.title('$'+'\quad'*10 +' R^2 =' + str(np.round(r_value**2,2))+'$');

    plt.grid(b=True)
    plt.gca().tick_params(axis="x", labelsize=12)
    plt.gca().tick_params(axis="y", labelsize=12)
    
    plotly_obj = mpl_to_plotly(fig1).update_layout(margin = dict(t=25,b=10),template="simple_white",autosize=True) #
    # return plotly scatter plot
    return plotly_obj 


## Graph 3 Generator ##

# Make Weather Sensitivity Bar Chart

def R2_chart(data_frames,date_i = dt.date(2016,1,1),date_f= dt.date(2018,1,1), fq = 30, bidx = 0, WQ = 'T'):
    '''
    Returns bar plot of R^2 values across different businesses. 
    
    Inputs
        - data_frame : time-indexed dataframe for business and weather.
        - date_i : initial date / lower bound to plot
        - date_f : final date / upper bound to plot
        - fq : sampling frequency (in days) for time domain
        - bidx : index of the business of interest
        - WQ : weather quantity of interest, can be 'T' (temperature), 'P' (precipitation), or 'WS' (wind speed)
    '''
    
    [yb50x,ts50x] = data_frames
    
    Freq = str(fq)+'D' # make frequency string
    
    ydata_W = ts50x.loc[date_i:date_f].resample(Freq).mean()[WQ]
    ydata_B = ts50x.loc[date_i:date_f].resample(Freq).sum() #[str(i)]
    
    # Loop over all businesses, get the regression R^2 and store the business name and R^2 value in the lists:
    Bname = []
    R2 = []

    for i in range(50):
        m, b, r_value, p_value, std_err = stats.linregress(ydata_W, ydata_B[str(i)])
        R2.append(r_value*r_value)
        Bname.append(yb50x.loc[i,'name'] + '  ')
        
    # Get index array which sorts the lists according to R^2
    idx = np.argsort(R2)[::-1]
    
    # Set the colors, and choose a special color for the business selected in the timetrace and scatter plots.
    colors = ['royalblue',]*len(np.array(R2)[idx]) #'cornflowerblue'
    colors[list(idx).index(bidx)] = 'darkorange'
    
    # return dictionary for plotly graph
    return [{'x': np.array(R2)[idx],'y':np.array(Bname)[idx], 'type': 'bar','orientation':'h', 'marker': { 'color':colors }}] 


## Some useful object to use in dash below ##

# get business name series for dropdown list
business_names = yb50['name']

# define function to get correct marks for the slider object
def get_marks(f):
    
    sampling = '182D' # sampling for slider object
    fsamp = f.resample(sampling).sum()
    dates = {}
    
    for z in fsamp.index:
        dates[f.index.get_loc(z)] = str(z.strftime('%Y-%m')) 
        
    return dates


### Dash it out ###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# setup dash object and link server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# setup mathjax to interpret Tex 
mathjax = '//cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS-MML_HTMLorMML' #default,TeX-MML-AM_SVG'#_CHTML'
app.scripts.append_script({ 'external_url' : mathjax })

## Design page layout ##    
app.layout = html.Div([
  
    # Page 1
    html.Div([
        
        #Header
        html.Div([

            html.Div([      
                html.H5('The Effect of Weather on Yelp Engagement',style=dict(fontSize = '26')),
                html.H6('An Exploratory Application on 50 Top Businesses in Pittsburgh', 
                        style=dict(color='#7F90AC',fontSize = '20')),
                ], className = "nine columns padded" ),

            html.Div([            
                html.H6([html.Span('Tamiro Villazon',style=dict(fontSize = '20'))]),
                html.H6('Data Physicist',style=dict(opacity=0.5,fontSize = '20'))
            ], className = "three columns gs-header gs-accent-header padded", style=dict(float='right') ),

        ], className = "row gs-header gs-text-header"),
        
         html.Br([]),
        
        #Paragraphs
        html.Div([
            html.P("Data science is gaining a prominent role in the world of business and investment. \
                    Data-driven, evidence-based analysis has become crucial \
                    for effective decision-making in uncertain and ever-evolving markets.",
                   style = dict(fontSize = '16') ),
            
            html.P("Particularly intriguing is the question of how we can use data to understand \
            consumer behavior. Even though people are fantastically complicated and of diverse tastes \
            and preferences, patterns begin to emerge when viewed through a proper lens.", 
                   style = dict(fontSize = '16') ),
            
            html.Div([
                html.H6('The App',
                        style=dict(opacity=0.80,backgroundColor= 'maroon',color='white',fontSize = '20'),
                        className = "two columns")
            ], className = "row",style={'padding-right':'10%'}),
            
            html.Br([]),
            
            html.P("To study consumer patterns, I built a simple app which compares business engagement \
            (as measured by check-in counts on Yelp) to different weather indicators (temperature, \
            precipitation, and wind-speed) for 50 top businesses in Pittburgh, Pennsylvania.",
                   style = dict(fontSize = '16') ),
            
            html.Div([
                html.H6('The Data',
                        style=dict(opacity=0.80,backgroundColor= 'maroon',color='white',fontSize = '20'),
                        className = "two columns")
            ], className = "row", style={'padding-right':'10%'}),
            
            html.Br([]),
            
            html.P(["Business and check-in data was retrieved from the ", 
                    html.A("Yelp Open Dataset", 
                           href='https://www.yelp.com/dataset', 
                           target="_blank") ,
                    ". Note: the data is used here for personal and educational purposes."], 
                    style = dict(fontSize = '16') ),
            
            html.P(["Weather data from a land-based station in the Pittsburgh International Airport\
            was obtained from the ",
            html.A("National Centers for Environmental Information", 
                   href='https://www.ncdc.noaa.gov/data-access/',
                   target="_blank"), "."], 
                   style = dict(fontSize = '16') ),
            
            html.P(["A portion of these data sets were cleaned and combined into a single\
            dataframe for this project. "], 
                   style = dict(fontSize = '16')),
            
            html.Div([
                html.H6('How It Works',
                        style=dict(opacity=0.80,backgroundColor= 'maroon',color='white',fontSize = '20'),
                        className = "three columns")
            ], className = "row",style={'padding-right':'20%'}),
            
            html.Br([]),
            
            html.P(["The app is intuitive. The first graph (left) visualizes the check-in count\
            and weather indicator time series, between January 2014 and December 2018. The second graph (right)\
            shows a scatter plot of this data (check-in count vs weather indicator), together with a linear\
            regression to identify any trends."], style = dict(fontSize = '16')),
            
            html.P(["It is also simple to use.\
            Use the drop menus to select  any business and weather indicator of your choice.\
            Below you will find a slider which narrows/expands the time window, if you wish to focus\
            on a particular time frame. You can also set the sampling frequency to any number of days;\
            the sampling frequency determines the smallest time division within which the weather data is\
            averaged and the check-in data is aggregated. For example, if sampling frequency is set\
            to 7 days, then the check-in time series data is grouped in batches of 7 days, and so each\
            data point in the graphs becomes the total number of check-ins within a 7 day period.\
            You may find that the correlation between engagement and weather varies significantly with \
            time window and sampling rate for some businesses."],
                   style = dict(fontSize = '16')),
            
            html.Div([
                html.H6('How It Really Works',
                        style=dict(opacity=0.80,backgroundColor= 'maroon',color='white',fontSize = '20'),
                        className = "four columns")
            ], className = "row",style={'padding-right':'15%'}),
            
            html.Br([]),
            
            html.P(["For a closer look at the code used to generate the app and other documentation\
            feel free to check out my ",html.A("Github repository", 
                   href='https://github.com/Tamiro2019/Business-and-Weather',
                   target="_blank")," for this project."], style = dict(fontSize = '16')),

        
        ## Row 2.5 : Hidden and used for debugging
        html.Div([
            dcc.Textarea(id='my-id', value='initial value'),
        ],style={'display': 'none'}),
        
        ## Begin app ##
        html.Div([
            html.H5(['Interactive App'], className = "gs-header gs-table-header padded",
                    style = {'textAlign': 'center','fontWeight':'bold'}),
            # Dropdowns to select business and weather indicator
            html.Div([
                html.Div([
                    html.H6(["Select Business"], className = "gs-header gs-table-header padded",
                            style=dict(fontSize = '20')),
                    dcc.Dropdown(
                        id = 'b-drop',
                        options=[{'label': i, 'value': i} for i in business_names],
                        value=business_names[0])
                ], className = "six columns",style={'padding-left':'3%', 'padding-right':'1%'} ), 



                 html.Div([
                    html.H6(["Select Weather Indicator"], className = "gs-header gs-table-header padded",
                            style=dict(fontSize = '20')),
                    dcc.Dropdown(
                        id = 'w-drop',
                        options=[{'label': 'Temperature (\u00b0F)', 'value': 'T'},{'label': 'Precipitation (in)', 'value': 'P'},{'label': 'Wind Speed (mi/h)', 'value': 'WS'}],
                        value='T') 

                 ], className = "six columns",style={'padding-left':'5%'}),  

            ], className = "row padded" ),

            html.Br([]),
            html.Div([
                    dcc.Textarea(id='b_des', value='Business Description',rows =1,readOnly='readOnly',style={'height': '80%','width': '100%','fontSize':'14'}),
                ],style={'padding-left':'4%','padding-right':'4%'}),
    
            html.Br([]),

            # Timetrace and Scatter plots
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='timetrace',
                        figure = timetrace(ts50),
                        style = dict({'width': '100%'}))
                ], className = "six columns"), 

                html.Div([
                    dcc.Graph(
                        id='scatter',
                        figure= BW_scatter(ts50),
                    style = dict({'width': '100%'}))
                ], className = "six columns"),

            dcc.Loading(id="loading", children=[html.Div(id="output-1")], type="default"),

            ], className = "row" ),

            html.Br([]),
            html.Br([]),
            html.Br([]),

            # Date range slider
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id='range-slider',
                        updatemode='mouseup',
                        min=0,
                        max=len(ts50.index) - 1,
                        count=1,
                        #step=3,
                        value=[1*(len(ts50.index) - 1)//5, 4*(len(ts50.index) - 1)//5],
                        marks=get_marks(ts50),
                    )
                ],className = "ten columns", style={'width': '96%', 'padding-left':'4%', 'padding-right':'1%'}) 

            ], className = "row" ),

            html.Br([]),
            html.Br([]),
            html.Br([]),

            # Input box for date sampling frequency
            html.Div([
                html.Div([
                    html.H5(["Sampling Frequency (in days) = "], className = "gs-header seven columns gs-table-header padded", style={'text-align': 'center'})
                ]), 
                html.Div([
                    dcc.Input(
                        id = 'sampling',
                        placeholder='Frequency',
                        type='number',
                        value=30,
                        debounce = True,
                        min = 1,
                        max = 365
                        )  
                ]) 
            ], className = "row", style=dict({'padding-left':'16%','verticalAlign': 'middle'})), 

            html.Br([]),
            html.Br([]),
        
        ], className = "row",style={"border":"2px black solid"} ), 
        # End app
        
        html.Br([]),
        html.Br([]),
        
        # Insight Paragraphs
        html.Div([
                html.H6('Insights',style=dict(opacity=0.80,backgroundColor= 'maroon',color='white',fontSize = '20'),
                        className = "two columns")
            ], className = "row",style={'padding-right':'10%'}),    
            
        html.Br([]),
            
        html.P("The figure below summarizes the correlation coefficients (R-squared) of all the businesses.\
            These coefficients track the correlation between check-in count and weather over the specified\
            period of time and sampling. They serve as a measure of the sensitivity of a business to weather\
            conditions. There is a broad range of correlation values, which indicates that there are businesses\
            whose engagement is highly dependent on weather, while others are much less impacted."
            , style = dict(fontSize = '16') ),
            
        html.P("This information can help businesses and investors identify trends and make strategic decisions.\
            For example, leaders of highly weather-dependent businesses can manage resource allocation based on\
            expected demand. Alternatively, weather correlations can be provide supplementary information on \
            business performance. For instance, if a business does not expect to be impacted by weather and \
            the data reveals otherwise, it may point to a problem that needs attention. \
            Businesses with negligible weather correlations may also be attractive to investors who may\
            not want seasonal variations in portfolio performance. Of course in any decision, many different \
            variables must be considered; sensitivity to weather conditions could be an important\
            piece of the puzzle.", 
                style = dict(fontSize = '16') )
        ], className = "row padded" ),    
        
        html.Br([]),
        html.Br([]),
        # Row 7: Plot bar plot for R^2 across businesses
        html.Div([
            html.H5(['Business R',
                     html.Sup([2]), 
                     ' Distribution'], className = "gs-header gs-table-header padded", style = {'textAlign': 'center','fontWeight':'bold'}),
            dcc.Graph(id='R-squared',
                            figure={
                                'data': R2_chart([yb50,ts50]),
                                'layout': {
                                    "height": 1000,  # px
                                    "xaxis":dict(side= 'top'), 
                                    "yaxis":dict(autorange="reversed"),
                                    "margin": dict(t=20, b= 20, l=270),
                                }
                            }           
                      
                   )], className = "row",style={"border":"2px black solid"} ) 
        
    ], className = "page" )
    
])

## Structure Callbacks (for interactive use of app) ##    

# Update all plots upon interaction
@app.callback(
    [Output(component_id='my-id', component_property='value'),
    Output(component_id='timetrace', component_property='figure'),
    Output(component_id='scatter', component_property='figure'),
    Output(component_id='R-squared', component_property='figure')],
    [Input(component_id='b-drop', component_property='value'),
    Input(component_id='w-drop', component_property='value'),
    Input(component_id='range-slider', component_property='value'),
    Input(component_id='sampling', component_property='value')])
def update_plots(b_val,w_val,rng_vals,s_val):
    
    # Get initial and final times from input rng_vals
    t1 = (ts50.index[rng_vals[0]].to_pydatetime())
    t2 = (ts50.index[rng_vals[1]].to_pydatetime())
    
    # Get business id from b_val
    b_idx = yb50[yb50['name'] == b_val].index[0]
    
    # Get new figure objects
    fig_t = timetrace(ts50,date_i=t1,date_f=t2, fq = s_val, bidx = b_idx, WQ = w_val)
    fig_s = BW_scatter(ts50,date_i=t1,date_f=t2, fq = s_val, bidx = b_idx, WQ = w_val)
    fig_r = {
                'data': R2_chart([yb50,ts50], date_i=t1,date_f=t2, fq = s_val, bidx = b_idx, WQ = w_val),
                'layout': {
                    "height": 1000, 
                    "xaxis":dict(mirror = "allticks", side= 'top',automargin=True), 
                    "yaxis":dict(autorange="reversed"),
                    "margin": dict(t=20, b= 20, l=270)
                }
            } 
    
    # Return objects to callback outputs [note: first object returns info to the debug div - Row 2.5 above]
    return 'You\'ve entered {},{},{},{} and'.format(b_val,w_val,rng_vals,s_val), fig_t, fig_s, fig_r

# Show loading spinner while waiting for plots to update
@app.callback(Output('output-1' , "children"),
    [Input(component_id='b-drop', component_property='value'),
    Input(component_id='w-drop', component_property='value'),
    Input(component_id='range-slider', component_property='value'),
    Input(component_id='sampling', component_property='value')])
def input_triggers_spinner(value):
    time.sleep(1)
    return value          

# Update business information 
@app.callback(Output('b_des' , "value"),
    [Input(component_id='b-drop', component_property='value')])
def update_business(b_value):
    b_idx = yb50[yb50['name'] == b_value].index[0]
    output_val = b_value + ':\n - ' + yb50[yb50['name'] == b_value].loc[b_idx,'categories'] + '.'
    return output_val    

# add external css and js templates
external_css = [ "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
        "//fonts.googleapis.com/css?family=Raleway:400,300,600",
        "https://codepen.io/plotly/pen/KmyPZr.css",
        "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css: 
    app.css.append_css({ "external_url": css })
    
external_js = [ "https://code.jquery.com/jquery-3.2.1.min.js",
        "https://codepen.io/plotly/pen/KmyPZr.js" ]
    
for js in external_js: 
    app.scripts.append_script({ "external_url": js })

# configure css and scripts for deployment
app.css.config.serve_locally = False
app.scripts.config.serve_locally = False

if __name__ == '__main__':
    app.run_server(debug=False)
