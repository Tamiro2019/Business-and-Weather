import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, date

import gc
import json
from scipy import stats

import matplotlib as mpl
from matplotlib import pyplot as plt
#%matplotlib notebook

from plotly.tools import mpl_to_plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

### Load Data ###
df100aux = pd.read_csv('Yelp100')
#print(df100.head())

ci100 = pd.read_csv('Yelp100_ci_count')
#print(ci100.head())

dfw = pd.read_csv('Weather')
#print(dfw.head())

# Frame as datetime data frames
ci100['Time'] = ci100.apply(lambda row: dt.datetime(row['year'], row['month'], row['day'],row['hour']), axis=1)
ci100.drop(['year','month','day','hour'],axis=1,inplace=True)
dfw['Time']= dfw.drop(['T','P','WS'],axis=1).apply(lambda row: dt.datetime(row['year'], row['month'], row['day'],row['hour']), axis=1)
dfw.drop(['year','month','day','hour'],axis=1,inplace=True)

ci100 = ci100.set_index('Time')
ci100 = ci100.resample('1D').sum()
#print(ci100.head())
dfw = dfw.set_index('Time')
dfw['P'] = (1.0/100)*dfw['P'].apply(round,-2) # convert once to millimeters
dfw = dfw.resample('1D').mean()
#print(dfw.head())

df100 = df100aux['name'].to_frame()
#df100.head()
#ci100.head()

### cut down long names ###

print(df100.head())
for bname in df100['name']:
    
    if len(bname) > 32:
        print(bname)
        bname_new = bname.split(' ')[0]+' '+bname.split(' ')[1]
        print(bname_new)
        df100['name'] = df100['name'].replace(bname,bname_new)
        
        #df100[df100['name']==bname] = bname_new
        

### Graph Generators ### 

Wdict = dict({'T': 'Temperature (\u00b0F)','P': 'Precipitation (in)','WS':'Wind Speed (mi/h)'})

# Make time-trace/scatter plot

def timetrace(data_frames,date_i = dt.datetime(2016,1,1),date_f= dt.datetime(2018,1,1), fq = 30, bidx = 2, WQ = 'T'):
    
    Freq = str(fq)+'D'
    
    [ci100, dfw] = data_frames
    
    # Figure
    fig0 , (ax1,ax2) = plt.subplots(2,1,figsize=(5.5,4),sharex=True)

    xdata_B = pd.to_datetime(ci100.loc[date_i:date_f].resample(Freq).sum().index)#.values #.to_pydatetime()#.values
    ydata_B = ci100.loc[date_i:date_f].resample(Freq).sum()[str(bidx)]
    
    ax1.plot(xdata_B,ydata_B);
    ax1.xaxis.set_visible(False)
    ax1.xaxis.set_ticklabels([])
    
    xdata_T = pd.to_datetime(dfw.loc[date_i:date_f].resample(Freq).mean().index)#.values
    ydata_T = dfw.loc[date_i:date_f].resample(Freq).mean()[WQ]
    ax2.plot(xdata_T,ydata_T,color='tab:green');

    ax1.set_title('Check-in Count',fontsize= '15',fontweight="bold",y=0.86)
    ax2.set_title(Wdict[WQ],fontsize= '15',fontweight="bold",y=0.86)

    ax1.grid(b=True)
    ax2.grid(b=True)
    
    ax1.yaxis.set_ticks(np.arange(round(ydata_B.min(),-1),round(ydata_B.max(),-1)+ 10,5))
    ax2.yaxis.set_ticks(np.arange(round(ydata_T.min(),-1),round(ydata_T.max(),-1)+20,20))
    ax1.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    
    fig0.tight_layout(pad=3.0)
    
    return fig0

# Make Scatter Plot

def BW_scatter(data_frames,date_i = dt.datetime(2016,1,1),date_f= dt.datetime(2018,1,1), fq = 30, bidx = 2, WQ = 'T'):
    
    Freq = str(fq)+'D'
    [ci100, dfw] = data_frames
    
    fig1 = plt.figure(figsize=(5.5,4))

    ydata_B = ci100.loc[date_i:date_f].resample(Freq).sum()[str(bidx)]
    ydata_T = dfw.loc[date_i:date_f].resample(Freq).mean()[WQ]
    
    m, b, r_value, p_value, std_err = stats.linregress(ydata_T,ydata_B)

    plt.scatter(ydata_T,ydata_B);
    plt.plot(ydata_T,m*ydata_T+b, '-r');
    plt.xlabel(Wdict[WQ],fontsize= '13',fontweight="bold")
    plt.ylabel('Check-in Count',fontsize= '13',fontweight="bold")
    plt.title('$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad R^2 =' + str(np.round(r_value**2,2))+'$');

    plt.grid(b=True)
    plt.gca().tick_params(axis="x", labelsize=12)
    plt.gca().tick_params(axis="y", labelsize=12)

    return fig1
    

# Make Weather Sensitivity Bar Chart

def R2_chart(data_frames,date_i = dt.datetime(2016,1,1),date_f= dt.datetime(2018,1,1), fq = 30, bidx = 2, WQ = 'T'):
    
    Freq = str(fq)+'D'
    [df100,ci100, dfw] = data_frames
    
    ydata_T = dfw.loc[date_i:date_f].resample(Freq).mean()[WQ]

    R2 = []
    Bname = []

    for i in range(100):

        ydata_B = ci100.loc[date_i:date_f].resample(Freq).sum()[str(i)]
        m, b, r_value, p_value, std_err = stats.linregress(ydata_T,ydata_B)
        R2.append(r_value**2)
        Bname.append(df100.loc[i,'name']+'  ')
        
    idx = np.argsort(R2)[::-1]

    colors = ['royalblue',]*len(np.array(R2)[idx]) #'cornflowerblue'
    colors[list(idx).index(bidx)] = 'darkorange'
    
    return [{'x': np.array(R2)[idx],'y':np.array(Bname)[idx], 'type': 'bar','orientation':'h', 'marker': { 'color':colors }}] #fig2  


### Dash it out ###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
app.scripts.append_script({ 'external_url' : mathjax })

business_names = df100['name']

epoch = dt.datetime.utcfromtimestamp(0)

#def unix_time_millis(dt):
#    return (dt - epoch).total_seconds() * 1000.0

def get_marks(f):
    
    sampling = '180D'
    fsamp=f.resample(sampling).sum()
    dates = {}
    
    for z in fsamp.index:
        #dates[f.resample(sampling).sum().index.get_loc(z)] = j
        dates[f.index.get_loc(z)] = str(z.strftime('%Y-%m')) #+ "-" + str(z.day)
        
    return dates
    
app.layout = html.Div([
  
    # Page 1
    html.Div([
        
        # Row 1
        html.Div([

            html.Div([      
                html.H5('The Effect of Weather on Yelp Engagement',style=dict(fontSize = '26')),
                html.H6('An Exploratory Analysis', style=dict(color='#7F90AC',fontSize = '20')),
                ], className = "nine columns padded" ),

            html.Div([            
                html.H6([html.Span('Tamiro Villazon',style=dict(fontSize = '20'))]),
                html.H6('Data Physicist',style=dict(opacity=0.5,fontSize = '20'))
            ], className = "three columns gs-header gs-accent-header padded", style=dict(float='right') ),

        ], className = "row gs-header gs-text-header"),
        
         html.Br([]),
        
        # Row 2
        html.Div([
            html.P("Data science is gaining a prominent role in the world of business and investment. \
                    Data-driven, evidence-based analyses of available information have become crucial \
                    for effective decision-making in uncertain and ever-evolving markets.", style = dict(fontSize = '16') ),
            html.P("Particularly intriguing is the question of how we can use data to understand consumer behavior. \
            Even though people are fantastically complicated and of diverse tastes and preferences, patterns begin \
            to emerge when viewed through a proper lens.", style = dict(fontSize = '16') ),
            html.P("To study consumer patterns, we need indicators that quantify an aspect of their behavior \
            (e.g. purchases, click-through rates, or arrival times), and how that behavior changes in different \
            circumstances (this could be anything like the time of year, or being exposed to a new ad campaign). \
            In this project, we shall examine the impact of weather conditions on consumer's engagement with \
            local businesses in Pittsburgh, Pennsylvania. To quantify engagement, we will use Yelp data on check-in \
            events for 100 businesses in the region. To quantify circumstance, we use weather indicators such as\
            temperature, precipitation or wind speed.", style = dict(fontSize = '16') ),
            html.P(["Business and check-in data was retrieved from the ", html.A("Yelp Open Dataset", href='https://www.yelp.com/dataset', target="_blank") ,".\
            Note: the data is used here for personal and educational purposes."], style = dict(fontSize = '16') ),
            html.P(["Weather data from a land-based station in the Pittsburgh International Airport was obtained from the ",
            html.A("National Centers for Environmental Information", href='https://www.ncdc.noaa.gov/data-access/', target="_blank"), "."], style = dict(fontSize = '16') ),
            html.P(["Part of these data sets were cleaned and combined into a single dataframe for this project."], style = dict(fontSize = '16')),
            html.P("The interactive app below is designed to look at check-in counts for each business and weather \
            over any period of time between September 1, 2015 and August 30, 2018. The first graph (left) visualizes\
            the respective time series. The second graph (right) shows a scatter plot of check-in count vs weather\
            indicator, together with a best fit line to identify any trends. Feel free to use the drop menus to select\
            any business and weather indicator of your choice. Below you will find a slider which narrows/expands the time\
            window, if you wish to focus on a particular time frame. You can also set the sampling frequency to any number\
            of days; the sampling frequency determines the smallest time scale within which the weather data is\
            averaged and the check-in data is aggregated. For example, if sampling frequency is set to 7 days, then the\
            check-in time series data is grouped in batches of 7 days, and so each data point in the graphs becomes total\
            number of check-ins within a 7 day period.", style = dict(fontSize = '16') ),
            html.P("The long figure at the bottom summarizes the correlation coefficients (R-squared) of all the businesses.\
            These coefficients track the correlation between check-in count and weather over the specified period of time and\
            sampling.  They serve as a measure of the sensitivity of a business to weather conditions. Remarkably, there is a\
            broad range of correlation values, which indicates that there are businesses whose engagement (at least as\
            measured by Yelp check-in data) is highly dependent on weather, while others are much less impacted. \
            ", style = dict(fontSize = '16') ),
            html.P("This information can help businesses and investors identify trends and make strategic decisions. \
            For example, high weather correlations may prompt leaders of seasonal businesses to spend fewer \
            resources during offseason, or one could have non-seasonal businesses with high weather correlations \
            (maybe due to special ad campaigns ran every summer), which may reveal a need to improve engagement year-round.\
            Businesses with small weather correlations may also be attractive to investors who may\
            not want seasonal variations in portfolio performance. Of course in any decision, many different variables \
            must be considered; sensitivity to weather conditions could be an important piece of the puzzle. ", style = dict(fontSize = '16') )
        ], className = "row" ),
        
        ## Row 2.5
        html.Div([
            dcc.Textarea(id='my-id', value='initial value'),
        ],style={'display': 'none'}),
        #html.Br([]),
        
        # Row 3
        html.Div([
            html.Div([
                html.H6(["Select Business"], className = "gs-header gs-table-header padded",style=dict(fontSize = '20')),
                dcc.Dropdown(
                    id = 'b-drop',
                    options=[{'label': i, 'value': i} for i in business_names],
                    value=business_names[2])
            ], className = "six columns" ), 

            
            
             html.Div([
                html.H6(["Select Weather Indicator"], className = "gs-header gs-table-header padded",style=dict(fontSize = '20')),
                dcc.Dropdown(
                    id = 'w-drop',
                    options=[{'label': 'Temperature (\u00b0F)', 'value': 'T'},{'label': 'Precipitation (in)', 'value': 'P'},{'label': 'Wind Speed (mi/h)', 'value': 'WS'}],
                    value='T') 

             ], className = "six columns"),  
                      
        ], className = "row" ),
        
        html.Br([]),
         
        # Row 4
        html.Div([
            html.Div([
                dcc.Graph(
                    id='timetrace',
                    #figure= timetrace([ci100,dfw])#,#mpl_to_plotly(timetrace([ci100,dfw])),
                    #figure= mpl_to_plotly(timetrace([ci100,dfw]))
                    figure =mpl_to_plotly(timetrace([ci100,dfw])).update_layout(xaxis=dict(showticklabels=False),template="simple_white")
                    #layout=dict(title=column, xaxis=dict(type='date'))
                    #style = {'width': '10%'}
                    #xaxis=dict(type='date')
                )
            ], className = "six columns"), #,, style={'width': '45%','padding-right':'55%'}
            
            html.Div([
                dcc.Graph(
                    id='scatter',
                    figure= mpl_to_plotly(BW_scatter([ci100,dfw])).update_layout(template="simple_white")
                )
            ], className = "six columns"),
        
        ], className = "row" ),
        
        html.Br([]),
        dcc.Loading(id="loading", children=[html.Div(id="output-1")], type="default"),#html.Div(dcc.Graph(id='timetrace1'))
        html.Br([]),
        
        # Row 5
        html.Div([
            html.Div([
                dcc.RangeSlider(
                    id='range-slider',
                    updatemode='mouseup',
                    min=0,
                    max=len(ci100.index) - 1,
                    count=1,
                    #step=3,
                    value=[1*(len(ci100.index) - 1)//6, 5*(len(ci100.index) - 1)//6],#[0, len(ci100.index) - 1],
                    marks=get_marks(ci100),
                )
            ],className = "ten columns", style={'width': '96%', 'padding-left':'3%', 'padding-right':'1%'}) #,style={'textAlign': 'center'}
            
        ], className = "row" ),
        
        html.Br([]),
        html.Br([]),
        html.Br([]),
        
        
        # Row 6
        html.Div([
            html.Div([
                html.H5(["Sampling Frequency (in days) = "], className = "gs-header seven columns gs-table-header padded", style={'text-align': 'center'})
            ]), #, style=dict(color='#7F90AC') #,className = "four columns" ,style=dict(fontSize = '12')
            html.Div([
                dcc.Input(
                    id = 'sampling',
                    placeholder='Frequency',
                    type='number',
                    value=30,
                    min = 1,
                    max = 365
                    )  
            ]) #,className = "two columns"
        ], className = "row", style=dict({'padding-left':'16%','verticalAlign': 'middle'})), #, ,style=dict({float: 'right'}
        
        html.Br([]),
        html.Br([]),
        
        # Row 7
        html.Div([
            html.H5(['$$\\textbf{Business }\mathbf{R^2} \\textbf{ Distribution}$$'], className = "gs-header gs-table-header padded"),
            dcc.Graph(id='R-squared',
                            figure={
                                'data': R2_chart([df100,ci100,dfw]),
                                'layout': {
                                    "height": 2000,  # px
                                    #"title": dict(text='$\\textbf{Business }\mathbf{R^2} \\textbf{ Distribution}$',xanchor= 'left', yanchor= 'top', y=3.0), #$R^2$ ,y=2.0
                                    #"titlefont": {'size':20}, # not working with latex
                                    "xaxis":dict(mirror = "allticks", side= 'top',automargin=True), #mirror='allticks'
                                    "yaxis":dict(autorange="reversed"),
                                    "margin": dict(t=20, b= 20, l=370)
                                }
                            }           
                      
                   )], className = "row",style={"border":"2px black solid"} ) #,'text-align':'left'
        
        
        
    ], className = "page" )
    
])

@app.callback( #Output(component_id='R-squared', component_property='figure')
    [Output(component_id='my-id', component_property='value'),
    Output(component_id='timetrace', component_property='figure'),
    Output(component_id='scatter', component_property='figure'),
    Output(component_id='R-squared', component_property='figure')],
    [Input(component_id='b-drop', component_property='value'),
    Input(component_id='w-drop', component_property='value'),
    Input(component_id='range-slider', component_property='value'),
    Input(component_id='sampling', component_property='value')])
def update_plots(b_val,w_val,rng_vals,s_val):
    #(data_frames,date_i = dt.datetime(2015,7,1),date_f=... , fq = 7, bidx = 0, WQ = 'T')
    t1 = (ci100.index[rng_vals[0]].to_pydatetime())#.date())#.to_pydatetime()
    t2 = (ci100.index[rng_vals[1]].to_pydatetime())#.date())#.to_pydatetime()
    b_idx = df100[df100['name'] == b_val].index[0]
    #date_i=t1,date_f=t2,
    fig_t = mpl_to_plotly(timetrace([ci100,dfw],date_i=t1,date_f=t2, fq = s_val, bidx = b_idx, WQ = w_val)).update_layout(xaxis=dict(showticklabels=False),template="simple_white")         
    fig_s = mpl_to_plotly(BW_scatter([ci100,dfw],date_i=t1,date_f=t2, fq = s_val, bidx = b_idx, WQ = w_val)).update_layout(template="simple_white")
    fig_r = {
                'data': R2_chart([df100,ci100,dfw], date_i=t1,date_f=t2, fq = s_val, bidx = b_idx, WQ = w_val),
                'layout': {
                    "height": 2000,  # px
                    "xaxis":dict(mirror = "allticks", side= 'top',automargin=True), #mirror='allticks'
                    "yaxis":dict(autorange="reversed"),
                    "margin": dict(t=20, b= 20, l=370)
                }
            } 
    
    return 'You\'ve entered {},{},{},{} and'.format(b_val,w_val,rng_vals,s_val),fig_t,fig_s,fig_r


@app.callback(Output('output-1' , "children"), #'output-1' 
    [Input(component_id='b-drop', component_property='value'),
    Input(component_id='w-drop', component_property='value'),
    Input(component_id='range-slider', component_property='value'),
    Input(component_id='sampling', component_property='value')])
def input_triggers_spinner(value):
    time.sleep(1)
    return value          



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

if __name__ == '__main__':
    app.run_server(debug=False)
