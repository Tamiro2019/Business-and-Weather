# Business-and-Weather
Independent project on correlations between business engagement and weather.

## Overview 

This project builds a [web app](https://business-and-weather.herokuapp.com) to visualize and analyze correlations
between weather and consumer engagement for 50 top businesses in Pittsburgh, Pennsylvania. It uses check-in data
from the [Yelp Dataset](https://www.yelp.com/dataset) to quantify consumer engagement, and weather data from the
[National Centers for Environmental Information](https://www.ncdc.noaa.gov/data-access/). The data is used here
for personal and educational purposes. 

## Technologies 

* Python 3.6

### Libraries

* Plotly Dash
* Pandas, Numpy, Scipy, Matplotlib

See __requirements.txt__ for the necessary compatibility requirements.  

## Key Files

The app.py files contains the core code used to run the Dash app run in Heroku. The app_notebook.ipybn 
contains the core code in notebook format.

The Data_Cleaning.ipynb gives a step by step explanation of how the data was cleaned and assembled into condensed 
dataframes to run the app. The clean data is stored in yb50.csv and ts50.csv.

The yb50.csv file stores basic business information. The ts50.csv stores time series data for check-in counts 
and weather indicators (temperature, precipitation, and wind-speed).

## Launch

To use the app online visit [https://business-and-weather.herokuapp.com](https://business-and-weather.herokuapp.com).

To run the code directly instead, you may fork and download this repository. Setup an environment in accordance with __requirements.txt__, and you should be able to run the app.py directly (or app_notebook.ipybn with jupyter).
