import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout='wide', page_title = 'Gross Value Added per capita Data Exploration')

st.title('Gross Value Added per capita Data Exploration')

st.markdown(
    '''
    This page aims to inspect the data for the Gross Value Added per capita.

    Below you can find multi-select area where you can select up to ten municipalities and up to ten years. Selection of municipalities and time periods impact the table below. Values in the table are denoted with red or green background depending on whether that particular value is in the category *Poor* (red color) or *Good* (green color). 
    '''
)

df_gva = pd.read_csv('data/gva.csv')
df_gva = df_gva.loc[df_gva['GVA Per Capita Normalized'].notna(), :] # Some LSGs do not hva GVA, thus they are omitted from the analysis

LSGs = np.unique(df_gva['Area Name'])
years = np.unique(df_gva['Time Period'])

selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)
selected_years = st.multiselect('Please select years for which you would want to observe the data:', years, max_selections=10)

def highlight_vehicle_density(x):
    color = ''
    if x > 21.286:
        color = 'red'
    elif x <= 10.272:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_motorcycle_density(x):
    color = ''
    if x <= 0.281:
        color = 'red'
    elif x > 0.762:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_main_road_acc(x):
    color = ''
    if x > 13.025:
        color = 'red'
    elif x <= 5.461:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_local_roads_den(x):
    color = ''
    if x > 0.382:
        color = 'red'
    elif x <= 0.213:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_preschool(x):
    color = ''
    if x <= 2.086:
        color = 'red'
    elif x > 2.682:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_employment(x):
    color = ''
    if x <= 0.291:
        color = 'red'
    elif x > 0.378:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_unemployment(x):
    color = ''
    if x > 122.0:
        color = 'red'
    elif x <= 77.0:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_active_companies(x):
    color = ''
    if x <= 7.755:
        color = 'red'
    elif x > 10.985:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_transport(x):
    color = ''
    if x < 0.562:
        color = 'red'
    elif x > 0.562:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_doctors(x):
    color = ''
    if x <= 1.3:
        color = 'red'
    elif x > 2.1:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_tourists(x):
    color = ''
    if x <= 978.0:
        color = 'red'
    elif x > 7629.333:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_gva(x):
    color = ''
    if x <= -0.25:
        color = 'red'
    elif x > 0:
        color = 'lightgreen'
    return f'background-color : {color}'

df_gva_s = df_gva.copy()
if (selected_lsgs != []) & (selected_years == []):
    df_gva_s = df_gva_s.loc[df_gva_s['Area Name'].isin(selected_lsgs), :]
elif (selected_lsgs == []) & (selected_years != []):
    df_gva_s = df_gva_s.loc[df_gva_s['Time Period'].isin(selected_years), :]
elif (selected_lsgs != []) & (selected_years != []):
    df_gva_s = df_gva_s.loc[df_gva_s['Area Name'].isin(selected_lsgs), :]
    df_gva_s = df_gva_s.loc[df_gva_s['Time Period'].isin(selected_years), :]

st.markdown("---")

st.table(df_gva_s.head(10).style.applymap(lambda x: highlight_vehicle_density(x), subset=['Vehicles density']).applymap(lambda x: highlight_motorcycle_density(x), subset=['Motorcycles density']).applymap(lambda x: highlight_main_road_acc(x), subset=['Main road accessibility']).applymap(lambda x: highlight_local_roads_den(x), subset=['Local roads density']).applymap(lambda x: highlight_preschool(x), subset=['Preschool children enrollment rate']).applymap(lambda x: highlight_doctors(x), subset=['Doctors accessibility']).applymap(lambda x: highlight_employment(x), subset=['Municipality employment rate']).applymap(lambda x: highlight_unemployment(x), subset=['Unemployed rate']).applymap(lambda x: highlight_active_companies(x), subset=['Active companies rate']).applymap(lambda x: highlight_transport(x), subset=['Transport and storage investments rate']).applymap(lambda x: highlight_tourists(x), subset=['Tourists Arrivals']).applymap(lambda x: highlight_gva(x), subset=['GVA Per Capita Normalized']))
st.markdown(
    '''
    - Vehicles Density - Number of passenger vehicles / Total Surface area
    - Motorcycle Density - (Number of mopeds + Number of bikes) / Total Surface area (km^2)
    - Main Roads Accessibility - 1000 * Length of total road segments (km) / Total Population
    - Local Roads Density - Length of local roads (km) / Total Surface area (km^2)
    - Preschool children enrollment rate - Number of reported children attending obligatory preschool education / Estimated number of 6 years old children
    - Municipality employment rate - Number of employed people within LSG / Estimated working-age population within LSG
    - Unemployment rate - 1000 * Number of registered unemployed inhabitants / Total Population
    - Active companies rate - 1000 * Number of active companies / Total Population
    - Transport and storage investments rate - The amount LSG invested in new fixed assets in transport and storage per capita / 1000 (unit of measurement - 1000 RSD/capita)
    - Doctors Accessibility - 1000 * Number of doctors / Total Population 
    - Tourist Arrivals - The total number of tourists arrivals within LSG (both domestic and international)
    - Gross Value Added per capita - Estimated Gross Value Added (constant prices) / Total Population. The value is normalized using Z transformation on a yearly basis (thus, a yearly average is zero, and standard deviation is one)
    '''
)