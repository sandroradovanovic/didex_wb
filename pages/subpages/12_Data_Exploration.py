import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout='wide', page_title = 'Net Internal Migration Data Exploration')

st.title('Net Internal Migration Data Exploration')

st.markdown(
    '''
    This page aims to inspect the data for the Net Internal Migrations.

    Below you can find multi-select area where you can select up to ten municipalities and up to ten years. Selection of municipalities and time periods impact the table below. Values in the table are denoted with red or green background depending on whether that particular value is in the category *Poor* (red color) or *Good* (green color). 
    '''
)

df_im = pd.read_csv('data/im.csv')
LSGs = np.unique(df_im['Area Name'])
years = np.unique(df_im['Time Period'])

selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)
selected_years = st.multiselect('Please select years for which you would want to observe the data:', years, max_selections=10)

def highlight_vehicle_density(x):
    color = ''
    if x <= 10.272:
        color = 'red'
    elif x > 21.286:
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
    if x > 1.676:
        color = 'red'
    elif x <= 0.693:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_local_roads_den(x):
    color = ''
    if x <= 107.736:
        color = 'red'
    elif x > 450.854:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_prim_school(x):
    color = ''
    if x <= 365.412:
        color = 'red'
    elif x > 590.641:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_sec_school(x):
    color = ''
    if x <= 269.804:
        color = 'red'
    elif x > 435.033:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_assistance(x):
    color = ''
    if x > 0.6:
        color = 'red'
    elif x <= 0.5:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_poverty(x):
    color = ''
    if x > 12.9:
        color = 'red'
    elif x <= 8.1:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_doctors_acc(x):
    color = ''
    if x <= 1.3:
        color = 'red'
    elif x > 2.1:
        color = 'lightgreen'
    return f'background-color : {color}'

def highlight_net_migrations(x):
    color = ''
    if x <= -4:
        color = 'red'
    elif x > 0:
        color = 'lightgreen'
    return f'background-color : {color}'

df_im_s = df_im.copy()
if (selected_lsgs != []) & (selected_years == []):
    df_im_s = df_im_s.loc[df_im_s['Area Name'].isin(selected_lsgs), :]
elif (selected_lsgs == []) & (selected_years != []):
    df_im_s = df_im_s.loc[df_im_s['Time Period'].isin(selected_years), :]
elif (selected_lsgs != []) & (selected_years != []):
    df_im_s = df_im_s.loc[df_im_s['Area Name'].isin(selected_lsgs), :]
    df_im_s = df_im_s.loc[df_im_s['Time Period'].isin(selected_years), :]

st.markdown("---")

st.table(df_im_s.head(10).style.applymap(lambda x: highlight_vehicle_density(x), subset=['Vehicles Density']).applymap(lambda x: highlight_motorcycle_density(x), subset=['Motorcycle Density']).applymap(lambda x: highlight_main_road_acc(x), subset=['Main Roads Accessibility']).applymap(lambda x: highlight_local_roads_den(x), subset=['Local Roads Density']).applymap(lambda x: highlight_prim_school(x), subset=['Primary School Attendance']).applymap(lambda x: highlight_sec_school(x), subset=['Secondary School Attendance']).applymap(lambda x: highlight_assistance(x), subset=['Assistance and Care Allowance Share']).applymap(lambda x: highlight_poverty(x), subset=['Poverty Share']).applymap(lambda x: highlight_doctors_acc(x), subset=['Doctors Accessibility']).applymap(lambda x: highlight_net_migrations(x), subset=['Net Migrations per 1000 inhabitants']))
st.markdown(
    '''
- Motorcycle Density - (Number of mopeds + Number of bikes) / Total Surface area (km^2)
- Doctors Accessibility - 1000 * Number of doctors / Total Population 
- Vehicles Density - Number of passenger vehicles / Total Surface area
- Primary School Attendance - Total number of children age 7-14 / Total number of Primary schools
- Secondary School Attendance - Total number of children age 14-18 / Total number of Secondary schools
- Assistance and Care Allowance Share - Number of inhabitants using assistance and care allowance / Total popuation
- Poverty Share - Number of social protection beneficiaries / Total popuation
- Local Roads Density - Amount of local roads (km) / Total Surface area (km^2)
- Main Roads Accessibility - 1000 * Length of main road segments (km) / Total Population
- Net Migrations per 1000 inhabitants - 1000 * (Recorded immigrations - Recorded emigrations) / Total Population
    '''
)