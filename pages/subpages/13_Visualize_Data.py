import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide', page_title = 'Net Internal Migration Data Visualization')

st.title('Net Internal Migration Data Visualization')

st.markdown(
    '''
    This page aims to inspect the data for the Net Internal Migrations.

    Below you can find multi-select area where you can select up to ten municipalities. In addition, please select the attribute you would like to investigate.
    '''
)

df_im = pd.read_csv('data/im.csv')
df_im_s = df_im.copy()
df_im_s['Main Roads Accessibility'] = 1/df_im_s['Main Roads Accessibility']
df_im_s['Assistance and Care Allowance Share'] = 1/df_im_s['Assistance and Care Allowance Share']
df_im_s['Poverty Share'] = 1/df_im_s['Poverty Share']

df_im_s.iloc[:, 2:-1] = df_im_s.iloc[:, 2:-1]/df_im_s.iloc[:, 2:-1].max()
df_im_s['Net Migrations per 1000 inhabitants'] = (df_im_s['Net Migrations per 1000 inhabitants'] - df_im_s['Net Migrations per 1000 inhabitants'].min())/(df_im_s['Net Migrations per 1000 inhabitants'].max() - df_im_s['Net Migrations per 1000 inhabitants'].min())

gj = gpd.read_file('util/Opstina_l.geojson')
df_im_c = df_im.copy()
df_im_c['Area Name'] = df_im_c['Area Name'].str.upper()

LSGs = np.unique(df_im['Area Name'])
years = np.unique(df_im['Time Period'])
attributes = df_im.drop(['Area Name', 'Time Period'], axis=1).columns

selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)

st.markdown("---")

st.subheader('Comparison for a selected year')

st.markdown('The first chart for comparison is a radar chart where one can observe how selected municipalities compare between each other. By default, radar chart will not show up due to having too many municipalities.')
st.markdown('NOTE: To make values comparable on the radar chart, we scaled values so that the highest value is one for each year. In addition, missing values are filled with zero. In addition, we inverted the values where the lower value is better so, 1 is always the best value and 0 is always the worst.')

selected_year = st.selectbox(label='Please select the year of observation', options=years, index=9)

fig = go.Figure()

if selected_lsgs != []:
    df_im_s_y = df_im_s.copy()
    if selected_lsgs != []:
        df_im_s_y = df_im_s.loc[df_im_s_y['Area Name'].isin(selected_lsgs), :]
    df_im_s_y = df_im_s_y.loc[df_im_s_y['Time Period'] == selected_year, :]


    fig = go.Figure()

    for lsg in selected_lsgs:
        df_im_s_y_lsg = df_im_s_y.loc[df_im_s_y['Area Name'] == lsg, :].drop(['Time Period', 'Area Name'], axis=1)
        fig.add_trace(go.Scatterpolar(
            r=df_im_s_y_lsg.to_numpy()[0],
            theta=df_im_s_y_lsg.columns.to_numpy(),
            fill='toself',
            name=lsg
    ))
    st.plotly_chart(fig)
else:
    st.markdown('Please select municipality/municipalities to plot radar chart')

st.markdown('---')
st.subheader('Comparison of Municipalities')

st.markdown('This section aims at comparing multiple municipalities across time dimension. One can observe the behaviour of the municipality across time for the selected attribute. In addition, one can comprate selected municipalities with the average value for Serbia (it is always the last value).')

selected_att = st.selectbox(label='Please select attribute', options=attributes, index=9)

desc = [
 'The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.', 
 'The total number of doctors within a municipality divided by the 1,000 inhabitants.', 
 'The number of registered passenger vehicles per square meter of the municipality.', 
 'The average number of children in primary school that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 7 and 14.', 
 'The average number of children in secondary school (both vocational and general) that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 15 and 18.', 
 'Percentage of people within one municipality that use an increased allowance for assistance and care of another person.', 
 'Percentage of people within one municipality that uses social protection.',
 'The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of the municipality.',
 'The total length (in kilometers) of 1st-class road segments without highways divided by 1,000 inhabitants of the municipality.', 
 'The number of inhabitants per 1,000 inhabitants immigrated to the LSG (positive value) or emigrated from the LSG (negative value)']

st.markdown(f'{desc[np.where(attributes == selected_att)[0][0]]}')

dd = df_im.drop('Area Name', axis=1).groupby(['Time Period']).mean()
dd = dd.reset_index()
dd['Area Name'] = 'Serbia - Average'
if selected_lsgs != []:
    df_im_s_y = df_im.loc[df_im['Area Name'].isin(selected_lsgs), :]
    df_im_s = pd.concat([df_im_s_y, dd])
else:
    df_im_s = pd.concat([df_im, dd])

fig = px.line(df_im_s, x="Time Period", y=selected_att, color='Area Name')
st.plotly_chart(fig)

st.markdown('---')
st.subheader('Map Visualization')

st.markdown('Please find below the map where one can visualize input attributes and observe the values on the map.')

selected_year_map = st.selectbox(label='Please select year', options=years, key=23, index=9)
selected_att_map = st.selectbox(label='Please select attribute', options=attributes, key=22, index=9)

st.markdown(f'{desc[np.where(attributes == selected_att_map)[0][0]]}')

df_im_c = df_im_c.loc[df_im_c['Time Period'] == selected_year_map, :]
gj.merge(df_im, left_on='opstina_imel', right_on='Area Name')

if selected_att_map == 'Net Migrations per 1000 inhabitants':
    aa = df_im_c[selected_att_map]
    sorted_data = np.sort(aa)
    position = np.searchsorted(sorted_data, 0)
    percentile = (position / len(sorted_data))

    custom_color_scale = [
        (0, 'red'),
        (4/6, 'white'),
        (1, 'green')
    ]
    vals = (-4, 2)
elif selected_att_map == 'Main Roads Accessibility':
    aa = df_im_c[selected_att_map]
    sorted_data = np.sort(aa)
    position = np.searchsorted(sorted_data, 0)
    percentile = (position / len(sorted_data))
    first_q = aa.quantile([0.25]).values[0]
    third_q = aa.quantile([0.75]).values[0]
    
    custom_color_scale = [
        (0, 'green'),
        (0.5, 'white'),
        (1, 'red')
    ]

    vals = (first_q, third_q)
elif selected_att_map == 'Assistance and Care Allowance Share':
    aa = df_im_c[selected_att_map]
    sorted_data = np.sort(aa)
    position = np.searchsorted(sorted_data, 0)
    percentile = (position / len(sorted_data))
    first_q = aa.quantile([0.25]).values[0]
    third_q = aa.quantile([0.75]).values[0]
    
    custom_color_scale = [
        (0, 'green'),
        (0.5, 'white'),
        (1, 'red')
    ]

    vals = (first_q, third_q)
elif selected_att_map == 'Poverty Share':
    aa = df_im_c[selected_att_map]
    sorted_data = np.sort(aa)
    position = np.searchsorted(sorted_data, 0)
    percentile = (position / len(sorted_data))
    first_q = aa.quantile([0.25]).values[0]
    third_q = aa.quantile([0.75]).values[0]
    
    custom_color_scale = [
        (0, 'green'),
        (0.5, 'white'),
        (1, 'red')
    ]

    vals = (first_q, third_q)
else:
    aa = df_im_c[selected_att_map]
    sorted_data = np.sort(aa)
    position = np.searchsorted(sorted_data, 0)
    percentile = (position / len(sorted_data))
    first_q = aa.quantile([0.25]).values[0]
    third_q = aa.quantile([0.75]).values[0]
    
    custom_color_scale = [
        (0, 'red'),
        (0.5, 'white'),
        (1, 'green')
    ]

    vals = (first_q, third_q)

fig = px.choropleth(data_frame=df_im_c, geojson=gj,
                    featureidkey='properties.opstina_imel',
                    locations='Area Name', 
                    color_continuous_scale=custom_color_scale,
                    range_color=vals,
                    color=selected_att_map, 
                    height=700, width=700)
fig.update_geos(fitbounds = 'locations', visible=False)
st.plotly_chart(fig)