import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide', page_title = 'Gross Value Added per capita Data Visualization')

st.title('Gross Value Added per capita Data Visualization')

st.markdown(
    '''
    This page aims to inspect the data for the Gross Value Added per capita.

    Below you can find multi-select area where you can select up to ten municipalities. In addition, please select the attribute you would like to investigate.
    '''
)

df_gva = pd.read_csv('data/gva.csv')
df_gva_s = df_gva.copy()
df_gva_s['Main road accessibility'] = 1/df_gva_s['Main road accessibility']
df_gva_s['Local roads density'] = 1/df_gva_s['Local roads density']
df_gva_s['Unemployed rate'] = 1/df_gva_s['Unemployed rate']
df_gva_s['Vehicles density'] = 1/df_gva_s['Vehicles density']

df_gva_s.iloc[:, 2:-1] = df_gva_s.iloc[:, 2:-1]/df_gva_s.iloc[:, 2:-1].max()
df_gva_s['GVA Per Capita Normalized'] = (df_gva_s['GVA Per Capita Normalized'] - df_gva_s['GVA Per Capita Normalized'].min())/(df_gva_s['GVA Per Capita Normalized'].max() - df_gva_s['GVA Per Capita Normalized'].min())

gj = gpd.read_file('util/Opstina_l.geojson')
df_gva_c = df_gva.copy()
df_gva_c['Area Name'] = df_gva_c['Area Name'].str.upper()

LSGs = np.unique(df_gva['Area Name'])
years = np.unique(df_gva['Time Period'])
attributes = df_gva.drop(['Area Name', 'Time Period'], axis=1).columns

selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)

st.markdown("---")

st.subheader('Comparison for a selected year')

st.markdown('The first chart for comparison is a radar chart where one can observe how selected municipalities compare between each other. By default, radar chart will not show up due to having too many municipalities.')
st.markdown('NOTE: To make values comparable on the radar chart, we scaled values so that the highest value is one for each year. In addition, we inverted the values where the lower value is better, thus, 1 is always the best value and 0 is always the worst.')
st.markdown('Regarding the *GVA Per Capita Normalized* values are scaled with max-min normalization technique, which means that zero is the worst value and one is the best value.')

selected_year = st.selectbox(label='Please select the year of observation', options=years, index=9)

fig = go.Figure()

if selected_lsgs != []:
    df_gva_s_y = df_gva_s.copy()
    if selected_lsgs != []:
        df_gva_s_y = df_gva_s.loc[df_gva_s_y['Area Name'].isin(selected_lsgs), :]
    df_gva_s_y = df_gva_s_y.loc[df_gva_s_y['Time Period'] == selected_year, :]


    fig = go.Figure()

    for lsg in selected_lsgs:
        df_gva_s_y_lsg = df_gva_s_y.loc[df_gva_s_y['Area Name'] == lsg, :].drop(['Time Period', 'Area Name'], axis=1)
        fig.add_trace(go.Scatterpolar(
            r=df_gva_s_y_lsg.to_numpy()[0],
            theta=df_gva_s_y_lsg.columns.to_numpy(),
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
 'The total length (in kilometers) of all road segments (I level without highways, II, and local road segments) per square meter of the municipality.', 
 'The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of an LSG.', 
 'The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.', 
 'The number of registered passenger vehicles per square meter of the municipality.', 
 'The total number of active companies per 1,000 inhabitants.',
 'Percentage of people that are registered to work in an LSG. This number is obtained by dividing the number of employed people within an LSG by the estimated working-age population in an LSG.', 
 'The total number of registered unemployed people per 1,000 inhabitants in an LSG.', 
 'The total amount (in RSD thousands) an LSG invested in new fixed assets in transport and storage per capita.', 
 'The total number of doctors within an LSG divided by the 1,000 inhabitants.', 
 'Percentage of children attending the obligatory preschool education within an LSG. To calculate the number of children, we used an estimate of the number of children having 6 years.', 
 'The total number of tourist arrivals within one year in an LSG.',
 'Estimated Gross Value Added (constant prices) per capita. The value is normalized using Z transformation on a yearly basis (thus, a yearly average is zero, and standard deviation is one).'
 ]

st.markdown(f'{desc[np.where(attributes == selected_att)[0][0]]}')

dd = df_gva.drop('Area Name', axis=1).groupby(['Time Period']).mean()
dd = dd.reset_index()
dd['Area Name'] = 'Serbia - Average'
if selected_lsgs != []:
    df_gva_s_y = df_gva.loc[df_gva['Area Name'].isin(selected_lsgs), :]
    df_gva_s = pd.concat([df_gva_s_y, dd])
else:
    df_gva_s = pd.concat([df_gva, dd])

fig = px.line(df_gva_s, x="Time Period", y=selected_att, color='Area Name')
st.plotly_chart(fig)

st.markdown('---')
st.subheader('Map Visualization')

st.markdown('Please find below the map where one can visualize input attributes and observe the values on the map.')

selected_year_map = st.selectbox(label='Please select year', options=years, key=23, index=9)
selected_att_map = st.selectbox(label='Please select attribute', options=attributes, key=22, index=9)

st.markdown(f'{desc[np.where(attributes == selected_att_map)[0][0]]}')

df_gva_c = df_gva_c.loc[df_gva_c['Time Period'] == selected_year_map, :]
gj.merge(df_gva, left_on='opstina_imel', right_on='Area Name')

if selected_att_map == 'Main road accessibility':
    aa = df_gva_c[selected_att_map]
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
elif selected_att_map == 'Local roads density':
    aa = df_gva_c[selected_att_map]
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
elif selected_att_map == 'Unemployed rate':
    aa = df_gva_c[selected_att_map]
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
elif selected_att_map == 'Vehicles density':
    aa = df_gva_c[selected_att_map]
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
    aa = df_gva_c[selected_att_map]
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

fig = px.choropleth(data_frame=df_gva_c, geojson=gj,
                    featureidkey='properties.opstina_imel',
                    locations='Area Name', 
                    color_continuous_scale=custom_color_scale,
                    range_color=vals,
                    color=selected_att_map, 
                    height=700, width=700)
fig.update_geos(fitbounds = 'locations', visible=False)
st.plotly_chart(fig)