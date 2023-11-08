import streamlit as st
import pandas as pd
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
df_im_bin = pd.read_csv('data/im_bin.csv')

LSGs = np.unique(df_im['Area Name'])
years = np.unique(df_im['Time Period'])
attributes = df_im.drop(['Area Name', 'Time Period'], axis=1).columns

selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)

df_im_s = df_im.copy()
if selected_lsgs != []:
    df_im_s = df_im_s.loc[df_im_s['Area Name'].isin(selected_lsgs), :]

st.markdown("---")

st.subheader('Comparison for a selected year')

st.markdown('The first chart for comparison is a radar chart where one can observe how selected municipalities compare between each other. By default, radar chart will not show up due to having too many municipalities.')
st.markdown('NOTE: To make values comparable on the radar chart, we scaled values so that the highest value is one for each year. In addition, missing values are filled with zero.')

selected_year = st.selectbox(label='Please select the year of observation', options=years, index=9)

fig = go.Figure()

if selected_lsgs != []:
    df_im_s_y = df_im_s.copy()
    df_im_s_y = df_im_s_y.loc[df_im_s_y['Time Period'] == selected_year, :]
    df_im_s_y['Net Migrations per 1000 inhabitants'] = (df_im_s_y['Net Migrations per 1000 inhabitants'] - df_im_s_y['Net Migrations per 1000 inhabitants'].min())/(df_im_s_y['Net Migrations per 1000 inhabitants'].max() - df_im_s_y['Net Migrations per 1000 inhabitants'].min())
    df_im_s_y[attributes] = df_im_s_y[attributes]/df_im_s_y[attributes].max()
    df_im_s_y = df_im_s_y.fillna(0)

    fig = go.Figure()

    for lsg in selected_lsgs:
        df_im_s_y_a = df_im_s_y.loc[df_im_s_y['Area Name'] == lsg, :].drop(['Time Period', 'Area Name'], axis=1)
        fig.add_trace(go.Scatterpolar(
            r=df_im_s_y_a.to_numpy()[0],
            theta=df_im_s_y_a.columns.to_numpy(),
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

dd = df_im.drop('Area Name', axis=1).groupby(['Time Period']).mean()
dd = dd.reset_index()
dd['Area Name'] = 'Serbia - Average'
df_im_s = pd.concat([df_im_s, dd])

fig = px.line(df_im_s, x="Time Period", y=selected_att, color='Area Name')
st.plotly_chart(fig)