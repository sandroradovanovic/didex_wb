import pandas as pd
import numpy as np

import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

st.set_page_config(layout='wide', 
                   page_title = 'Accessibility Attributes')

st.title('Accessibility Attributes')

st.markdown(
    '''
    One part of this project was design and development of the accessibility attributes. Accessibility facilitates interactions between communities, companies, and various services in an area. In this context, resource accessibility is a key determinant of internal migration patterns, as individuals seek to relocate to regions with better access to education, employment, healthcare, and other critical resources. In contrast, areas with limited access to essential resources serve as push factors for outmigration.
    
    Accessibility to resources, in this context, implies the access to resources both within its own municipality, as well as within neighbouring ones, provided that the distance between the two municipalities falls within 0 (value per capita or per 1,000 inhabitants), 15, 30, 45, and 60 minutes. Within our analysis, accessibility is calculated for an access to health and education institutions, as well as to jobs – normalized by population. 
    '''
)

df = pd.read_csv('data/accessibility_final_2.csv')
df_time = pd.read_csv('data/time_distance_matrix.csv')
df_time = df_time.set_index('Source')
df_time = df_time.fillna(0.0)

LSGs = np.unique(df['Area Name'])
years = np.unique(df['Time Period'])

selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)
selected_year = st.selectbox('Please select year for which you would want to observe the data:', years, index=10)

with st.expander('Explanation'):
    st.markdown(
        '''
        Due to restrictions on the availability of data and the locations of the resources we utilized special *interaction-based accessibility measures*. More specifically, **accessibility to a resource from one municipality is equal to availability to reach the desired resource within a predefined time-frame (of 0, 15, 30, 45, and 60 minutes). However, population size within the specified time-frame is also taken into account as a discount factor. In other words, total amount of resources are divided by population size leading to a measurment unit of resource per capita**. To calculate the distance between two municipalities we used Google Maps Directions API and calculated the distance between the centers of the two municipalities using the car. The center of the municipality is, according to Google Maps, a place where the main building of the municipality is located. This might not be the center of the municipality (the middle location of the municipality), nor the place where most people in the municipality live. The positive sides of the Google Maps Direction API are that the exact roads were used, with information about the speed limits and traffic state (at noon). Consequently, as a result, we obtained information about the distance in meters between the two municipalities, as well as the time needed to get from one municipality to another.

        Every attribute in the accessibility format is calculated *per 1,000 inhabitants* except for the *Life expectancy at birth*, *Average annual net salaries and wages*, and *Total Business Incomes* which present the value *per capita*. 
        '''
    )

atts = df.columns[~df.columns.str.startswith('access_')].drop(['Time Period', 'Area Name', 'Population size - Total', 'GVA Current prices'])

selected_attribute = st.multiselect('Please select the attribute you want to inspect', atts, max_selections=10)
selected_time = st.selectbox('Please select the time distance for the :', [15, 30, 45, 60], index=2)

if (selected_attribute != []) & (selected_time != []) & (selected_lsgs != []):
    st.subheader('Data Table')
    st.markdown('Before the chart where one can compare values of multiple LSGs, we present a data table with the original value, normalized value (per 1,000 inhabitants or per capita) that represents accessibility within the LSG and normalized value (per 1,000 inhabitants or per capita) that represents accessibility to a resource within the selected time-frame.')

    gj = gpd.read_file('util/Opstina_l.geojson')

    fig = go.Figure()

    if selected_lsgs != []:
        df_s = df.copy()
        df_s = df_s.loc[:, ['Area Name', 'Time Period', *selected_attribute, *[f'access_0_' + x for x in selected_attribute], *[f'access_{selected_time}_' + x for x in selected_attribute]]]
        df_s_tab = df_s.loc[df_s['Area Name'].isin(selected_lsgs), :]
        st.table(df_s_tab.head(10))

        df_s = df_s.loc[:, ['Area Name', 'Time Period', *[f'access_{selected_time}_' + x for x in selected_attribute]]]
        

        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        df_s_a = df_s.copy()
        for i in range(2,len(df_s.columns)):
            df_s_a.iloc[:, i] = sigmoid((df_s_a.iloc[:, i] - df_s_a.iloc[:, i].mean())/df_s_a.iloc[:, i].std())

        if selected_lsgs != []:
            df_s_a = df_s_a.loc[df_s_a['Area Name'].isin(selected_lsgs), :]
        df_s_a = df_s_a.loc[df_s_a['Time Period'] == selected_year, :]


        st.subheader('Visualization for the selected attributes')
        st.markdown('Values in the Visualization are normalized such that the highest value for the selected year in the entire Serbia is equal to one, while the lowest value is equal to zero. In addition, the average value for Serbia is 0.5.')
        
        fig = go.Figure()

        for lsg in selected_lsgs:
            df_s_y = df_s_a.loc[df_s['Area Name'] == lsg, :].drop(['Time Period', 'Area Name'], axis=1)
            fig.add_trace(go.Scatterpolar(
                r=df_s_y.to_numpy()[0],
                theta=df_s_y.columns.to_numpy(),
                fill='toself',
                name=lsg
        ))
        st.plotly_chart(fig)

        st.subheader('Map Visualization')
        st.markdown('Finally, we present the data on the map.')
        
        custom_color_scale = [
            (0, 'red'),
            (0.5, 'white'),
            (1, 'green')
        ]
        
        for att in selected_attribute:
            vals = (df_s[f'access_{selected_time}_{att}'].quantile([0.33]).values[0], df_s[f'access_{selected_time}_{att}'].quantile([0.66]).values[0])
        
            opstine = []
            for l in selected_lsgs:
                rad_mun = df_time.loc[l, :] <= selected_time * 60
                for o in df_time.index[rad_mun].to_list():
                    opstine.append(o)
            opstine = list(set(opstine))
            df_s_s = df_s.loc[df_s['Area Name'].isin(opstine), :]
            
            df_s_s['Area Name'] = df_s_s['Area Name'].str.upper().str.replace(' ', '')

            fig = px.choropleth(data_frame=df_s_s, 
                            geojson=gj,
                            featureidkey='properties.opstina_imel',
                            locations='Area Name', 
                            color_continuous_scale=custom_color_scale,
                            range_color=vals,
                            color=f'access_{selected_time}_{att}', 
                            height=700, width=700)
            fig.update_geos(fitbounds = 'locations', visible=False)
            st.plotly_chart(fig)
    
    else:
        st.markdown('Please select municipality/municipalities to plot radar chart')
