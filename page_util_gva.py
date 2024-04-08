import streamlit as st

import pandas as pd
import numpy as np
import geopandas as gpd

import dex_model_gva as dm

import plotly.express as px
import plotly.graph_objects as go

from annotated_text import annotated_text

def introduction_text():
    st.markdown(
        '''
        The final DIDEX extracted DEX model for the (Pooled) GVA Per Capita is shown below. **NOTE: This is our best effort having in mind the available data, constraints in data acquisition, the validity and accuracy of data, as well as the usability of the solution. The proposed model is intentially created to work with categories as this human decision makers are more prone to use qualitative values instead of exact numbers. Another note is that the accuracy of the predictive model is around 70% (using yearly based cross-validation) and this result is comparable to the results obtained using more complex machine learning models (that are not understandable to the human and that do not have an option to provide a list of possible policy interventions).**
        
        Gross-value-added per capita prediction has a confounding effect as the economy improves over time. Consequently, the best-performing municipalities in 2010 had similar economic activity as the worst-performing municipalities in 2021. Although expected, one can argue about whether these numbers are comparable. In other words, using these values to explain or predict the gross value added would lead to inaccuracies in interpretation and consequently policymaking. To make each year comparable (in other words, to eliminate the effect of time), we performed year-wise pooled Z transformation. This means gross value added per capita has an average value equal to zero and a standard deviation equal to one for every year. Thus, municipalities that are consistently below average would have negative values, and vice-versa municipalities that are consistently above average would have positive values.

        The obtained DEX model can be divided into three categories. Namely, *Tourism and Traffic*, *Social Factors*, and *Economy and Investments*. This depicts the multifaceted nature of the gross value added prediction. While logically the state of economy in LSG and investments in transportation have a high influence on the outcome, accessibility to resources identified by *Tourism and Traffic* factor, as well as *Social Factors* influence the outcome as well.  
        '''
    )

def attribute_description():
    st.subheader('Attribute Description')

    data = {
        'Attribute Name': ['Vehicles density', 'Motorcycles density', 'Main road accessibility', 
                        'Local roads density', 'Preschool children enrollment rate', 'Municipality employment rate', 
                        'Unemployment rate', 'Active companies rate', 'Transport and storage investments rate', 
                        'Doctors accessibility', 'Tourist Arrivals'],
        'Description': ['The number of registered passenger vehicles per square meter of the municipality.', 
                        'The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.', 
                        'The total length (in kilometers) of all road segments (I level without highways, II, and local road segments) per square meter of the municipality.', 
                        'The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of an LSG.',
                        'Percentage of children attending the obligatory preschool education within an LSG. To calculate the number of children, we used an estimate of the number of children having 6 years.', 
                        'Percentage of people that are registered to work in an LSG. This number is obtained by dividing the number of employed people within an LSG by the estimated working-age population in an LSG.', 
                        'The total number of registered unemployed people per 1,000 inhabitants in an LSG.', 
                        'The total number of active companies per 1,000 inhabitants.',
                        'The total amount (in RSD thousands) an LSG invested in new fixed assets in transport and storage per capita.', 
                        'The total number of doctors within an LSG divided by the 1,000 inhabitants.', 
                        'The total number of tourist arrivals within one year in an LSG.'],
        'Tooltip': ['The attribute Vehicles density has a positive correlation with the different output – gross value added per capita.  This is expected as vehicles do act as a sign of the economic power of a LSG.', 
                    'A clearer signal is observed for the Motorcycles density attribute. Again, the correlation between the number of motorcycles density and gross value added per capita is positive.', 
                    'An example of a negative correlation with the gross value added per capita is observed with the main roads accessibility attribute.', 
                    'Regarding the local roads density attribute, there is a slightly negative correlation with gross value added per capita.', 
                    'Interestingly, one of the most informative attributes for gross value added per capita explanation and prediction was the preschool children enrollment rate. This attribute has a very high positive correlation with the output attribute with a high signal that LSGs with a low share of enrolled children in preschool education have a low gross value added per capita and that LSGs with a high share of enrolled children in preschool education have a high gross value added per capita.', 
                    'However, the most informative attribute is the municipality employment rate. The interpretation is the same as for the previous one as a higher municipality employment rate indicates a higher gross value added per capita, and vice-versa, a lower municipality employment rate indicates a lower gross value added per capita.', 
                    'Another attribute we investigated is unemployment rate. As expected, the correlation is negative – a high number of unemployed people indicates a lower economic activity in an LSG, which consequently leads to a low gross value added. On the other side of the spectrum, a low level of unemployment is associated with a high gross value added.', 
                    'The active companies rate is very similar to the municipality employment rate in terms of interpretation. More specifically, the correlation with gross value added per capita is positive with high values indicating a high gross value added per capita, and low values indicating a low gross value added per capita.', 
                    'Transport and storage investments rate has a positive correlation with the gross value added per capita, however, the strength of the correlation is not as strong as for the employment rate or preschool enrollment rate.', 
                    'Doctors accessibility is another attribute used for the explanation and prediction of gross value added per capita. This attribute does not follow a linear association (or correlation).', 
                    'The final attribute was the total number of tourist arrivals, which has a positive association with the gross value added per capita.']
    }

    df_text = pd.DataFrame(data)

    def add_tooltip(row):
        return f'<span title="{row["Tooltip"]}">{row["Attribute Name"]}</span>'

    df_text['Attribute Name'] = df_text.apply(add_tooltip, axis=1)
    # st.table(df_text, unsafe_allow_html=True)
    st.write(df_text[['Attribute Name', 'Description']].to_html(escape=False), unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Model Hierarchy')

    st.markdown(
        '''
        - Gross Value Added *per capita* (pooled)
            - Tourism and Traffic
                - Tourists Arrivals
                - Road and Traffic
                    - Local Traffic
                        - Local roads density
                        - Motorcycle density
                    - General Traffic
                        - Vehicles density
                        - Main roads accessibility
            - Social Factors
                - Preschool children enrollment rate
                - Doctors accessibility
            - Economy and Investments
                - Economy State
                    - Employment State
                        - Municipality employment rate
                        - Unemployed rate
                    - Active companies rate
                - Transport and storage investments rate
        '''
    )

    st.markdown(
        '''
        The DEX model merged the eleven attributes from data into new extracted features. 
    - The attributes **Local roads density** and the **Motorcycles density** make decisions about the feature **Local Traffic**.
    - The attributes **Vehicles density** and **Main road accessibility** make decisions about the feature **General Traffic**.
        - Newly created attributes **Local Traffic** and **General Traffic** are merged into **Road and Traffic**.
            - **Road and Traffic** and **Tourists Arrivals** are creating a concept entitled **Tourism and Traffic**.
    - The attributes **Preschool children enrollment rate** and **Doctors accessibility** make decisions about the feature **Social Factors**.
    - The attributes **Municipality employment rate** and **Unemployed rate** make decisions about the feature **Employment State**.
        - **Employment State** and **Active companies rate** make **Economy State**.
            - Then, **Economy State** and **Transport and storage investments rate** create **Economy and Investments**.
    - Finally, **Tourism and Traffic**, **Social Factors**, and **Economy and Investments** are used to predict the value of **Gross Value Added *per capita* (pooled)**.
        '''
    )

def attribute_values():
    st.subheader('Attribute Values')

    st.markdown(
        '''
        Since DEX models are working with qualitative data, we discretized data according to the following values:
        '''
    )

    df = [['1', 'Vehicles density', '> 21.286', '(10.272; 21.286)', '<= 10.272'], 
    ['2', 'Motorcycles density', '<= 0.281', '(0.281; 0.762)', '> 0.762'],
    ['3', 'Main Road Accessibility', '> 13.025', '(5.461; 13.025)', '<= 5.461'],
    ['4', 'Local roads density', '> 0.382', '(0.213; 0.382)', '<= 0.213'],
    ['5', 'Preschool children enrollment rate', '<= 2.086', '(2.086; 2.682)', '> 2.682'],
    ['6', 'Municipality employment rate', '<= 0.291', '(0.291; 0.378)', '> 0.378'],
    ['7', 'Unemployment rate', '> 122.0', '(77.0; 122.0)', '<= 77.0'],
    ['8', 'Active companies rate', '<= 7.755', '(7.755; 10.985)', '> 10.985'],
    ['9', 'Transport and storage investments rate', '< 0.562', '= 0.562', '> 0.562'],
    ['10', 'Doctors accessibility', '<= 1.3', '(1.3; 2.1)', '> 2.1'], 
    ['11', 'Tourist Arrivals', '<= 978.0', '(978.0; 7629.333)', '> 7629.333'], 
    ['12', 'Gross Value Added per capita (pooled)', '<= -0.25', '(-0.25; 0]', '> 0']]  

    df = pd.DataFrame(df, columns = ['RN', 'Attribute Name', 'Poor', 'Medium', 'Good'])
    df = df.set_index('RN')

    coldict = {'Poor':'red', 'Good':'lightgreen'}

    def highlight_cols(s, coldict):
        if s.name in coldict.keys():
            return ['background-color: {}'.format(coldict[s.name])] * len(s)
        return [''] * len(s)

    st.table(df.style.apply(highlight_cols, coldict=coldict))

def decision_table():
    st.subheader('Decision Rules')

    st.markdown(
        '''
        Once the discretization is done, the following decision rules were conducted to obtain the explanation of the **Gross Value Added**. Please find all decision tables below:
        '''
    )

    def highlight(x):
        color = ''
        if 'Poor' in x:
            color = 'red'
        elif 'Good' in x:
            color = 'lightgreen'
        return f'background-color : {color}'

    atts = ['Local Traffic', 'General Traffic', 'Road and Traffic', 'Tourism and Traffic', 'Social Factors', 'Employment State', 'Economy State', 'Economy and Investments', 'Gross Value Added per capita']
    selected_att = st.selectbox(label='Please select the attribute you want to see', options=atts, index=0)

    if selected_att == 'Local Traffic':
        df_1 = [['Poor', '*', 'Poor'], ['<= Medium', 'Poor', 'Poor'], ['Medium', '>= Medium', 'Medium'], ['>= Medium', 'Medium', 'Medium'], ['Good', '<= Medium', 'Medium'], ['Good', 'Good', 'Good']]
        df_1 = pd.DataFrame(df_1, columns = ['Local roads density', 'Motorcycles density', 'Local Traffic'])

        st.table(df_1.style.applymap(lambda x: highlight(x)))

    if selected_att == 'General Traffic':
        df_2 = [['<= Medium', '<= Medium', 'Poor'], ['*', 'Poor', 'Poor'], ['<= Medium', 'Good', 'Medium'], ['Good', 'Medium', 'Medium'], ['Good', 'Good', 'Good']]
        df_2 = pd.DataFrame(df_2, columns = ['Vehicles density', 'Main road accessibility', 'General Traffic'])

        st.table(df_2.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Road and Traffic':
        df_3 = [['Poor', '*', 'Poor'], ['<= Medium', 'Poor', 'Poor'], ['Medium', '>= Medium', 'Medium'], ['Good', '*', 'Good']]
        df_3 = pd.DataFrame(df_3, columns = ['Local Traffic', 'General Traffic', 'Road and Traffic'])

        st.table(df_3.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Tourism and Traffic':
        df_4 = [['Poor', '<= Medium', 'Poor'], ['<= Medium', 'Poor', 'Poor'], ['Poor:Medium', 'Medium', 'Medium'], ['*', 'Good', 'Good'], ['Good', '*', 'Good']]
        df_4 = pd.DataFrame(df_4, columns = ['Tourists Arrivals', 'Road and Traffic', 'Tourism and Traffic'])

        st.table(df_4.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Social Factors':
        df_5 = [['Poor', '*', 'Poor'], ['Medium', '<= Good', 'Medium'], ['>= Medium', 'Good', 'Good'], ['Good', '*', 'Good']]
        df_5 = pd.DataFrame(df_5, columns = ['Preschool children enrollment rate', 'Doctors accessibility', 'Social Factors'])

        st.table(df_5.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Employment State':
        df_6 = [['Poor', '*', 'Poor'], ['Medium', '<= Good', 'Poor'], ['Medium', '>= Medium', 'Medium'], ['>= Medium', 'Medium', 'Medium'], ['Good', 'Good', 'Good']]
        df_6 = pd.DataFrame(df_6, columns = ['Municipality employment rate', 'Unemployed rate', 'Employment State'])

        st.table(df_6.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Economy State':
        df_7 = [['Poor', '<= Good', 'Poor'], ['Medium', '<= Medium', 'Medium'], ['>= Medium', 'Poor', 'Medium'], ['*', 'Good', 'Good'], ['Good', '>= Medium', 'Good']]
        df_7 = pd.DataFrame(df_7, columns = ['Employment State', 'Active companies rate', 'Economy State'])

        st.table(df_7.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Economy and Investments':
        df_8 = [['<= Medium', '<= Medium', 'Poor'], ['*', 'Good', 'Medium'], ['Good', 'Poor', 'Medium'], ['Good', '>= Medium', 'Good']]
        df_8 = pd.DataFrame(df_8, columns = ['Economy State', 'Transport and storage investments rate', 'Economy and Investments'])

        st.table(df_8.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Gross Value Added per capita':
        df_9 = [['Poor', 'Poor', '<= Medium', 'Poor'], ['Poor', '*', 'Poor', 'Poor'], ['*', 'Poor', 'Poor', 'Poor'], ['Poor', '<= Medium', 'Medium', 'Poor'], ['<= Medium', 'Poor', 'Good', 'Medium'], ['Poor', 'Medium', '>= Medium', 'Medium'], ['*', 'Medium', '>= Medium', 'Medium'], ['Medium', 'Poor', '>= Medium', 'Medium'], ['Medium', '*', 'Medium', 'Medium'], ['>= Medium', '<= Medium', 'Medium', 'Medium'], ['Medium', '>= Medium', '<= Medium', 'Medium'], ['>= Medium', '>= Medium', 'Poor', 'Medium'], ['*', 'Good', 'Good', 'Good'], ['>= Medium', '>= Medium', 'Good', 'Good'], ['Good', '*', 'Good', 'Good'], ['Good', 'Good', '>= Medium', 'Good']]
        df_9 = pd.DataFrame(df_9, columns = ['Tourism and Traffic', 'Social Factors', 'Economy and Investments', 'Gross Value Added per capita'])

        st.table(df_9.style.applymap(lambda x: highlight(x)))

def data_exploration_text():
    st.markdown(
        '''
        This page aims to inspect the data for the Gross Value Added per capita.

        Below you can find multi-select area where you can select up to ten municipalities and up to ten years. Selection of municipalities and time periods impact the table below. Values in the table are denoted with red or green background depending on whether that particular value is in the category *Poor* (red color) or *Good* (green color). 
        '''
    )

def data_exploration():
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

def visualize_data():
    with st.expander('Information'):
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

    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    for i in range(2,len(df_gva_s.columns)):
        df_gva_s.iloc[:, i] = sigmoid((df_gva_s.iloc[:, i] - df_gva_s.iloc[:, i].mean())/df_gva_s.iloc[:, i].std())

    gj = gpd.read_file('util/Opstina_l.geojson')
    df_gva_c = df_gva.copy()
    df_gva_c['Area Name'] = df_gva_c['Area Name'].str.upper()

    LSGs = np.unique(df_gva['Area Name'])
    years = np.unique(df_gva['Time Period'])
    attributes = df_gva.drop(['Area Name', 'Time Period'], axis=1).columns

    selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data:', LSGs, max_selections=10)

    st.markdown("---")

    st.subheader('Comparison for a selected year')

    with st.expander('Description'):
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

        fig.add_trace(go.Scatterpolar(
            r=np.repeat(0.5, len(df_gva_s_y.drop(['Time Period', 'Area Name'], axis=1).columns)),
            theta=df_gva_s_y.drop(['Time Period', 'Area Name'], axis=1).columns.to_numpy(),
            fill='toself',
            name='Serbia - Average'
        ))

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

def historical_info():
    df_gva = pd.read_csv('data/gva.csv')
    df_gva = df_gva.loc[df_gva['GVA Per Capita Normalized'].notna(), :]

    df_gva['Main road accessibility'] = pd.cut(df_gva['Main road accessibility'], bins=[0, 5.461, 13.025, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Local roads density'] = pd.cut(df_gva['Local roads density'], bins=[0, 0.213, 0.382, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Motorcycles density'] = pd.cut(df_gva['Motorcycles density'], bins=[0, 0.281, 0.762, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Vehicles density'] = pd.cut(df_gva['Vehicles density'], bins=[0, 10.272, 21.286, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Active companies rate'] = pd.cut(df_gva['Active companies rate'], bins=[0, 7.755, 10.985, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Municipality employment rate'] = pd.cut(df_gva['Municipality employment rate'], bins=[0, 0.291, 0.378, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Unemployed rate'] = pd.cut(df_gva['Unemployed rate'], bins=[0, 77.0, 122.0, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Transport and storage investments rate'] = pd.cut(df_gva['Transport and storage investments rate'], bins=[-222220, 0, 0.562, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Doctors accessibility'] = pd.cut(df_gva['Doctors accessibility'], bins=[0, 1.3, 2.1, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Preschool children enrollment rate'] = pd.cut(df_gva['Preschool children enrollment rate'], bins=[0, 2.086, 2.682, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['Tourists Arrivals'] = pd.cut(df_gva['Tourists Arrivals'], bins=[0, 978.0, 7629.333, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva['GVA Per Capita Normalized Category'] = pd.cut(df_gva['GVA Per Capita Normalized'], bins=[-500000000, -0.25, 0, 99999999], labels=['1', '2', '3']).astype(str)
    df_gva = df_gva.replace({'nan': 'U'})

    LSGs = np.unique(df_gva['Area Name'])
    years = np.unique(df_gva['Time Period'])

    selected_lsg = st.selectbox(label='Please select municipality', options=LSGs, index=79)
    selected_year = st.selectbox(label='Please select year', options=years, index=9)

    df_mun = df_gva.loc[(df_gva['Area Name'] == selected_lsg) & (df_gva['Time Period'] == selected_year), :]

    st.markdown('---')
    st.subheader('Values')

    st.table(df_mun.head(10))

    st.markdown('---')
    st.subheader('Prediction Outcome')

    if df_mun.shape[0] == 0:
        st.markdown("We don't have data for the selected municipality and the selected year")
    else:
        pred = dm.d.predict(df_mun, return_intermediate=True)

        col11, col21 = st.columns(2)
        col12, col22 = st.columns(2)
        col13, col23 = st.columns(2)
        col14, col24 = st.columns(2)
        col15, col25 = st.columns(2)
        col16, col26 = st.columns(2)
        col17, col27 = st.columns(2)
        col18, col28 = st.columns(2)
        col19, col29 = st.columns(2)
        col110, col210 = st.columns(2)
        col111, col211 = st.columns(2)
        col112, col212 = st.columns(2)
        col113, col213 = st.columns(2)
        col114, col214 = st.columns(2)
        col115, col215 = st.columns(2)
        col116, col216 = st.columns(2)
        col117, col217 = st.columns(2)
        col118, col218 = st.columns(2)

        with col11:
            st.markdown('Local roads density')
        with col12:
            st.markdown('Motorcycles density')
        with col13:
            st.markdown('&emsp;Local roads density and Motorcycle Density lead to - **Local Traffic**')
        with col14:
            st.markdown('Vehicles density')
        with col15:
            st.markdown('Main road accessibility')
        with col16:
            st.markdown('&emsp;Vehicles density and Main road accessibility lead to - **General Traffic**')
        with col17:
            st.markdown('Tourists Arrivals')
        with col18:
            st.markdown('&emsp;&emsp;Tourists Arrivals and General Traffic lead to - **Tourism and Traffic**')
        with col19:
            st.markdown('Preschool children enrollment rate')
        with col110:
            st.markdown('Doctors accessibility')
        with col111:
            st.markdown('&emsp;Preschool children enrollment rate and Doctors accessibility lead to **Social Factors**')
        with col112:
            st.markdown('Municipality employment rate')
        with col113:
            st.markdown('Unemployed rate')
        with col114:
            st.markdown('&emsp;Municipality employment rate and Unemployed rate lead to - **Employment State**')
        with col115:
            st.markdown('Active companies rate')
        with col116:
            st.markdown('&emsp;&emsp;Active companies rate and Employment State lead to - **Economy State**')
        with col117:
            st.markdown('Transport and storage investments rate')
        with col118:
            st.markdown('&emsp;&emsp;&emsp;Economy State and Transport and storage investments rate lead to - **Economy and Investments**')

        with col21:
            annotated_text((pred['Local roads density'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'})[0], '', pred['Local roads density'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
        with col22:
            annotated_text((pred['Motorcycles density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Motorcycles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col23:
            annotated_text((pred['Local Traffic'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Local Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col24:
            annotated_text((pred['Vehicles density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Vehicles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col25:
            annotated_text((pred['Main road accessibility'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'})[0], '', pred['Main road accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
        with col26:
            annotated_text((pred['General Traffic'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['General Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col27:
            annotated_text((pred['Tourists Arrivals'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Tourists Arrivals'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col28:
            annotated_text((pred['Tourism and Traffic'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Tourism and Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col29:
            annotated_text((pred['Preschool children enrollment rate'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Preschool children enrollment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col210:
            annotated_text((pred['Doctors accessibility'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Doctors accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col211:
            annotated_text((pred['Social Factors'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col212:
            annotated_text((pred['Municipality employment rate'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Municipality employment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col213:
            annotated_text((pred['Unemployed rate'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'})[0], '', pred['Unemployed rate'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
        with col214:
            annotated_text((pred['Employment State'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Employment State'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col215:
            annotated_text((pred['Active companies rate'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Active companies rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col216:
            annotated_text((pred['Economy State'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Economy State'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col217:
            annotated_text((pred['Transport and storage investments rate'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Transport and storage investments rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col218:
            annotated_text((pred['Economy and Investments'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Economy and Investments'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))

        st.markdown('**The predicted outcome is:**')
        prediction = dm.d.predict(df_mun).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction == 'Poor':
            annotated_text((prediction, '', 'red'))
            st.markdown('This model signals the **Gross Value Added per capita** lower by 25% than the average of Serbia')
        elif prediction == 'Good':
            annotated_text((prediction, '', 'lightgreen'))
            st.markdown('This model signals the **Gross Value Added per capita** higher than the average of Serbia')
        else:
            annotated_text((prediction, '', 'gray'))
            st.markdown('This model signals the **Gross Value Added per capita** lower, but close to the average of  Serbia')

        st.markdown('Please find below the projected outcome and the potential interventions:')

        st.markdown('---')
        st.subheader('Possible Interventions')
        st.markdown("This part of the page discuss the potential possibility of interventions on Gross Value Added per capita. We show potential interventions by changing the values of input attributes for one degree in improvement and one degree in deterioration. If a single intervention improves or deteriorates the outcome it will be presented in an appropriate color - red for the low Gross Value added per capita (-0.25 or lower), gray for negative (between -0.25 and 0), and green (greater than 0).")

        ### Tourists Arrivals
        to_write = True
        if (df_mun['Tourists Arrivals'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 1]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) - 1
            df_mun_v['Tourists Arrivals'] = dm.d._levels[0][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Tourists Arrivals* to a value between 978.0 and 7629.333 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Tourists Arrivals'].values[0] == dm.d._levels[0][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) + 1
            df_mun_v['Tourists Arrivals'] = dm.d._levels[0][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Tourists Arrivals* to a value between 978.0 and 7629.333 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) + 1
            down_v = dm.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) - 1

            df_mun_v['Tourists Arrivals'] = dm.d._levels[0][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Tourists Arrivals* to a value lower than 978.0 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Tourists Arrivals'] = dm.d._levels[0][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Tourists Arrivals* to a value greater than 7629.333 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Local roads density
        to_write = True
        if (df_mun['Local roads density'].values[0] == dm.d._levels[1][1][len(dm.d._levels[1][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[1][1].index(df_mun['Local roads density'].values[0]) - 1
            df_mun_v['Local roads density'] = dm.d._levels[1][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('The **deterioration** of *Local roads density* to a value between 0.213 and 0.382 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Local roads density'].values[0] == dm.d._levels[1][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[1][1].index(df_mun['Local roads density'].values[0]) + 1
            df_mun_v['Local roads density'] = dm.d._levels[1][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Local roads density* to a value between 0.213 and 0.382 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[1][1].index(df_mun['Local roads density'].values[0]) + 1
            down_v = dm.d._levels[1][1].index(df_mun['Local roads density'].values[0]) - 1

            df_mun_v['Local roads density'] = dm.d._levels[1][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('The **deterioration** in *Local roads density* to a value greater than 0.382 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Local roads density'] = dm.d._levels[1][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Local roads density* to a value lower than 0.213 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        ### Motorcycles density
        to_write = True
        if (df_mun['Motorcycles density'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) - 2
            df_mun_v['Motorcycles density'] = dm.d._levels[2][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Motorcycles density* to a value between 0.281 and 0.762 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Motorcycles density'].values[0] == dm.d._levels[2][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) + 1
            df_mun_v['Motorcycles density'] = dm.d._levels[2][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Motorcycles density* to a value between 0.281 and 0.762 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) + 1
            down_v = dm.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) - 1

            df_mun_v['Motorcycles density'] = dm.d._levels[2][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Motorcycles density* to a value lower than 0.281 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Motorcycles density'] = dm.d._levels[2][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Motorcycles density* to a value greater than 0.762 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Vehicles density
        to_write = True
        if (df_mun['Vehicles density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 1]) | (df_mun['Vehicles density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) - 1
            df_mun_v['Vehicles density'] = dm.d._levels[3][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Vehicles density* to a value between 10.272 and 21.286 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Vehicles density'].values[0] == dm.d._levels[3][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) + 1
            df_mun_v['Vehicles density'] = dm.d._levels[3][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Vehicles density* to a value between 10.272 and 21.286 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) + 1
            down_v = dm.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) - 1

            df_mun_v['Vehicles density'] = dm.d._levels[3][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Vehicles density* to a value greater than 21.286 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Vehicles density'] = dm.d._levels[3][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Vehicles density* to a value lower than 10.272 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Main road accessibility
        to_write = True
        if (df_mun['Main road accessibility'].values[0] == dm.d._levels[4][1][len(dm.d._levels[4][1]) - 1]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) - 1
            df_mun_v['Main road accessibility'] = dm.d._levels[4][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Main road accessibility* to a value between 5.461 and 13.025 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Main road accessibility'].values[0] == dm.d._levels[4][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) + 1
            df_mun_v['Main road accessibility'] = dm.d._levels[4][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Main road accessibility* to a value between 5.461 and 13.025 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) + 1
            down_v = dm.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) - 1

            df_mun_v['Main road accessibility'] = dm.d._levels[4][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Main road accessibility* to a value greater than 13.025 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Main road accessibility'] = dm.d._levels[4][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Main road accessibility* to a value lower than 5.461 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Municipality employment rate
        to_write = True
        if (df_mun['Municipality employment rate'].values[0] == dm.d._levels[5][1][len(dm.d._levels[5][1]) - 1]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) - 1
            df_mun_v['Municipality employment rate'] = dm.d._levels[5][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Municipality employment rate* to a value between 0.291 and 0.378 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Municipality employment rate'].values[0] == dm.d._levels[5][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) + 1
            df_mun_v['Municipality employment rate'] = dm.d._levels[5][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Municipality employment rate* to a value between 0.291 and 0.378 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) + 1
            down_v = dm.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) - 1

            df_mun_v['Municipality employment rate'] = dm.d._levels[5][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Municipality employment rate* to a value lower than 0.291 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Municipality employment rate'] = dm.d._levels[5][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Municipality employment rate* to a value greater than 0.378 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Unemployed rate
        to_write = True
        if (df_mun['Unemployed rate'].values[0] == dm.d._levels[6][1][len(dm.d._levels[6][1]) - 1]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) - 1
            df_mun_v['Unemployed rate'] = dm.d._levels[6][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Unemployed rate* to a value between 77.0 and 122.0 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if (df_mun['Unemployed rate'].values[0] == dm.d._levels[6][1][1]):
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) + 1
            df_mun_v['Unemployed rate'] = dm.d._levels[6][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Unemployed rate* to a value between 77.0 and 122.0 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) + 1
            down_v = dm.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) - 1

            df_mun_v['Unemployed rate'] = dm.d._levels[6][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Unemployed rate* to a value greater than 122.0 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) + 1
            df_mun_v['Unemployed rate'] = dm.d._levels[6][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Unemployed rate* to a value lower than 77.0 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Active companies rate
        to_write = True
        if (df_mun['Active companies rate'].values[0] == dm.d._levels[6][1][len(dm.d._levels[7][1]) - 1]) | (df_mun['Active companies rate'].values[0] == dm.d._levels[6][1][len(dm.d._levels[7][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) - 1
            df_mun_v['Active companies rate'] = dm.d._levels[7][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Active companies rate* to a value between 7.755 and 10.985 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Active companies rate'].values[0] == dm.d._levels[7][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) + 1
            df_mun_v['Active companies rate'] = dm.d._levels[7][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Active companies rate* to a value between 7.755 and 10.985 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) + 1
            down_v = dm.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) - 1

            df_mun_v['Active companies rate'] = dm.d._levels[7][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Active companies rate* to a value lower than 7.755 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Active companies rate'] = dm.d._levels[7][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Active companies rate* to a value lower than 10.985 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Transport and storage investments rate
        to_write = True
        if (df_mun['Transport and storage investments rate'].values[0] == dm.d._levels[8][1][len(dm.d._levels[8][1]) - 1]) | (df_mun['Transport and storage investments rate'].values[0] == dm.d._levels[8][1][len(dm.d._levels[8][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) - 1
            df_mun_v['Transport and storage investments rate'] = dm.d._levels[8][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Transport and storage investments rate* to a value 0.562 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Transport and storage investments rate'].values[0] == dm.d._levels[8][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) + 1
            df_mun_v['Transport and storage investments rate'] = dm.d._levels[8][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Transport and storage investments rate* to a value 0.562 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) + 1
            down_v = dm.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) - 1

            df_mun_v['Transport and storage investments rate'] = dm.d._levels[8][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Transport and storage investments rate* to a value lower than 0.562 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Transport and storage investments rate'] = dm.d._levels[8][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Transport and storage investments rate* to a value greater than 0.562 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Preschool children enrollment rate
        to_write = True
        if (df_mun['Preschool children enrollment rate'].values[0] == dm.d._levels[9][1][len(dm.d._levels[9][1]) - 1]) | (df_mun['Preschool children enrollment rate'].values[0] == dm.d._levels[9][1][len(dm.d._levels[9][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) - 1
            df_mun_v['Preschool children enrollment rate'] = dm.d._levels[9][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Preschool children enrollment rate* to a value between 2.086 and 2.682 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Preschool children enrollment rate'].values[0] == dm.d._levels[9][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) + 1
            df_mun_v['Preschool children enrollment rate'] = dm.d._levels[9][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Preschool children enrollment rate* to a value between 2.086 and 2.682 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) + 1
            down_v = dm.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) - 1

            df_mun_v['Preschool children enrollment rate'] = dm.d._levels[9][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Preschool children enrollment rate* to a value lower than 2.086 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Preschool children enrollment rate'] = dm.d._levels[9][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Preschool children enrollment rate* to a value lower than 2.682 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Doctors accessibility
        to_write = True
        if (df_mun['Doctors accessibility'].values[0] == dm.d._levels[10][1][len(dm.d._levels[10][1]) - 1]) | (df_mun['Doctors accessibility'].values[0] == dm.d._levels[10][1][len(dm.d._levels[10][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) - 1
            df_mun_v['Doctors accessibility'] = dm.d._levels[10][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Doctors accessibility* to a value between 1.3 and 2.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Doctors accessibility'].values[0] == dm.d._levels[10][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) + 1
            df_mun_v['Doctors accessibility'] = dm.d._levels[10][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Doctors accessibility* to a value between 1.3 and 2.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) + 1
            down_v = dm.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) - 1

            df_mun_v['Doctors accessibility'] = dm.d._levels[10][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Doctors accessibility* to a value lower than 1.3 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Doctors accessibility'] = dm.d._levels[10][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Doctors accessibility* to a value lower than 2.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
