import streamlit as st

import pandas as pd
import numpy as np
import geopandas as gpd

import dex_model as dm

import plotly.express as px
import plotly.graph_objects as go

from annotated_text import annotated_text

def introduction_text():
    st.markdown(
        '''
        The final DIDEX extracted DEX model for internal migration management is shown below. **NOTE: This is our best effort having in mind the available data, constraints in data acquisition, the validity and accuracy of data, as well as the usability of the solution. The proposed model is intentially created to work with categories as this human decision makers are more prone to use qualitative values instead of exact numbers. Another note is that the accuracy of the predictive model is around 65% (using yearly based cross-validation) and this result is comparable to the results obtained using more complex machine learning models (that are not understandable to the human and that do not have an option to provide a list of possible policy interventions).**
        
        As one can see, **internal net migrations** are described using **Traffic** and **Social Determinants** factors. 

        Traffic conditions and transportation infrastructure play a pivotal role in internal migration predictions. Regions with well-developed and efficient transportation networks are more likely to attract people. Access to job opportunities, educational institutions, and healthcare facilities is often determined by traffic ease and connectivity. Regions with congested traffic, limited public transportation, or inadequate road networks may experience outward migration as people seek better access to essential services and improved quality of life elsewhere. 

        Access to quality education is also a fundamental driver of internal migration. Families often relocate to areas with better school opportunities to provide their children with better educational opportunities. Regions with diverse and well-performing schools tend to attract migrants, leading to net inward migration. Conversely, areas with underfunded or struggling education may experience outward migration as people seek better services elsewhere. Social factors, such as **poverty share** and **assistance and care allowance share**, can also influence internal migration patterns. People often migrate to areas where they feel a sense of belonging and acceptance. Communities that have high poverty share and are in the need for care are often struggling and experiencing outward migrations. Additionally, factors like crime rates, social amenities, and recreational opportunities can impact migration decisions. Regions that provide a safe, vibrant, and fulfilling social environment are more likely to experience net inward migration.
        '''
    )

def model_hierarchy():
    st.markdown(
    '''
    - Internal Net Migrations
        - Traffic
            - Vehicles
                - Vehicles density
                - Motorcycle density
            - Roads
                - Main roads accessibility
                - Local roads density
        - Social Determinants
            - School
                - Primary school attendance
                - Secondary school attendance
            - Health and Social Factors
                - Social Factors
                    - Assistance and care allowance share
                    - Poverty share
                - Doctors accessibility
    '''
    )

    st.markdown(
        '''
        The DEX model merged the nine attributes from data into new extracted features. 
    - The attributes **Vehicles density** and the **Motorcycles density** make decisions about the feature **Vehicles**.
    - The attributes **Main roads accessibility** and **Local roads density** make decisions about the feature **Roads**.
    - The attributes **Primary school's attendance** and **Secondary school's attendance** make decisions about the feature **School**.
    - The attributes **Assistance and care allowance share** and **Poverty share** make decisions about the feature **Social Factors**.
    - The decision making process to make decisions about the **Net Internal Migrations** is as follows:
        - The attribute **Doctors accessibility** and the feature **Social Factors** make the decision about the feature **Health and Social Factors**
        - In parallel: 
            - The factors **Health and Social Factors** and **School** make the decision about the feature **Social Determinants**.
            - The factors **Roads** and **Vehicles** make the decision about the feature **Traffic**.
        - The factors **Social Determinants** and **Traffic** make the final decision about the **Internal Net Migrations**.

        '''
    )

def attribute_description():
    data = {
        'Attribute Name': ['Vehicles density', 'Motorcycles density', 'Main roads accessibility', 
                        'Local roads density', 'Primary schools attendance', 'Secondary schools attendance', 
                        'Assistance and care allowance share', 'Poverty share', 'Doctors accessibility'],
        'Description': ['The number of registered passenger vehicles per square meter of the municipality.', 
                        'The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.', 
                        'The total length (in kilometers) of 1st-class road segments without highways divided by 1,000 inhabitants of the municipality.', 
                        'The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of the municipality.',
                        'The average number of children in primary school that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 7 and 14.', 
                        'The average number of children in secondary school (both vocational and general) that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 15 and 18.', 
                        'Percentage of people within one municipality that use an increased allowance for assistance and care of another person.', 
                        'Percentage of people within one municipality that uses social protection.',
                        'The total number of doctors within a municipality divided by the 1,000 inhabitants.'],
        'Tooltip': ['**Vehicles density** represents the number of registered passenger vehicles per square meter of the municipality. According to the European driving license classification, passenger vehicles are four-wheel vehicles that have one or more passenger spaces but weigh less than 3,500 kilograms. This information is collected from the Ministry of Internal Affairs, and it represents the total number of passenger vehicles registered in an LSG. Further, to normalize the total number of vehicles to a common measure, we divided the total number of passenger vehicles with the size of the municipality leading to information about the number of passenger vehicles per square kilometer.', 
                    'Attribute **Motorcycles density** has a pattern of positive correlation between this attribute and net migrations, meaning that greater motorcycle density corresponds to higher immigration in the municipality. In other words, it carries similar information as vehicle density. However, this attribute has a better discrimination power for high emigration LSGs than the number of passenger vehicles per square meter. Having the interpretations above, a combination of these two attributes would be a good discriminator for internal migrations.', 
                    'Interestingly, the **Main roads accessibility** has a negative correlation with net migrations. In other words, the higher the net migrations lower the value for the main roads accessibility, as well as when net migrations are highly negative and main roads accessibility are very high. This occurrence may be explained by the normalization used. More specifically, areas that are highly populated and have positive migration have their main road segments (category I road segments without highways) lower due to higher value in the denominator. Conversely, municipalities with a low population have a very low denominator, which leads to a high number of total length of main roads accessibility. It is worth noting that we ensured that the confounding effect between the aforementioned attributes is minimized as net migration was also calculated using the same normalization.', 
                    'On the contrary, if we use **local road density** we get a positive correlation with net migrations. The signal is not as strong as with the number of passenger vehicles or the number of motors and motorcycles, but municipalities with a high level of local roads density have a high net migration.', 
                    'By observing the relation between **Primary schools attendance** and net migrations, we notied that LSGs with a low number of pupils per primary school have negative migrations. This is a common finding in Serbia, as depopulation and internal migration leave schools with fewer pupils. These municipalities simply have more schools than needed (however, this is a different topic compared to the one in this project and requires a deeper discussion). On the other side of the spectrum, LSGs with a high number of pupils are associated with positive migrations.', 
                    'A much crisper signal is observable for the **Secondary schools attendance**. As there are municipalities without any type of secondary school, those that do have secondary schools act as attractors. This is clearly observable for high net migrations.', 
                    'Attribute **Assistance and care allowance share** is the first one on the list. We can state that the relationship with net migrations is negative. In other words, higher the share of assistance and care allowance, lower the net migrations.', 
                    'Similarly, the attribute **Poverty share** has a negative correlation with net migrations.', 
                    'Finally, the **doctors accessibility** is an attribute that represents the availability of medical services in the municipality. Compared to every other attribute in this group, it does not have a perfectly linear correlation with net migrations. It is having a U-shape relationship In areas with very low and with very high doctors accessibility there is a high net migration. However, in areas with medium accessibility to doctors, net migrations are the lowest.']
    }

    df_text = pd.DataFrame(data)

    def add_tooltip(row):
        return f'<span title="{row["Tooltip"]}">{row["Attribute Name"]}</span>'

    df_text['Attribute Name'] = df_text.apply(add_tooltip, axis=1)
    st.write(df_text[['Attribute Name', 'Description']].to_html(escape=False), unsafe_allow_html=True)

def attribute_values():
    st.markdown(
        '''
        Since DEX models are working with qualitative data, we discretized data according to the following values:
        '''
    )

    df = [['1', 'Vehicles Density', '<= 10.272', '(10.272; 21.286)', '> 21.286'], 
    ['2', 'Motorcycle Density', '<= 0.281', '(0.281; 0.762)', '> 0.762'],
    ['3', 'Main Road Accessibility', '> 1.676', '(0.693; 1.676)', '<= 0.693'],
    ['4', 'Local Road Density', '<= 107.736', '(107.736; 450.854)', '> 450.854'],
    ['5', 'Primary School Attendance', '<= 365.412', '(365.412; 590.641)', '> 590.641'],
    ['6', 'Secondary School Attendance', '<= 269.804', '(269.804; 435.033)', '> 435.033'],
    ['7', 'Assistance and Care Allowance Share', '> 0.6', '(0.5; 0.6)', '<= 0.5'],
    ['8', 'Poverty Share', '> 12.9', '(8.1; 12.9)', '<= 8.1'],
    ['9', 'Doctors Accessibility', '<= 1.3', '(1.3; 2.1)', '> 2.1'],
    ['10', 'Internal Net Migrations (per 1,000 inhabitants)', '<= -4', '(-4; 0)', '> 0']]

    df = pd.DataFrame(df, columns = ['RN', 'Attribute Name', 'Poor', 'Medium', 'Good'])
    df = df.set_index('RN')

    coldict = {'Poor':'red', 'Good':'lightgreen'}

    def highlight_cols(s, coldict):
        if s.name in coldict.keys():
            return ['background-color: {}'.format(coldict[s.name])] * len(s)
        return [''] * len(s)

    st.table(df.style.apply(highlight_cols, coldict=coldict))

def decision_rules():
    st.markdown(
        '''
        Once the discretization is done, the following decision rules were conducted to obtain the explanation of the **Net Internal Migrations**. Please find all decision tables below:
        '''
    )

    def highlight(x):
        color = ''
        if 'Poor' in x:
            color = 'red'
        elif 'Good' in x:
            color = 'lightgreen'
        return f'background-color : {color}'

    atts = ['Health and Social Factors', 'Social Factors', 'Social Determinants', 'School', 'Traffic', 'Vehicles', 'Roads', 'Internal Net Migrations']
    selected_att = st.selectbox(label='Please select the attribute you want to see', options=atts, index=0)

    if selected_att == 'Health and Social Factors':
        df_1 = [['Poor', '*', 'Poor'], ['Medium', '<= Good', 'Medium'], ['>= Medium', 'Medium', 'Good'], ['Good', '*', 'Good']]
        df_1 = pd.DataFrame(df_1, columns = ['Social factors', 'Doctors Accessibility', 'Health and Social Factors'])

        st.table(df_1.style.applymap(lambda x: highlight(x)))

        st.markdown(
            '''
            The table below can be read as follows: If **Social Factors** are *Poor* and whatever the value for the attribute **Doctors Accessibility** the value of factor **Health and Social Factors** will be Poor. If **Social Factors** is *Medium* and if the value of **Doctors Accessibility** is *Good* or lower than that value (i.e. *Medium* or *Poor*) then the factor **Health and Social Factors** will be Medium. A *Good* value of the resulting factor is achieved in two cases. When the factor **Social Factors** is *Medium* or *Good*, and the attribute **Doctors Accessibility** is *Medium*, or when the factor **Social Factors** is *Good*.
            '''
        )

    if selected_att == 'Social Factors':
        df_2 = [['Poor', '<= Medium', 'Poor'], ['<= Medium', '>= Good', 'Medium'], ['Medium', '*', 'Medium'], ['>= Medium', 'Poor', 'Medium'], ['>= Good', '>= Medium', 'Good']]
        df_2 = pd.DataFrame(df_2, columns = ['Assistance and care allowance share', 'Poverty share', 'Social Factors'])

        st.table(df_2.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Social Determinants':
        df_3 = [['Poor', '*', 'Poor'], ['<= Medium', 'Poor', 'Poor'], ['Medium', '>= Medium', 'Medium'], ['Good', '*', 'Good']]
        df_3 = pd.DataFrame(df_3, columns = ['School', 'Health and Social Factors', 'Social Determinants'])

        st.table(df_3.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'School':
        df_4 = [['<= Medium', 'Poor', 'Poor'], ['<= Medium', '>= Medium', 'Medium'], ['*', 'Medium', 'Medium'], ['Good', '<= Medium', 'Medium'], ['Good', 'Good', 'Good']]
        df_4 = pd.DataFrame(df_4, columns = ['Primary schools attendance', 'Secondary schools attendance', 'School'])

        st.table(df_4.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Traffic':
        df_5 = [['Poor', '*', 'Poor'], ['Medium', '*', 'Medium'], ['>= Medium', 'Poor', 'Medium'], ['Good', '>= Medium', 'Good']]
        df_5 = pd.DataFrame(df_5, columns = ['Vehicles', 'Roads', 'Traffic'])

        st.table(df_5.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Vehicles':
        df_6 = [['<= Medium', 'Poor', 'Poor'], ['<= Medium', '>= Medium', 'Medium'], ['>= Good', '*', 'Good']]
        df_6 = pd.DataFrame(df_6, columns = ['Vehicles Density', 'Motorcycles Density', 'Vehicles'])

        st.table(df_6.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Roads':
        df_7 = [['Poor', '*', 'Poor'], ['Medium', '*', 'Medium'], ['Medium', '<= Medium', 'Medium'], ['>= Medium', 'Poor', 'Medium'], ['>= Medium', 'Good', 'Good'], ['Good', '>= Medium', 'Good']]
        df_7 = pd.DataFrame(df_7, columns = ['Main Road Accessibility', 'Local Roads Density', 'Roads'])

        st.table(df_7.style.applymap(lambda x: highlight(x)))

    elif selected_att == 'Internal Net Migrations':
        df_8 = [['Poor', 'Poor', 'Poor'], ['<= Medium', '>= Medium', 'Medium'], ['*', 'Medium', 'Medium'], ['Medium', '*', 'Medium'], ['>= Medium', '<= Medium', 'Medium'], ['Good', 'Good', 'Good']]
        df_8 = pd.DataFrame(df_8, columns = ['Traffic', 'Social Determinants', 'Internal Net Migrations'])

        st.table(df_8.style.applymap(lambda x: highlight(x)))

def data_exploration():
    with st.expander('Information'):
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

def data_visualization():
    with st.expander('Information'):
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

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    for i in range(2,len(df_im_s.columns)):
        df_im_s.iloc[:, i] = sigmoid((df_im_s.iloc[:, i] - df_im_s.iloc[:, i].mean())/df_im_s.iloc[:, i].std())

    gj = gpd.read_file('util/Opstina_l.geojson')
    df_im_c = df_im.copy()
    df_im_c['Area Name'] = df_im_c['Area Name'].str.upper()

    LSGs = np.unique(df_im['Area Name'])
    years = np.unique(df_im['Time Period'])
    attributes = df_im.drop(['Area Name', 'Time Period'], axis=1).columns

    selected_lsgs = st.multiselect('Please select municipalities for which you would want to observe the data: ', LSGs, max_selections=10)

    st.markdown("---")

    st.subheader('Comparison for a selected year')

    with st.expander('Description'):
        st.markdown('The first chart for comparison is a radar chart where one can observe how selected municipalities compare between each other. By default, radar chart will not show up due to having too many municipalities.')
        st.markdown('NOTE: To make values comparable on the radar chart, we scaled values so that the highest value is one for each year. In addition, we inverted the values where the lower value is better, thus, 1 is always the best value and 0 is always the worst.')
        st.markdown('Regarding the *Net Migrations per 1000 inhabitants* values are scaled with max-min normalization technique, which means that zero is the worst value and one is the best value.')

    selected_year = st.selectbox(label='Please select the year of observation', options=years, index=9)

    fig = go.Figure()

    if selected_lsgs != []:
        df_im_s_y = df_im_s.copy()
        if selected_lsgs != []:
            df_im_s_y = df_im_s.loc[df_im_s_y['Area Name'].isin(selected_lsgs), :]
        df_im_s_y = df_im_s_y.loc[df_im_s_y['Time Period'] == selected_year, :]


        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
                r=np.repeat(0.5, len(df_im_s_y.drop(['Time Period', 'Area Name'], axis=1).columns)),
                theta=df_im_s_y.drop(['Time Period', 'Area Name'], axis=1).columns.to_numpy(),
                fill='toself',
                name='Serbia - Average'
        ))

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

def historical_info():
    df_im = pd.read_csv('data/im.csv')
    df_im['Vehicles Density'] = pd.cut(df_im['Vehicles Density'], bins=[0, 10.272, 21.286, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Motorcycle Density'] = pd.cut(df_im['Motorcycle Density'], bins=[0, 0.281, 0.762, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Main Roads Accessibility'] = pd.cut(df_im['Main Roads Accessibility'], bins=[0, 0.693, 1.676, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Local Roads Density'] = pd.cut(df_im['Local Roads Density'], bins=[0, 107.736, 450.854, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Primary School Attendance'] = pd.cut(df_im['Primary School Attendance'], bins=[0, 365.412, 590.641, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Secondary School Attendance'] = pd.cut(df_im['Secondary School Attendance'], bins=[0, 269.804, 435.033, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Assistance and Care Allowance Share'] = pd.cut(df_im['Assistance and Care Allowance Share'], bins=[0, 0.5, 0.6, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Poverty Share'] = pd.cut(df_im['Poverty Share'], bins=[0, 8.1, 12.9, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Doctors Accessibility'] = pd.cut(df_im['Doctors Accessibility'], bins=[0, 1.3, 2.1, 99999999], labels=['1', '2', '3']).astype(str)
    df_im['Net Migrations per 1000 inhabitants Category'] = pd.cut(df_im['Net Migrations per 1000 inhabitants'], bins=[-500000000, -4, 0, 99999999], labels=['1', '2', '3']).astype(str)
    df_im = df_im.replace({'nan': 'U'})

    LSGs = np.unique(df_im['Area Name'])
    years = np.unique(df_im['Time Period'])

    selected_lsg = st.selectbox(label='Please select municipality', options=LSGs, index=79)
    selected_year = st.selectbox(label='Please select year', options=years, index=9)

    df_mun = df_im.loc[(df_im['Area Name'] == selected_lsg) & (df_im['Time Period'] == selected_year), :]

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

        with col11:
            st.markdown('Vehicles Density')
        with col12:
            st.markdown('Motorcycle Density')
        with col13:
            st.markdown('&emsp;Vehicles Density and Motorcycle Density lead to - **Vehicles**')
        with col14:
            st.markdown('Main Roads Accessibility')
        with col15:
            st.markdown('Local Roads Density')
        with col16:
            st.markdown('&emsp;Main Roads Accessibility and Local Roads Density lead to - **Roads**')
        with col17:
            st.markdown('Primary School Attendance')
        with col18:
            st.markdown('Secondary School Attendance')
        with col19:
            st.markdown('&emsp;Primary School Attendance and Secondary School Attendance lead to - **School**')
        with col110:
            st.markdown('Assistance and Care Allowance Share')
        with col111:
            st.markdown('Poverty Share')
        with col112:
            st.markdown('&emsp;Assistance and Care Allowance Share and Poverty Share lead to - **Social Factors**')
        with col113:
            st.markdown('Doctors Accessibility')
        with col114:
            st.markdown('&emsp;&emsp;Social Factors and Doctors Accessibility lead to - **Health and Social Factors**')
        with col115:
            st.markdown('&emsp;&emsp;&emsp;Schools and Health and Social Factors lead to - **Social Determinants**')

        with col21:
            annotated_text((pred['Vehicles Density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Vehicles Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col22:
            annotated_text((pred['Motorcycle Density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Motorcycle Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col23:
            annotated_text((pred['Vehicles'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Vehicles'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col24:
            annotated_text((pred['Main Roads Accessibility'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'})[0], '', pred['Main Roads Accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
        with col25:
            annotated_text((pred['Local Roads Density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Local Roads Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col26:
            annotated_text((pred['Roads'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Roads'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col27:
            annotated_text((pred['Primary School Attendance'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Primary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col28:
            annotated_text((pred['Secondary School Attendance'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Secondary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col29:
            annotated_text((pred['School'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['School'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col210:
            annotated_text((pred['Assistance and Care Allowance Share'].map({'1': 'Good', '2': 'Medium', '3': 'red'})[0], '', pred['Assistance and Care Allowance Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
        with col211:
            annotated_text((pred['Poverty Share'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'})[0], '', pred['Poverty Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
        with col212:
            annotated_text((pred['Social Factors'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col213:
            annotated_text((pred['Doctors Accessibility'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Doctors Accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col214:
            annotated_text((pred['Health and Social Factors'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Health and Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
        with col215:
            annotated_text((pred['Social Determinants'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'})[0], '', pred['Social Determinants'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))

        st.markdown('**The predicted outcome is:**')
        prediction = dm.d.predict(df_mun).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction == 'Poor':
            annotated_text((prediction, '', 'red'))
            st.markdown('This model signals the **very high emigration** (greater than 4 persons on 1,000 inhabitants) from the municipality')
        elif prediction == 'Good':
            annotated_text((prediction, '', 'lightgreen'))
            st.markdown('This model signals the **immigration** to the municipality')
        else:
            annotated_text((prediction, '', 'gray'))
            st.markdown('This model signals the emigration (between 0 and 4 persons on 1,000 inhabitants) from the municipality')

        st.markdown('Please find below the projected outcome and the potential interventions:')

        st.markdown('---')
        st.subheader('Possible Interventions')
        st.markdown("This part of the page discuss the potential possibility of interventions on Net Internal migrations. We show potential interventions by changing the values of input attributes for one degree in improvement and one degree in deterioration. If a single intervention improves or deteriorates the outcome it will be presented in an appropriate color - red for the very negative net internal migrations (-4 or bigger emigration per 1,000 inhabitants), gray for negative (between -4 and 0 emigrations per 1,000 inhabitants), and green (positive migrations, or immigration).")

        ### Vehicles Density
        to_write = True
        if (df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 1]) | (df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[0][1].index(df_mun['Vehicles Density'].values[0]) - 1
            df_mun_v['Vehicles Density'] = dm.d._levels[0][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Vehicles Density* to a value between 10.272 and 21.286 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[0][1].index(df_mun['Vehicles Density'].values[0]) + 1
            df_mun_v['Vehicles Density'] = dm.d._levels[0][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Vehicles Density* to a value between 10.272 and 21.286 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[0][1].index(df_mun['Vehicles Density'].values[0]) + 1
            down_v = dm.d._levels[0][1].index(df_mun['Vehicles Density'].values[0]) - 1

            df_mun_v['Vehicles Density'] = dm.d._levels[0][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Vehicles Density* to a value lower than 10.272 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Vehicles Density'] = dm.d._levels[0][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Vehicles Density* to a value greater than 21.286 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Motorcycle Density
        to_write = True
        if (df_mun['Motorcycle Density'].values[0] == dm.d._levels[1][1][len(dm.d._levels[1][1]) - 1]) | (df_mun['Motorcycle Density'].values[0] == dm.d._levels[1][1][len(dm.d._levels[1][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[1][1].index(df_mun['Motorcycle Density'].values[0]) - 1
            df_mun_v['Motorcycle Density'] = dm.d._levels[1][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Motorcycle Density* to a value between 0.281 and 0.762 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Motorcycle Density'].values[0] == dm.d._levels[1][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[1][1].index(df_mun['Motorcycle Density'].values[0]) + 1
            df_mun_v['Motorcycle Density'] = dm.d._levels[1][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Motorcycle Density* to a value between 0.281 and 0.762 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[1][1].index(df_mun['Motorcycle Density'].values[0]) + 1
            down_v = dm.d._levels[1][1].index(df_mun['Motorcycle Density'].values[0]) - 1

            df_mun_v['Motorcycle Density'] = dm.d._levels[1][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Motorcycle Density* to a value lower than 0.281 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Motorcycle Density'] = dm.d._levels[1][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Motorcycle Density* to a value greater than 0.762 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        ### Main Road Accessibility
        to_write = True
        if (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 1]) | (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[2][1].index(df_mun['Main Roads Accessibility'].values[0]) - 2
            df_mun_v['Main Roads Accessibility'] = dm.d._levels[2][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Main Roads Accessibility* to a value between 0.693 and 1.676 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[2][1].index(df_mun['Main Roads Accessibility'].values[0]) + 1
            df_mun_v['Main Roads Accessibility'] = dm.d._levels[2][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Main Roads Accessibility* to a value between 0.693 and 1.676 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[2][1].index(df_mun['Main Roads Accessibility'].values[0]) + 1
            down_v = dm.d._levels[2][1].index(df_mun['Main Roads Accessibility'].values[0]) - 1

            df_mun_v['Main Roads Accessibility'] = dm.d._levels[2][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Main Roads Accessibility* to a value greater than 1.676 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Main Roads Accessibility'] = dm.d._levels[2][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Main Roads Accessibility* to a value lower than 0.693 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Local Roads Density
        to_write = True
        if (df_mun['Local Roads Density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 1]) | (df_mun['Local Roads Density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[3][1].index(df_mun['Local Roads Density'].values[0]) - 1
            df_mun_v['Local Roads Density'] = dm.d._levels[3][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Local Roads Density* to a value between 107.736 and 450.854 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Local Roads Density'].values[0] == dm.d._levels[3][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[3][1].index(df_mun['Local Roads Density'].values[0]) + 1
            df_mun_v['Local Roads Density'] = dm.d._levels[3][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Local Roads Density* to a value between 107.736 and 450.854 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[3][1].index(df_mun['Local Roads Density'].values[0]) + 1
            down_v = dm.d._levels[3][1].index(df_mun['Local Roads Density'].values[0]) - 1

            df_mun_v['Local Roads Density'] = dm.d._levels[3][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Local Roads Density* to a value lower than 107.736 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Local Roads Density'] = dm.d._levels[3][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Local Roads Density* to a value greater than 450.854 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Primary School Attendance
        to_write = True
        if (df_mun['Primary School Attendance'].values[0] == dm.d._levels[4][1][len(dm.d._levels[4][1]) - 1]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[4][1].index(df_mun['Primary School Attendance'].values[0]) - 1
            df_mun_v['Primary School Attendance'] = dm.d._levels[4][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Primary School Attendance* to a value between 365.412 and 590.641 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Primary School Attendance'].values[0] == dm.d._levels[4][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[4][1].index(df_mun['Primary School Attendance'].values[0]) + 1
            df_mun_v['Primary School Attendance'] = dm.d._levels[4][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Primary School Attendance* to a value between 365.412 and 590.641 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[4][1].index(df_mun['Primary School Attendance'].values[0]) + 1
            down_v = dm.d._levels[4][1].index(df_mun['Primary School Attendance'].values[0]) - 1

            df_mun_v['Primary School Attendance'] = dm.d._levels[4][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Primary School Attendance* to a value lower than 365.412 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Primary School Attendance'] = dm.d._levels[4][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Primary School Attendance* to a value greater than 590.641 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Secondary School Attendance
        to_write = True
        if (df_mun['Secondary School Attendance'].values[0] == dm.d._levels[5][1][len(dm.d._levels[5][1]) - 1]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[5][1].index(df_mun['Secondary School Attendance'].values[0]) - 1
            df_mun_v['Secondary School Attendance'] = dm.d._levels[5][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Secondary School Attendance* to a value between 269.804 and 435.033 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Secondary School Attendance'].values[0] == dm.d._levels[5][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[5][1].index(df_mun['Secondary School Attendance'].values[0]) + 1
            df_mun_v['Secondary School Attendance'] = dm.d._levels[5][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Secondary School Attendance* to a value between 269.804 and 435.033 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[5][1].index(df_mun['Secondary School Attendance'].values[0]) + 1
            down_v = dm.d._levels[5][1].index(df_mun['Secondary School Attendance'].values[0]) - 1

            df_mun_v['Secondary School Attendance'] = dm.d._levels[5][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Secondary School Attendance* to a value lower than 269.804 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[5][1].index(df_mun['Secondary School Attendance'].values[0]) + 1
            df_mun_v['Secondary School Attendance'] = dm.d._levels[5][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Secondary School Attendance* to a value greater than 435.033 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Assistance and Care Allowance Share
        to_write = True
        if (df_mun['Assistance and Care Allowance Share'].values[0] == dm.d._levels[6][1][len(dm.d._levels[6][1]) - 1]) | (df_mun['Assistance and Care Allowance Share'].values[0] == dm.d._levels[6][1][len(dm.d._levels[6][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[6][1].index(df_mun['Assistance and Care Allowance Share'].values[0]) - 1
            df_mun_v['Assistance and Care Allowance Share'] = dm.d._levels[6][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Assistance and Care Allowance Share* to a value between 0.5 and 0.6 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Assistance and Care Allowance Share'].values[0] == dm.d._levels[6][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[6][1].index(df_mun['Assistance and Care Allowance Share'].values[0]) + 1
            df_mun_v['Assistance and Care Allowance Share'] = dm.d._levels[6][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Assistance and Care Allowance Share* to a value between 0.5 and 0.6 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[6][1].index(df_mun['Assistance and Care Allowance Share'].values[0]) + 1
            down_v = dm.d._levels[6][1].index(df_mun['Assistance and Care Allowance Share'].values[0]) - 1

            df_mun_v['Assistance and Care Allowance Share'] = dm.d._levels[6][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Assistance and Care Allowance Share* to a value greater than 0.6 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Assistance and Care Allowance Share'] = dm.d._levels[6][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Assistance and Care Allowance Share* to a value lower than 0.5 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        ### Poverty Share
        to_write = True
        if (df_mun['Poverty Share'].values[0] == dm.d._levels[7][1][len(dm.d._levels[7][1]) - 1]) | (df_mun['Poverty Share'].values[0] == dm.d._levels[7][1][len(dm.d._levels[7][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[7][1].index(df_mun['Poverty Share'].values[0]) - 1
            df_mun_v['Poverty Share'] = dm.d._levels[7][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Poverty Share* to a value between 8.1 and 12.9 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Poverty Share'].values[0] == dm.d._levels[7][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[7][1].index(df_mun['Poverty Share'].values[0]) + 1
            df_mun_v['Poverty Share'] = dm.d._levels[7][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Poverty Share* to a value between 8.1 and 12.9 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[7][1].index(df_mun['Poverty Share'].values[0]) + 1
            down_v = dm.d._levels[7][1].index(df_mun['Poverty Share'].values[0]) - 1

            df_mun_v['Poverty Share'] = dm.d._levels[7][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **deterioration** in *Poverty Share* to a value greater than 12.9 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Poverty Share'] = dm.d._levels[7][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **improvement** in *Poverty Share* to a value lower than 8.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        ### Doctors Accessibility
        to_write = True
        if (df_mun['Doctors Accessibility'].values[0] == dm.d._levels[8][1][len(dm.d._levels[8][1]) - 1]) | (df_mun['Doctors Accessibility'].values[0] == dm.d._levels[8][1][len(dm.d._levels[8][1]) - 2]):
            to_write = False

            df_mun_v = df_mun.copy()
            down_v = dm.d._levels[8][1].index(df_mun['Doctors Accessibility'].values[0]) - 1
            df_mun_v['Doctors Accessibility'] = dm.d._levels[8][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Doctors Accessibility* to a value between 1.3 and 2.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
        
        if df_mun['Doctors Accessibility'].values[0] == dm.d._levels[8][1][0]:
            to_write = False

            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[8][1].index(df_mun['Doctors Accessibility'].values[0]) + 1
            df_mun_v['Doctors Accessibility'] = dm.d._levels[8][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Doctors Accessibility* to a value between 1.3 and 2.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

        if to_write:
            df_mun_v = df_mun.copy()
            up_v = dm.d._levels[8][1].index(df_mun['Doctors Accessibility'].values[0]) + 1
            down_v = dm.d._levels[8][1].index(df_mun['Doctors Accessibility'].values[0]) - 1

            df_mun_v['Doctors Accessibility'] = dm.d._levels[8][1][down_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('A **decrease** in *Doctors Accessibility* to a value lower than 1.3 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))

            df_mun_v = df_mun.copy()
            df_mun_v['Doctors Accessibility'] = dm.d._levels[8][1][up_v]
            prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
            if prediction != prediction_n:
                st.markdown('An **increase** in *Doctors Accessibility* to a value greater than 2.1 leads to the outcome:')
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))
