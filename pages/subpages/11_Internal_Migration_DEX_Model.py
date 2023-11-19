import streamlit as st
# from streamlit_agraph import agraph, Node, Edge, Config

import pandas as pd

st.set_page_config(layout='wide', page_title = 'Net Internal Migration DEX Model')

st.title('DEX Models for Net Internal Migration')

st.markdown(
    '''
    The final DIDEX extracted DEX model for internal migration management is shown below. **NOTE: This is our best effort having in mind the available data, constraints in data acquisition, the validity and accuracy of data, as well as the usability of the solution. The proposed model is intentially created to work with categories as this human decision makers are more prone to use qualitative values instead of exact numbers. Another note is that the accuracy of the predictive model is around 65% (using yearly based cross-validation) and this result is comparable to the results obtained using more complex machine learning models (that are not understandable to the human and that do not have an option to provide a list of possible policy interventions).** Triangles represent concepts that are available in existing data, while sqaures represent derived concepts that are calculated using the table of decision rules.
    
    As one can see, **internal net migrations** are described using **Traffic** and **Social Determinants** factors. 

    Traffic conditions and transportation infrastructure play a pivotal role in internal migration predictions. Regions with well-developed and efficient transportation networks are more likely to attract people. Access to job opportunities, educational institutions, and healthcare facilities is often determined by traffic ease and connectivity. Regions with congested traffic, limited public transportation, or inadequate road networks may experience outward migration as people seek better access to essential services and improved quality of life elsewhere. 

    Access to quality education is also a fundamental driver of internal migration. Families often relocate to areas with better school opportunities to provide their children with better educational opportunities. Regions with diverse and well-performing schools tend to attract migrants, leading to net inward migration. Conversely, areas with underfunded or struggling education may experience outward migration as people seek better services elsewhere. Social factors, such as **poverty share** and **assistance and care allowance share**, can also influence internal migration patterns. People often migrate to areas where they feel a sense of belonging and acceptance. Communities that have high poverty share and are in the need for care are often struggling and experiencing outward migrations. Additionally, factors like crime rates, social amenities, and recreational opportunities can impact migration decisions. Regions that provide a safe, vibrant, and fulfilling social environment are more likely to experience net inward migration.
    '''
)

st.markdown('---')
st.subheader('Attribute Description')

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
# st.table(df_text, unsafe_allow_html=True)
st.write(df_text[['Attribute Name', 'Description']].to_html(escape=False), unsafe_allow_html=True)

# st.markdown(
#     '''
#     However, one needs an explanation of the attributes.

#     Table 1. List of attributes used for IM explanation and prediction
#     RN | Attribute Name | Description
#     ---|:---|:---
#     1|Vehicles density|The number of registered passenger vehicles per square meter of the municipality.
#     2|Motorcycles density|The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.
#     3|Main roads accessibility|The total length (in kilometers) of 1st-class road segments without highways divided by 1,000 inhabitants of the municipality.
#     4|Local roads density|The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of the municipality.
#     5|Primary school's attendance|The average number of children in primary school that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 7 and 14.
#     6|Secondary school's attendance|The average number of children in secondary school (both vocational and general) that are within the municipality.  To calculate the number of children, we used an estimate of the number of people between the ages of 15 and 18.
#     7|Assistance and care allowance share|Percentage of people within one municipality that use an increased allowance for assistance and care of another person.
#     8|Poverty share|Percentage of people within one municipality that uses social protection.
#     9|Doctors accessibility|The total number of doctors within a municipality divided by the 1,000 inhabitants.
    
    
#     **Vehicles density** represents the number of registered passenger vehicles per square meter of the municipality. According to the European driving license classification, passenger vehicles are four-wheel vehicles that have one or more passenger spaces but weigh less than 3,500 kilograms. This information is collected from the Ministry of Internal Affairs, and it represents the total number of passenger vehicles registered in an LSG. Further, to normalize the total number of vehicles to a common measure, we divided the total number of passenger vehicles with the size of the municipality leading to information about the number of passenger vehicles per square kilometer. 
    
#     Attribute **Motorcycles density** has a pattern of positive correlation between this attribute and net migrations, meaning that greater motorcycle density corresponds to higher immigration in the municipality. In other words, it carries similar information as vehicle density. However, this attribute has a better discrimination power for high emigration LSGs than the number of passenger vehicles per square meter. Having the interpretations above, a combination of these two attributes would be a good discriminator for internal migrations.

#     Interestingly, the **Main roads accessibility** has a negative correlation with net migrations. In other words, the higher the net migrations lower the value for the main roads accessibility, as well as when net migrations are highly negative and main roads accessibility are very high. This occurrence may be explained by the normalization used. More specifically, areas that are highly populated and have positive migration have their main road segments (category I road segments without highways) lower due to higher value in the denominator. Conversely, municipalities with a low population have a very low denominator, which leads to a high number of total length of main roads accessibility. It is worth noting that we ensured that the confounding effect between the aforementioned attributes is minimized as net migration was also calculated using the same normalization. On the contrary, if we use **local road density** we get a positive correlation with net migrations. The signal is not as strong as with the number of passenger vehicles or the number of motors and motorcycles, but municipalities with a high level of local roads density have a high net migration.

#     By observing the relation between **Primary school's attendance** and net migrations, we notied that LSGs with a low number of pupils per primary school have negative migrations. This is a common finding in Serbia, as depopulation and internal migration leave schools with fewer pupils. These municipalities simply have more schools than needed (however, this is a different topic compared to the one in this project and requires a deeper discussion). On the other side of the spectrum, LSGs with a high number of pupils are associated with positive migrations. A much crisper signal is observable for the **Secondary school's attendance**. As there are municipalities without any type of secondary school, those that do have secondary schools act as attractors. This is clearly observable for high net migrations.

#     The next set of attributes is related to the social factors in the municipality. Attribute **Assistance and care allowance share** is the first one on the list. We can state that the relationship with net migrations is negative. In other words, higher the share of assistance and care allowance, lower the net migrations. Similarly, the attribute **Poverty share** has a negative correlation with net migrations.

#     Finally, the **doctor's accessibility** is an attribute that represents the availability of medical services in the municipality. Compared to every other attribute in this group, it does not have a perfectly linear correlation with net migrations. It is having a U-shape relationship In areas with very low and with very high doctors accessibility there is a high net migration. However, in areas with medium accessibility to doctors, net migrations are the lowest.
#     '''
# )

# nodes = []
# edges = []

# nodes.append(Node(id='Internal Net Migrations', label='Internal Net Migrations', shape='square', color = 'blue'))

# nodes.append(Node(id='Traffic', label='Traffic', shape='square'))

# nodes.append(Node(id='Vehicles', label='Vehicles', shape='square'))
# nodes.append(Node(id='Vehicles density', label='Vehicles density', shape='triangle'))
# nodes.append(Node(id='Motorcycle density', label='Motorcycle density', shape='triangle'))

# nodes.append(Node(id='Roads', label='Roads', shape='square'))
# nodes.append(Node(id='Main roads accessibility', label='Main roads accessibility', shape='triangle'))
# nodes.append(Node(id='Local roads density', label='Local roads density', shape='triangle'))

# nodes.append(Node(id='Social Determinants', label='Social Determinants', shape='square'))

# nodes.append(Node(id='School', label='School', shape='square'))
# nodes.append(Node(id='Primary school attendance', label='Primary school attendance', shape='triangle'))
# nodes.append(Node(id='Secondary school attendance', label='Secondary school attendance', shape='triangle'))

# nodes.append(Node(id='Health and Social Factors', label='Health and Social Factors', shape='square'))

# nodes.append(Node(id='Social Factors', label='Social Factors', shape='square'))
# nodes.append(Node(id='Assistance and care allowance share', label='Assistance and care allowance share', shape='triangle'))
# nodes.append(Node(id='Poverty share', label='Poverty share', shape='triangle'))

# nodes.append(Node(id='Doctors accessibility', label='Doctors accessibility', shape='triangle'))

# edges.append(Edge(source='Internal Net Migrations', target='Traffic'))
# edges.append(Edge(source='Traffic', target='Vehicles'))
# edges.append(Edge(source='Vehicles', target='Vehicles density'))
# edges.append(Edge(source='Vehicles', target='Motorcycle density'))
# edges.append(Edge(source='Traffic', target='Roads'))
# edges.append(Edge(source='Roads', target='Main roads accessibility'))
# edges.append(Edge(source='Roads', target='Local roads density'))
# edges.append(Edge(source='Internal Net Migrations', target='Social Determinants'))
# edges.append(Edge(source='Social Determinants', target='School'))
# edges.append(Edge(source='School', target='Primary school attendance'))
# edges.append(Edge(source='School', target='Secondary school attendance'))
# edges.append(Edge(source='Social Determinants', target='Health and Social Factors'))
# edges.append(Edge(source='Health and Social Factors', target='Social Factors'))
# edges.append(Edge(source='Social Factors', target='Assistance and care allowance share'))
# edges.append(Edge(source='Social Factors', target='Poverty share'))
# edges.append(Edge(source='Health and Social Factors', target='Doctors accessibility'))

# config = Config(directed=True,  hierarchical=True)

# st.markdown('---')
# st.subheader('DEX Model')

# return_value = agraph(nodes = nodes, edges = edges, config = config)

st.markdown('---')
st.subheader('Model Hierarchy')

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

st.markdown('---')
st.subheader('Attribute Values')

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

st.markdown('---')
st.subheader('Decision Rules')

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
    df_1 = [['Poor', '*', 'Poor'], ['Medium', '<= Good', 'Medium'], ['>= Medium', 'Unknown', 'Good'], ['Good', '*', 'Good']]
    df_1 = pd.DataFrame(df_1, columns = ['Social factors', 'Doctors Accessibility', 'Health and Social Factors'])

    st.table(df_1.style.applymap(lambda x: highlight(x)))

    st.markdown(
        '''
        The table below can be read as follows: If **Social Factors** are *Poor* and whatever the value for the attribute **Doctors Accessibility** the value of factor **Health and Social Factors** will be Poor. If **Social Factors** is *Medium* and if the value of **Doctors Accessibility** is *Good* or lower than that value (i.e. *Medium* or *Poor*) then the factor **Health and Social Factors** will be Medium. A *Good* value of the resulting factor is achieved in two cases. When the factor **Social Factors** is *Medium* or *Good*, and the attribute **Doctors Accessibility** is *Unknown*, or when the factor **Social Factors** is *Good*.
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