import streamlit as st
# from streamlit_agraph import agraph, Node, Edge, Config

import pandas as pd

st.set_page_config(layout='wide', page_title = 'GVA Per Capita DEX Model')

st.title('DEX Models for (Pooled) GVA Per Capita')

st.markdown(
    '''
    The final DIDEX extracted DEX model for the (Pooled) GVA Per Capita is shown below. **NOTE: This is our best effort having in mind the available data, constraints in data acquisition, the validity and accuracy of data, as well as the usability of the solution. The proposed model is intentially created to work with categories as this human decision makers are more prone to use qualitative values instead of exact numbers. Another note is that the accuracy of the predictive model is around 70% (using yearly based cross-validation) and this result is comparable to the results obtained using more complex machine learning models (that are not understandable to the human and that do not have an option to provide a list of possible policy interventions).**
    
    Gross-value-added per capita prediction has a confounding effect as the economy improves over time. Consequently, the best-performing municipalities in 2010 had similar economic activity as the worst-performing municipalities in 2021. Although expected, one can argue about whether these numbers are comparable. In other words, using these values to explain or predict the gross value added would lead to inaccuracies in interpretation and consequently policymaking. To make each year comparable (in other words, to eliminate the effect of time), we performed year-wise pooled Z transformation. This means gross value added per capita has an average value equal to zero and a standard deviation equal to one for every year. Thus, municipalities that are consistently below average would have negative values, and vice-versa municipalities that are consistently above average would have positive values.

    The obtained DEX model can be divided into three categories. Namely, *Tourism and Traffic*, *Social Factors*, and *Economy and Investments*. This depicts the multifaceted nature of the gross value added prediction. While logically the state of economy in LSG and investments in transportation have a high influence on the outcome, accessibility to resources identified by *Tourism and Traffic* factor, as well as *Social Factors* influence the outcome as well.  
    '''
)

st.markdown('---')
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

st.markdown('---')
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

st.markdown('---')
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