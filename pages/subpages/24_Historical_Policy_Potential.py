import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np

import dex_model_gva as dm

st.set_page_config(layout='wide', page_title = 'Historical Gross Value Added per capita Policy Outcomes')

st.title('Historical Gross Value Added per capita Policy Outcomes')

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
        st.markdown('Local roads density and Motorcycle Density lead to - **Local Traffic**')
    with col14:
        st.markdown('Vehicles density')
    with col15:
        st.markdown('Main road accessibility')
    with col16:
        st.markdown('Vehicles density and Main road accessibility lead to - **General Traffic**')
    with col17:
        st.markdown('Tourists Arrivals')
    with col18:
        st.markdown('Tourists Arrivals and General Traffic lead to - **Tourism and Traffic**')
    with col19:
        st.markdown('Preschool children enrollment rate')
    with col110:
        st.markdown('Doctors accessibility')
    with col111:
        st.markdown('Preschool children enrollment rate and Doctors accessibility lead to **Social Factors**')
    with col112:
        st.markdown('Municipality employment rate')
    with col113:
        st.markdown('Unemployed rate')
    with col114:
        st.markdown('Municipality employment rate and Unemployed rate lead to - **Employment State**')
    with col115:
        st.markdown('Active companies rate')
    with col116:
        st.markdown('Active companies rate and Employment State lead to - **Economy State**')
    with col117:
        st.markdown('Transport and storage investments rate')
    with col118:
        st.markdown('Economy State and Transport and storage investments rate lead to - **Economy and Investments**')

    with col21:
        annotated_text((pred['Local roads density'], '', pred['Local roads density'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
    with col22:
        annotated_text((pred['Motorcycles density'], '', pred['Motorcycles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col23:
        annotated_text((pred['Local Traffic'], '', pred['Local Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col24:
        annotated_text((pred['Vehicles density'], '', pred['Vehicles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col25:
        annotated_text((pred['Main road accessibility'], '', pred['Main road accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
    with col26:
        annotated_text((pred['General Traffic'], '', pred['General Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col27:
        annotated_text((pred['Tourists Arrivals'], '', pred['Tourists Arrivals'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col28:
        annotated_text((pred['Tourism and Traffic'], '', pred['Tourism and Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col29:
        annotated_text((pred['Preschool children enrollment rate'], '', pred['Preschool children enrollment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col210:
        annotated_text((pred['Doctors accessibility'], '', pred['Doctors accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col211:
        annotated_text((pred['Social Factors'], '', pred['Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col212:
        annotated_text((pred['Municipality employment rate'], '', pred['Municipality employment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col213:
        annotated_text((pred['Unemployed rate'], '', pred['Unemployed rate'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
    with col214:
        annotated_text((pred['Employment State'], '', pred['Employment State'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col215:
        annotated_text((pred['Active companies rate'], '', pred['Active companies rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col216:
        annotated_text((pred['Economy State'], '', pred['Economy State'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col217:
        annotated_text((pred['Transport and storage investments rate'], '', pred['Transport and storage investments rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col218:
        annotated_text((pred['Economy and Investments'], '', pred['Economy and Investments'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))

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
    st.markdown('#### Tourists Arrivals')
    to_write = True
    if (df_mun['Tourists Arrivals'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 1]):
        to_write = False
        st.markdown('The **increase** in *Tourists Arrivals* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Tourists Arrivals* to a value between 978.0 and 7629.333 **does not change** the outcome.')
    
    if df_mun['Tourists Arrivals'].values[0] == dm.d._levels[0][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Tourists Arrivals* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Tourists Arrivals* to a value between 978.0 and 7629.333 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Tourists Arrivals* to a value lower than 978.0 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Tourists Arrivals* to a value greater than 7629.333 **does not change** the outcome.')

    ### Local roads density
    st.markdown('#### Local roads density')
    to_write = True
    if (df_mun['Local roads density'].values[0] == dm.d._levels[1][1][len(dm.d._levels[1][1]) - 2]):
        to_write = False
        st.markdown('An **improvement** of *Local roads density* will not influence the outcome.')

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
        else:
            st.markdown('The **deterioration** of *Local roads density* to a value between 0.213 and 0.382 **does not change** the outcome.')
    
    if df_mun['Local roads density'].values[0] == dm.d._levels[1][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Local roads density* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Local roads density* to a value between 0.213 and 0.382 **does not change** the outcome.')
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
        else:
            st.markdown('The **deterioration** in *Local roads density* to a value greater than 0.382 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Local roads density* to a value lower than 0.213 **does not change** the outcome.')
    
    ### Motorcycles density
    st.markdown('#### Motorcycles density')
    to_write = True
    if (df_mun['Motorcycles density'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Motorcycles density* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Motorcycles density* to a value between 0.281 and 0.762 **does not change** the outcome.')
    
    if df_mun['Motorcycles density'].values[0] == dm.d._levels[2][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Motorcycles density* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Motorcycles density* to a value between 0.281 and 0.762 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Motorcycles density* to a value lower than 0.281 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Motorcycles density* to a value greater than 0.762 **does not change** the outcome.')

    ### Vehicles density
    st.markdown('#### Vehicles density')
    to_write = True
    if (df_mun['Vehicles density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 1]) | (df_mun['Vehicles density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 2]):
        to_write = False
        st.markdown('An **improvement** in *Vehicles density* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Vehicles density* to a value between 10.272 and 21.286 **does not change** the outcome.')
    
    if df_mun['Vehicles density'].values[0] == dm.d._levels[3][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Vehicles density* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Vehicles density* to a value between 10.272 and 21.286 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Vehicles density* to a value lower than 21.286 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Vehicles density* to a value lower than 10.272 **does not change** the outcome.')

    ### Main road accessibility
    st.markdown('#### Main road accessibility')
    to_write = True
    if (df_mun['Main road accessibility'].values[0] == dm.d._levels[4][1][len(dm.d._levels[4][1]) - 1]):
        to_write = False
        st.markdown('The **improvement** in *Main road accessibility* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Main road accessibility* to a value between 5.461 and 13.025 **does not change** the outcome.')
    
    if df_mun['Main road accessibility'].values[0] == dm.d._levels[4][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Main road accessibility* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Main road accessibility* to a value between 5.461 and 13.025 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Main road accessibility* to a value greater than 13.025 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Main road accessibility* to a value lower than 5.461 **does not change** the outcome.')

    ### Municipality employment rate
    st.markdown('#### Municipality employment rate')
    to_write = True
    if (df_mun['Municipality employment rate'].values[0] == dm.d._levels[5][1][len(dm.d._levels[5][1]) - 1]):
        to_write = False
        st.markdown('The **increase** in *Municipality employment rate* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Municipality employment rate* to a value between 0.291 and 0.378 **does not change** the outcome.')
    
    if df_mun['Municipality employment rate'].values[0] == dm.d._levels[5][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Municipality employment rate* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Municipality employment rate* to a value between 0.291 and 0.378 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Municipality employment rate* to a value lower than 0.291 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Municipality employment rate* to a value greater than 0.378 **does not change** the outcome.')

    ### Unemployed rate
    st.markdown('#### Unemployed rate')
    to_write = True
    if (df_mun['Unemployed rate'].values[0] == dm.d._levels[6][1][len(dm.d._levels[6][1]) - 1]):
        to_write = False
        st.markdown('The **increase** in *Unemployed rate* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Unemployed rate* to a value between 77.0 and 122.0 **does not change** the outcome.')
    
    if (df_mun['Unemployed rate'].values[0] == dm.d._levels[6][1][1]):
        to_write = False
        st.markdown('The **deterioration** in *Unemployed rate* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Unemployed rate* to a value between 77.0 and 122.0 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Unemployed rate* to a value greater than 122.0 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Unemployed rate* to a value lower than 77.0 **does not change** the outcome.')

    ### Active companies rate
    st.markdown('#### Active companies rate')
    to_write = True
    if (df_mun['Active companies rate'].values[0] == dm.d._levels[6][1][len(dm.d._levels[7][1]) - 1]) | (df_mun['Active companies rate'].values[0] == dm.d._levels[6][1][len(dm.d._levels[7][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Active companies rate* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Active companies rate* to a value between 7.755 and 10.985 **does not change** the outcome.')
    
    if df_mun['Active companies rate'].values[0] == dm.d._levels[7][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Active companies rate* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Active companies rate* to a value between 7.755 and 10.985 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Active companies rate* to a value lower than 7.755 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Active companies rate* to a value lower than 10.985 **does not change** the outcome.')

    ### Transport and storage investments rate
    st.markdown('#### Transport and storage investments rate')
    to_write = True
    if (df_mun['Transport and storage investments rate'].values[0] == dm.d._levels[8][1][len(dm.d._levels[8][1]) - 1]) | (df_mun['Transport and storage investments rate'].values[0] == dm.d._levels[8][1][len(dm.d._levels[8][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Transport and storage investments rate* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Transport and storage investments rate* to a value 0.562 **does not change** the outcome.')
    
    if df_mun['Transport and storage investments rate'].values[0] == dm.d._levels[8][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Transport and storage investments rate* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Transport and storage investments rate* to a value 0.562 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Transport and storage investments rate* to a value lower than 0.562 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Transport and storage investments rate* to a value greater than 0.562 **does not change** the outcome.')

    ### Preschool children enrollment rate
    st.markdown('#### Preschool children enrollment rate')
    to_write = True
    if (df_mun['Preschool children enrollment rate'].values[0] == dm.d._levels[9][1][len(dm.d._levels[9][1]) - 1]) | (df_mun['Preschool children enrollment rate'].values[0] == dm.d._levels[9][1][len(dm.d._levels[9][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Preschool children enrollment rate* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Preschool children enrollment rate* to a value between 2.086 and 2.682 **does not change** the outcome.')
    
    if df_mun['Preschool children enrollment rate'].values[0] == dm.d._levels[9][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Preschool children enrollment rate* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Preschool children enrollment rate* to a value between 2.086 and 2.682 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Preschool children enrollment rate* to a value lower than 2.086 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Preschool children enrollment rate* to a value lower than 2.682 **does not change** the outcome.')

    ### Doctors accessibility
    st.markdown('#### Doctors accessibility')
    to_write = True
    if (df_mun['Doctors accessibility'].values[0] == dm.d._levels[10][1][len(dm.d._levels[10][1]) - 1]) | (df_mun['Doctors accessibility'].values[0] == dm.d._levels[10][1][len(dm.d._levels[10][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Doctors accessibility* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Doctors accessibility* to a value between 1.3 and 2.1 **does not change** the outcome.')
    
    if df_mun['Doctors accessibility'].values[0] == dm.d._levels[10][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Doctors accessibility* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Doctors accessibility* to a value between 1.3 and 2.1 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Doctors accessibility* to a value lower than 1.3 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Doctors accessibility* to a value lower than 2.1 **does not change** the outcome.')