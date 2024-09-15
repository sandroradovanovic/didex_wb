import streamlit as st
from annotated_text import annotated_text

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import dex_model as dm
import dex_model_gva as dm_gva

from page_util import apply_style

st.set_page_config(layout='wide', page_title = 'Home')

st.title('DEX Models for Internal Migration and Gross Value Added per capita Explanation and Potential Suggestions')

# st.image('img/world-bank-logo.png', width=300)
apply_style()

with st.expander('This web app acts as a service to policy- and decision-makers in the local self-governing units or Ministry of Public Administration and Local Self-Government to inspect effects of decisions and/or policies on **net internal migrations** and **gross value added per capita (constant prices)**.'):
    st.markdown('---')
    st.markdown("**NOTE: This is our best effort having in mind the available data, constraints in data acquisition, the validity and accuracy of data, as well as the usability of the solution. The proposed model is intentially created to work with categories as this human decision makers are more prone to use qualitative values instead of exact numbers. Another note is that the accuracy of the predictive model is around 65% and 70% for net internal migrations and gross value added per capita respectively (using yearly based cross-validation) and these results are comparable to the results obtained using more complex machine learning models (that are not understandable to the human and that do not have an option to provide a list of possible policy interventions) using the same or more comprehensive set of attributes and using the same validation.**")

st.markdown('For a short demonstration of the proposed models, please select a municipality and predicted results for net migrations and gross value added will be shown below. We will take the data for 2021 from the Open Data portal of Statistical Office of the Republic of Serbia.')
st.markdown('You can test the proposed tool by clicking the approapriate menu item at the sidebar (either *Internal Migrations DEX Model* or *Gross Value Added DEX Model*)')

df_im = pd.read_csv('data/im.csv')
df_im_values = pd.read_csv('data/im.csv')
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

selected_lsg = st.selectbox(label='Please select municipality', options=LSGs, index=79)
df_mun = df_im.loc[(df_im['Area Name'] == selected_lsg) & (df_im['Time Period'] == 2021), :]
df_mun_val = df_im_values.loc[(df_im_values['Area Name'] == selected_lsg) & (df_im_values['Time Period'] == 2021), :]
df_im_values_selected = df_im_values.loc[df_im_values['Time Period'] == 2021, :]

st.markdown('---')
st.subheader('Visualization of the data')

with st.expander('Visualization for comparison'):
    st.subheader('[Net Migrations Visualization](Internal_Migrations_DEX_Model)')
    df_im_s = df_im_values.copy()
    df_im_s = df_im_s.loc[df_im_s['Time Period'] == 2021, :]
    
    df_im_s['Main Roads Accessibility'] = 1/df_im_s['Main Roads Accessibility']
    df_im_s['Assistance and Care Allowance Share'] = 1/df_im_s['Assistance and Care Allowance Share']
    df_im_s['Poverty Share'] = 1/df_im_s['Poverty Share']

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    for i in range(2,len(df_im_s.columns)):
        df_im_s.iloc[:, i] = sigmoid((df_im_s.iloc[:, i] - df_im_s.iloc[:, i].mean())/df_im_s.iloc[:, i].std())
    # df_im_s.iloc[:, 2:-1] = df_im_s.iloc[:, 2:-1]/df_im_s.iloc[:, 2:-1].max()
    # df_im_s['Net Migrations per 1000 inhabitants'] = (df_im_s['Net Migrations per 1000 inhabitants'] - df_im_s['Net Migrations per 1000 inhabitants'].min())/(df_im_s['Net Migrations per 1000 inhabitants'].max() - df_im_s['Net Migrations per 1000 inhabitants'].min())

    fig1 = go.Figure()

    df_im_s_y_lsg = df_im_s.loc[df_im_s['Area Name'] == selected_lsg, :].drop(['Time Period', 'Area Name'], axis=1)

    st.markdown('The value of the Net Internal Migrations (per 1,000 inhabitants) is:')
    color = 'red' if df_im_s_y_lsg['Net Migrations per 1000 inhabitants'].values[0] < -4 else ('lightgreen' if df_im_s_y_lsg['Net Migrations per 1000 inhabitants'].values[0] > 0 else 'gray')
    annotated_text((f"{np.round(df_im_s_y_lsg['Net Migrations per 1000 inhabitants'].values[0], 3)}", '', color))

    fig1.add_trace(go.Scatterpolar(
        r=np.repeat(0.5, len(df_im_s_y_lsg.columns) - 1),
        theta=df_im_s_y_lsg.columns.to_numpy()[:-1],
        fill='toself',
        name='Serbia - Average'
    ))

    fig1.add_trace(go.Scatterpolar(
        r=df_im_s_y_lsg.to_numpy()[0][:-1],
        theta=df_im_s_y_lsg.columns.to_numpy()[:-1],
        fill='toself',
        name=selected_lsg
    ))
    st.plotly_chart(fig1)

    st.subheader('[Gross Value Added per capita Visualization](Gross_Value_Added_DEX_Model)')

    df_gva = pd.read_csv('data/gva.csv')
    df_gva_s = df_gva.copy()
    df_gva_s = df_gva_s.loc[df_gva_s['Time Period'] == 2021, :]
    df_gva_s['Main road accessibility'] = 1/df_gva_s['Main road accessibility']
    df_gva_s['Local roads density'] = 1/df_gva_s['Local roads density']
    df_gva_s['Unemployed rate'] = 1/df_gva_s['Unemployed rate']
    df_gva_s['Vehicles density'] = 1/df_gva_s['Vehicles density']

    for i in range(2,len(df_gva_s.columns)):
        df_gva_s.iloc[:, i] = sigmoid((df_gva_s.iloc[:, i] - df_gva_s.iloc[:, i].mean())/df_gva_s.iloc[:, i].std())

    # df_gva_s.iloc[:, 2:-1] = df_gva_s.iloc[:, 2:-1]/df_gva_s.iloc[:, 2:-1].max()
    # df_gva_s['GVA Per Capita Normalized'] = (df_gva_s['GVA Per Capita Normalized'] - df_gva_s['GVA Per Capita Normalized'].min())/(df_gva_s['GVA Per Capita Normalized'].max() - df_gva_s['GVA Per Capita Normalized'].min())

    fig2 = go.Figure()

    df_gva_s_y_lsg = df_gva_s.loc[df_gva_s['Area Name'] == selected_lsg, :].drop(['Time Period', 'Area Name'], axis=1)

    st.markdown('The value of the Gross Value Added per capita (with values pooled year-wise) is:')
    color = 'red' if df_gva_s_y_lsg['GVA Per Capita Normalized'].values[0] < -0.25 else ('lightgreen' if df_gva_s_y_lsg['GVA Per Capita Normalized'].values[0] > 0 else 'gray')
    annotated_text((f"{np.round(df_gva_s_y_lsg['GVA Per Capita Normalized'].values[0], 3)}", '', color))

    fig2.add_trace(go.Scatterpolar(
        r=np.repeat(0.5, len(df_gva_s_y_lsg.columns) - 1),
        theta=df_gva_s_y_lsg.columns.to_numpy()[:-1],
        fill='toself',
        name='Serbia - Average'
    ))
    fig2.add_trace(go.Scatterpolar(
        r=df_gva_s_y_lsg.to_numpy()[0][:-1],
        theta=df_gva_s_y_lsg.columns.to_numpy()[:-1],
        fill='toself',
        name=selected_lsg
    ))
    st.plotly_chart(fig2)

st.markdown('---')
st.subheader('[Net Migration Prediction and Policy Potential](Internal_Migrations_DEX_Model)')

with st.expander('Details'):
    st.markdown('### Selected Municipality Attributes')
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

        with col11:
            st.markdown('Vehicles Density')
        with col12:
            st.markdown('Motorcycle Density')
        with col13:
            st.markdown('Main Roads Accessibility')
        with col14:
            st.markdown('Local Roads Density')
        with col15:
            st.markdown('Primary School Attendance')
        with col16:
            st.markdown('Secondary School Attendance')
        with col17:
            st.markdown('Assistance and Care Allowance Share')
        with col18:
            st.markdown('Poverty Share')
        with col19:
            st.markdown('Doctors Accessibility')

        with col21:
            annotated_text((np.round(df_mun_val['Vehicles Density'], 3), '', pred['Vehicles Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_im_values_selected['Vehicles Density'].mean(), 3)})", 'Serbia', ''))
        with col22:
            annotated_text((np.round(df_mun_val['Motorcycle Density'], 3), '', pred['Motorcycle Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_im_values_selected['Motorcycle Density'].mean(), 3)})", 'Serbia', ''))
        with col23:
            annotated_text((np.round(df_mun_val['Main Roads Accessibility'], 3), '', pred['Main Roads Accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]), (f"({np.round(df_im_values_selected['Main Roads Accessibility'].mean(), 3)})", 'Serbia', ''))
        with col24:
            annotated_text((np.round(df_mun_val['Local Roads Density'], 3), '', pred['Local Roads Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_im_values_selected['Local Roads Density'].mean(), 3)})", 'Serbia', ''))
        with col25:
            annotated_text((np.round(df_mun_val['Primary School Attendance'], 3), '', pred['Primary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_im_values_selected['Primary School Attendance'].mean(), 3)})", 'Serbia', ''))
        with col26:
            annotated_text((np.round(df_mun_val['Secondary School Attendance'], 3), '', pred['Secondary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_im_values_selected['Secondary School Attendance'].mean(), 3)})", 'Serbia', ''))
        with col27:
            annotated_text((np.round(df_mun_val['Assistance and Care Allowance Share'], 3), '', pred['Assistance and Care Allowance Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]), (f"({np.round(df_im_values_selected['Assistance and Care Allowance Share'].mean(), 3)})", 'Serbia', ''))
        with col28:
            annotated_text((np.round(df_mun_val['Poverty Share'], 3), '', pred['Poverty Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]), (f"({np.round(df_im_values_selected['Poverty Share'].mean(), 3)})", 'Serbia', ''))
        with col29:
            annotated_text((np.round(df_mun_val['Doctors Accessibility'], 3), '', pred['Doctors Accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_im_values_selected['Doctors Accessibility'].mean(), 3)})", 'Serbia', ''))

    st.markdown('### Predicted Outcome')
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

    st.markdown('### Policy Potential')
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

    ### Main Road Accessibility
    to_write = True
    if (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 1]) | (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm.d._levels[2][1].index(df_mun['Main Roads Accessibility'].values[0]) - 1
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
        st.markdown('The **increase** in *Doctors Accessibility* will not influence the outcome.')

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

### ----- ----- ----- -----
df_gva = pd.read_csv('data/gva.csv')
df_gva_values = pd.read_csv('data/gva.csv')
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

df_mun = df_gva.loc[(df_gva['Area Name'] == selected_lsg) & (df_gva['Time Period'] == 2021), :]
df_mun_values = df_gva_values.loc[(df_gva_values['Area Name'] == selected_lsg) & (df_gva_values['Time Period'] == 2021), :]
df_gva_values_selected = df_gva_values.loc[df_gva_values['Time Period'] == 2021, :]

st.markdown('---')
st.subheader('[Gross Value Added per capita Prediction and Policy Potential](Gross_Value_Added_DEX_Model)')

with st.expander('Details'):
    st.markdown('### Selected Municipality Attributes')
    if df_mun.shape[0] == 0:
        st.markdown("We don't have data for the selected municipality and the selected year")
    else:
        pred = dm_gva.d.predict(df_mun, return_intermediate=True)

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

        with col11:
            st.markdown('Local roads density')
        with col12:
            st.markdown('Motorcycles density')
        with col13:
            st.markdown('Vehicles density')
        with col14:
            st.markdown('Main road accessibility')
        with col15:
            st.markdown('Tourists Arrivals')
        with col16:
            st.markdown('Preschool children enrollment rate')
        with col17:
            st.markdown('Doctors accessibility')
        with col18:
            st.markdown('Municipality employment rate')
        with col19:
            st.markdown('Unemployed rate')
        with col110:
            st.markdown('Active companies rate')
        with col111:
            st.markdown('Transport and storage investments rate')

        with col21:
            annotated_text((np.round(df_mun_values['Local roads density'], 3), '', pred['Local roads density'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]), (f"({np.round(df_gva_values_selected['Local roads density'].mean(), 3)})", 'Serbia', ''))
        with col22:
            annotated_text((np.round(df_mun_values['Motorcycles density'], 3), '', pred['Motorcycles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Motorcycles density'].mean(), 3)})", 'Serbia', ''))
        with col23:
            annotated_text((np.round(df_mun_values['Vehicles density'], 3), '', pred['Vehicles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Vehicles density'].mean(), 3)})", 'Serbia', ''))
        with col24:
            annotated_text((np.round(df_mun_values['Main road accessibility'], 3), '', pred['Main road accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]), (f"({np.round(df_gva_values_selected['Main road accessibility'].mean(), 3)})", 'Serbia', ''))
        with col25:
            annotated_text((np.round(df_mun_values['Tourists Arrivals'], 3), '', pred['Tourists Arrivals'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Tourists Arrivals'].mean(), 3)})", 'Serbia', ''))
        with col26:
            annotated_text((np.round(df_mun_values['Preschool children enrollment rate'], 3), '', pred['Preschool children enrollment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Preschool children enrollment rate'].mean(), 3)})", 'Serbia', ''))
        with col27:
            annotated_text((np.round(df_mun_values['Doctors accessibility'], 3), '', pred['Doctors accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Doctors accessibility'].mean(), 3)})", 'Serbia', ''))
        with col28:
            annotated_text((np.round(df_mun_values['Municipality employment rate'], 3), '', pred['Municipality employment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Municipality employment rate'].mean(), 3)})", 'Serbia', ''))
        with col29:
            annotated_text((np.round(df_mun_values['Unemployed rate'], 3), '', pred['Unemployed rate'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]), (f"({np.round(df_gva_values_selected['Unemployed rate'].mean(), 3)})", 'Serbia', ''))
        with col210:
            annotated_text((np.round(df_mun_values['Active companies rate'], 3), '', pred['Active companies rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Active companies rate'].mean(), 3)})", 'Serbia', ''))
        with col211:
            annotated_text((np.round(df_mun_values['Transport and storage investments rate'], 3), '', pred['Transport and storage investments rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]), (f"({np.round(df_gva_values_selected['Transport and storage investments rate'].mean(), 3)})", 'Serbia', ''))

    st.markdown('### Predicted Outcome')
    st.markdown('**The predicted outcome is:**')
    prediction = dm_gva.d.predict(df_mun).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
    if prediction == 'Poor':
        annotated_text((prediction, '', 'red'))
        st.markdown('This model signals the **Gross Value Added per capita** lower by 25% than the average of Serbia')
    elif prediction == 'Good':
        annotated_text((prediction, '', 'lightgreen'))
        st.markdown('This model signals the **Gross Value Added per capita** higher than the average of Serbia')
    else:
        annotated_text((prediction, '', 'gray'))
        st.markdown('This model signals the **Gross Value Added per capita** lower, but close to the average of  Serbia')

    st.markdown('### Policy Potential')
    st.markdown("This part of the page discuss the potential possibility of interventions on Gross Value Added per capita. We show potential interventions by changing the values of input attributes for one degree in improvement and one degree in deterioration. If a single intervention improves or deteriorates the outcome it will be presented in an appropriate color - red for the low Gross Value added per capita (-0.25 or lower), gray for negative (between -0.25 and 0), and green (greater than 0).")

    ### Tourists Arrivals
    to_write = True
    if (df_mun['Tourists Arrivals'].values[0] == dm_gva.d._levels[0][1][len(dm_gva.d._levels[0][1]) - 1]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) - 1
        df_mun_v['Tourists Arrivals'] = dm_gva.d._levels[0][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Tourists Arrivals* to a value between 978.0 and 7629.333 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Tourists Arrivals'].values[0] == dm_gva.d._levels[0][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) + 1
        df_mun_v['Tourists Arrivals'] = dm_gva.d._levels[0][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) + 1
        down_v = dm_gva.d._levels[0][1].index(df_mun['Tourists Arrivals'].values[0]) - 1

        df_mun_v['Tourists Arrivals'] = dm_gva.d._levels[0][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Tourists Arrivals* to a value lower than 978.0 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Tourists Arrivals'] = dm_gva.d._levels[0][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Local roads density'].values[0] == dm_gva.d._levels[1][1][len(dm_gva.d._levels[1][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[1][1].index(df_mun['Local roads density'].values[0]) - 1
        df_mun_v['Local roads density'] = dm_gva.d._levels[1][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('The **deterioration** of *Local roads density* to a value between 0.213 and 0.382 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Local roads density'].values[0] == dm_gva.d._levels[1][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[1][1].index(df_mun['Local roads density'].values[0]) + 1
        df_mun_v['Local roads density'] = dm_gva.d._levels[1][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[1][1].index(df_mun['Local roads density'].values[0]) + 1
        down_v = dm_gva.d._levels[1][1].index(df_mun['Local roads density'].values[0]) - 1

        df_mun_v['Local roads density'] = dm_gva.d._levels[1][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('The **deterioration** in *Local roads density* to a value greater than 0.382 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Local roads density'] = dm_gva.d._levels[1][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Motorcycles density'].values[0] == dm_gva.d._levels[2][1][len(dm_gva.d._levels[2][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) - 2
        df_mun_v['Motorcycles density'] = dm_gva.d._levels[2][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Motorcycles density* to a value between 0.281 and 0.762 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Motorcycles density'].values[0] == dm_gva.d._levels[2][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) + 1
        df_mun_v['Motorcycles density'] = dm_gva.d._levels[2][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) + 1
        down_v = dm_gva.d._levels[2][1].index(df_mun['Motorcycles density'].values[0]) - 1

        df_mun_v['Motorcycles density'] = dm_gva.d._levels[2][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Motorcycles density* to a value lower than 0.281 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Motorcycles density'] = dm_gva.d._levels[2][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Vehicles density'].values[0] == dm_gva.d._levels[3][1][len(dm_gva.d._levels[3][1]) - 1]) | (df_mun['Vehicles density'].values[0] == dm_gva.d._levels[3][1][len(dm_gva.d._levels[3][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) - 1
        df_mun_v['Vehicles density'] = dm_gva.d._levels[3][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **deterioration** in *Vehicles density* to a value between 10.272 and 21.286 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Vehicles density'].values[0] == dm_gva.d._levels[3][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) + 1
        df_mun_v['Vehicles density'] = dm_gva.d._levels[3][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) + 1
        down_v = dm_gva.d._levels[3][1].index(df_mun['Vehicles density'].values[0]) - 1

        df_mun_v['Vehicles density'] = dm_gva.d._levels[3][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **deterioration** in *Vehicles density* to a value greater than 21.286 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Vehicles density'] = dm_gva.d._levels[3][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Main road accessibility'].values[0] == dm_gva.d._levels[4][1][len(dm_gva.d._levels[4][1]) - 1]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) - 1
        df_mun_v['Main road accessibility'] = dm_gva.d._levels[4][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **deterioration** in *Main road accessibility* to a value between 5.461 and 13.025 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Main road accessibility'].values[0] == dm_gva.d._levels[4][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) + 1
        df_mun_v['Main road accessibility'] = dm_gva.d._levels[4][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) + 1
        down_v = dm_gva.d._levels[4][1].index(df_mun['Main road accessibility'].values[0]) - 1

        df_mun_v['Main road accessibility'] = dm_gva.d._levels[4][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **deterioration** in *Main road accessibility* to a value greater than 13.025 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Main road accessibility'] = dm_gva.d._levels[4][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Municipality employment rate'].values[0] == dm_gva.d._levels[5][1][len(dm_gva.d._levels[5][1]) - 1]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) - 1
        df_mun_v['Municipality employment rate'] = dm_gva.d._levels[5][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Municipality employment rate* to a value between 0.291 and 0.378 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Municipality employment rate'].values[0] == dm_gva.d._levels[5][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) + 1
        df_mun_v['Municipality employment rate'] = dm_gva.d._levels[5][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) + 1
        down_v = dm_gva.d._levels[5][1].index(df_mun['Municipality employment rate'].values[0]) - 1

        df_mun_v['Municipality employment rate'] = dm_gva.d._levels[5][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Municipality employment rate* to a value lower than 0.291 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Municipality employment rate'] = dm_gva.d._levels[5][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Unemployed rate'].values[0] == dm_gva.d._levels[6][1][len(dm_gva.d._levels[6][1]) - 1]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) - 1
        df_mun_v['Unemployed rate'] = dm_gva.d._levels[6][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **deterioration** in *Unemployed rate* to a value between 77.0 and 122.0 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if (df_mun['Unemployed rate'].values[0] == dm_gva.d._levels[6][1][1]):
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) + 1
        df_mun_v['Unemployed rate'] = dm_gva.d._levels[6][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) + 1
        down_v = dm_gva.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) - 1

        df_mun_v['Unemployed rate'] = dm_gva.d._levels[6][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **deterioration** in *Unemployed rate* to a value greater than 122.0 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[6][1].index(df_mun['Unemployed rate'].values[0]) + 1
        df_mun_v['Unemployed rate'] = dm_gva.d._levels[6][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Active companies rate'].values[0] == dm_gva.d._levels[6][1][len(dm_gva.d._levels[7][1]) - 1]) | (df_mun['Active companies rate'].values[0] == dm_gva.d._levels[6][1][len(dm_gva.d._levels[7][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) - 1
        df_mun_v['Active companies rate'] = dm_gva.d._levels[7][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Active companies rate* to a value between 7.755 and 10.985 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Active companies rate'].values[0] == dm_gva.d._levels[7][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) + 1
        df_mun_v['Active companies rate'] = dm_gva.d._levels[7][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) + 1
        down_v = dm_gva.d._levels[7][1].index(df_mun['Active companies rate'].values[0]) - 1

        df_mun_v['Active companies rate'] = dm_gva.d._levels[7][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Active companies rate* to a value lower than 7.755 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Active companies rate'] = dm_gva.d._levels[7][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Transport and storage investments rate'].values[0] == dm_gva.d._levels[8][1][len(dm_gva.d._levels[8][1]) - 1]) | (df_mun['Transport and storage investments rate'].values[0] == dm_gva.d._levels[8][1][len(dm_gva.d._levels[8][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) - 1
        df_mun_v['Transport and storage investments rate'] = dm_gva.d._levels[8][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Transport and storage investments rate* to a value 0.562 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Transport and storage investments rate'].values[0] == dm_gva.d._levels[8][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) + 1
        df_mun_v['Transport and storage investments rate'] = dm_gva.d._levels[8][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) + 1
        down_v = dm_gva.d._levels[8][1].index(df_mun['Transport and storage investments rate'].values[0]) - 1

        df_mun_v['Transport and storage investments rate'] = dm_gva.d._levels[8][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Transport and storage investments rate* to a value lower than 0.562 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Transport and storage investments rate'] = dm_gva.d._levels[8][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Preschool children enrollment rate'].values[0] == dm_gva.d._levels[9][1][len(dm_gva.d._levels[9][1]) - 1]) | (df_mun['Preschool children enrollment rate'].values[0] == dm_gva.d._levels[9][1][len(dm_gva.d._levels[9][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) - 1
        df_mun_v['Preschool children enrollment rate'] = dm_gva.d._levels[9][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Preschool children enrollment rate* to a value between 2.086 and 2.682 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Preschool children enrollment rate'].values[0] == dm_gva.d._levels[9][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) + 1
        df_mun_v['Preschool children enrollment rate'] = dm_gva.d._levels[9][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) + 1
        down_v = dm_gva.d._levels[9][1].index(df_mun['Preschool children enrollment rate'].values[0]) - 1

        df_mun_v['Preschool children enrollment rate'] = dm_gva.d._levels[9][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Preschool children enrollment rate* to a value lower than 2.086 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Preschool children enrollment rate'] = dm_gva.d._levels[9][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
    if (df_mun['Doctors accessibility'].values[0] == dm_gva.d._levels[10][1][len(dm_gva.d._levels[10][1]) - 1]) | (df_mun['Doctors accessibility'].values[0] == dm_gva.d._levels[10][1][len(dm_gva.d._levels[10][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm_gva.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) - 1
        df_mun_v['Doctors accessibility'] = dm_gva.d._levels[10][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Doctors accessibility* to a value between 1.3 and 2.1 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))
    
    if df_mun['Doctors accessibility'].values[0] == dm_gva.d._levels[10][1][0]:
        to_write = False

        df_mun_v = df_mun.copy()
        up_v = dm_gva.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) + 1
        df_mun_v['Doctors accessibility'] = dm_gva.d._levels[10][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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
        up_v = dm_gva.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) + 1
        down_v = dm_gva.d._levels[10][1].index(df_mun['Doctors accessibility'].values[0]) - 1

        df_mun_v['Doctors accessibility'] = dm_gva.d._levels[10][1][down_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('A **decrease** in *Doctors accessibility* to a value lower than 1.3 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

        df_mun_v = df_mun.copy()
        df_mun_v['Doctors accessibility'] = dm_gva.d._levels[10][1][up_v]
        prediction_n = dm_gva.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            st.markdown('An **increase** in *Doctors accessibility* to a value lower than 2.1 leads to the outcome:')
            if prediction_n == 'Poor':
                annotated_text((prediction_n, '', 'red'))
            elif prediction_n == 'Good':
                annotated_text((prediction_n, '', 'lightgreen'))
            else:
                annotated_text((prediction_n, '', 'gray'))

st.image('img/image_3.png')