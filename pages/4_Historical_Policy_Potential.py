import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np

import dex_model as dm

st.set_page_config(layout='wide', page_title = 'Historical Net Interal Migration Policy Potential')

st.title('Historical Net Interal Migration Policy Potential')

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
    st.markdown('Please find below the projected outcome and the potential interventions:')

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

    st.markdown('---')
    st.subheader('Possible Interventions')
    st.markdown("This part of the page discuss the potential possibility of interventions on Net Internal migrations. We show potential interventions by changing the values of input attributes for one degree in improvement and one degree in deterioration. If a single intervention improves or deteriorates the outcome it will be presented in an appropriate color - red for the very negative net internal migrations (-4 or bigger emigration per 1,000 inhabitants), gray for negative (between -4 and 0 emigrations per 1,000 inhabitants), and green (positive migrations, or immigration).")

    ### Vehicles Density
    st.markdown('#### Vehicles Density')
    to_write = True
    if (df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 1]) | (df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Vehicles Density* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Vehicles Density* to a value between 10.272 and 21.286 **does not change** the outcome.')
    
    if df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Vehicles Density* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Vehicles Density* to a value between 10.272 and 21.286 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Vehicles Density* to a value lower than 10.272 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Vehicles Density* to a value greater than 21.286 **does not change** the outcome.')

    ### Motorcycle Density
    st.markdown('#### Motorcycle Density')
    to_write = True
    if (df_mun['Motorcycle Density'].values[0] == dm.d._levels[1][1][len(dm.d._levels[1][1]) - 1]) | (df_mun['Motorcycle Density'].values[0] == dm.d._levels[1][1][len(dm.d._levels[1][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Motorcycle Density* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Motorcycle Density* to a value between 0.281 and 0.762 **does not change** the outcome.')
    
    if df_mun['Motorcycle Density'].values[0] == dm.d._levels[1][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Motorcycle Density* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Motorcycle Density* to a value between 0.281 and 0.762 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Motorcycle Density* to a value lower than 0.281 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Motorcycle Density* to a value greater than 0.762 **does not change** the outcome.')
    
    ### Main Road Accessibility
    st.markdown('#### Main Roads Accessibility')
    to_write = True
    if (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 1]) | (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 2]):
        to_write = False
        st.markdown('The **improvement** in *Main Roads Accessibility* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Main Roads Accessibility* to a value between 0.693 and 1.676 **does not change** the outcome.')
    
    if df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Main Roads Accessibility* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Main Roads Accessibility* to a value between 0.693 and 1.676 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Main Roads Accessibility* to a value greater than 1.676 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Main Roads Accessibility* to a value lower than 0.693 **does not change** the outcome.')

    ### Local Roads Density
    st.markdown('#### Local Roads Density')
    to_write = True
    if (df_mun['Local Roads Density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 1]) | (df_mun['Local Roads Density'].values[0] == dm.d._levels[3][1][len(dm.d._levels[3][1]) - 2]):
        to_write = False
        st.markdown('The **increase** in *Local Roads Density* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Local Roads Density* to a value between 107.736 and 450.854 **does not change** the outcome.')
    
    if df_mun['Local Roads Density'].values[0] == dm.d._levels[3][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Local Roads Density* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Local Roads Density* to a value between 107.736 and 450.854 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Local Roads Density* to a value lower than 107.736 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Local Roads Density* to a value greater than 450.854 **does not change** the outcome.')

    ### Main Road Accessibility
    st.markdown('#### Main Roads Accessibility')
    to_write = True
    if (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 1]) | (df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][len(dm.d._levels[2][1]) - 2]):
        to_write = False
        st.markdown('The **improvement** in *Main Roads Accessibility* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Main Roads Accessibility* to a value between 0.693 and 1.676 **does not change** the outcome.')
    
    if df_mun['Main Roads Accessibility'].values[0] == dm.d._levels[2][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Main Roads Accessibility* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Main Roads Accessibility* to a value between 0.693 and 1.676 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Main Roads Accessibility* to a value greater than 1.676 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Main Roads Accessibility* to a value lower than 0.693 **does not change** the outcome.')

    ### Primary School Attendance
    st.markdown('#### Primary School Attendance')
    to_write = True
    if (df_mun['Primary School Attendance'].values[0] == dm.d._levels[4][1][len(dm.d._levels[4][1]) - 1]):
        to_write = False
        st.markdown('The **increase** in *Primary School Attendance* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Primary School Attendance* to a value between 365.412 and 590.641 **does not change** the outcome.')
    
    if df_mun['Primary School Attendance'].values[0] == dm.d._levels[4][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Primary School Attendance* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Primary School Attendance* to a value between 365.412 and 590.641 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Primary School Attendance* to a value lower than 365.412 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Primary School Attendance* to a value greater than 590.641 **does not change** the outcome.')

    ### Secondary School Attendance
    st.markdown('#### Secondary School Attendance')
    to_write = True
    if (df_mun['Secondary School Attendance'].values[0] == dm.d._levels[5][1][len(dm.d._levels[5][1]) - 1]):
        to_write = False
        st.markdown('The **increase** in *Secondary School Attendance* will not influence the outcome.')

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
        else:
            st.markdown('A **decrease** in *Secondary School Attendance* to a value between 269.804 and 435.033 **does not change** the outcome.')
    
    if df_mun['Secondary School Attendance'].values[0] == dm.d._levels[5][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Secondary School Attendance* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Secondary School Attendance* to a value between 269.804 and 435.033 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Secondary School Attendance* to a value lower than 269.804 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Secondary School Attendance* to a value greater than 435.033 **does not change** the outcome.')

    ### Assistance and Care Allowance Share
    st.markdown('#### Assistance and Care Allowance Share')
    to_write = True
    if (df_mun['Assistance and Care Allowance Share'].values[0] == dm.d._levels[6][1][len(dm.d._levels[6][1]) - 1]) | (df_mun['Assistance and Care Allowance Share'].values[0] == dm.d._levels[6][1][len(dm.d._levels[6][1]) - 2]):
        to_write = False
        st.markdown('The **improvement** in *Assistance and Care Allowance Share* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Assistance and Care Allowance Share* to a value between 0.5 and 0.6 **does not change** the outcome.')
    
    if df_mun['Assistance and Care Allowance Share'].values[0] == dm.d._levels[6][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Assistance and Care Allowance Share* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Assistance and Care Allowance Share* to a value between 0.5 and 0.6 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Assistance and Care Allowance Share* to a value greater than 0.6 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Assistance and Care Allowance Share* to a value lower than 0.5 **does not change** the outcome.')

    ### Poverty Share
    st.markdown('#### Poverty Share')
    to_write = True
    if (df_mun['Poverty Share'].values[0] == dm.d._levels[7][1][len(dm.d._levels[7][1]) - 1]) | (df_mun['Poverty Share'].values[0] == dm.d._levels[7][1][len(dm.d._levels[7][1]) - 2]):
        to_write = False
        st.markdown('The **improvement** in *Poverty Share* will not influence the outcome.')

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
        else:
            st.markdown('A **deterioration** in *Poverty Share* to a value between 8.1 and 12.9 **does not change** the outcome.')
    
    if df_mun['Poverty Share'].values[0] == dm.d._levels[7][1][0]:
        to_write = False
        st.markdown('The **deterioration** in *Poverty Share* will not influence the outcome.')

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
        else:
            st.markdown('An **improvement** in *Poverty Share* to a value between 8.1 and 12.9 **does not change** the outcome.')
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
        else:
            st.markdown('A **deterioration** in *Poverty Share* to a value greater than 12.9 **does not change** the outcome.')

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
        else:
            st.markdown('An **improvement** in *Poverty Share* to a value lower than 8.1 **does not change** the outcome.')
    
    ### Doctors Accessibility
    st.markdown('#### Doctors Accessibility')
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
        else:
            st.markdown('A **decrease** in *Doctors Accessibility* to a value between 1.3 and 2.1 **does not change** the outcome.')
    
    if df_mun['Doctors Accessibility'].values[0] == dm.d._levels[8][1][0]:
        to_write = False
        st.markdown('The **decrease** in *Doctors Accessibility* will not influence the outcome.')

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
        else:
            st.markdown('An **increase** in *Doctors Accessibility* to a value between 1.3 and 2.1 **does not change** the outcome.')
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
        else:
            st.markdown('A **decrease** in *Doctors Accessibility* to a value lower than 1.3 **does not change** the outcome.')

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
        else:
            st.markdown('An **increase** in *Doctors Accessibility* to a value greater than 2.1 **does not change** the outcome.')
