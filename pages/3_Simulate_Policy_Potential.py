import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np

import dex_model as dm

st.set_page_config(layout='wide', page_title = 'Simulate Net Interal Migration Policy Potential')

st.title('Simulate Net Interal Migration Policy Potential')

st.markdown(
    '''
    This page aims to helping you see the effects of the policy on the municipality.
    '''
)

df_im = pd.read_csv('data/im.csv')
LSGs = np.unique(df_im['Area Name'])

selected_lsg = st.selectbox(label='Please select municipality', options=LSGs)
df_mun = df_im.loc[(df_im['Area Name'] == selected_lsg) & (df_im['Time Period'] == 2021), :]

st.markdown('---')
st.subheader('Please insert appropriate values')

col11, col21, col31 = st.columns(3)
col12, col22, col32 = st.columns(3)
col13, col23, col33 = st.columns(3)
col14, col24, col34 = st.columns(3)
col15, col25, col35 = st.columns(3)
col16, col26, col36 = st.columns(3)
col17, col27, col37 = st.columns(3)
col18, col28, col38 = st.columns(3)
col19, col29, col39 = st.columns(3)
col110, col210, col310 = st.columns(3)

with col11:
    st.markdown('Insert needed values to estimate Net Interal Migration Policy Potential')
with col12:
    veh_dens = st.number_input(label='Vehicles Density', value = 0 if np.isnan(df_mun['Vehicles Density'].values[0]) else df_mun['Vehicles Density'].values[0])
with col13:
    motor_dens = st.number_input(label='Motorcycle Density', value = 0 if np.isnan(df_mun['Motorcycle Density'].values[0]) else df_mun['Motorcycle Density'].values[0])
with col14:
    main_road_acc = st.number_input(label='Main Roads Accessibility', value = 0 if np.isnan(df_mun['Main Roads Accessibility'].values[0]) else df_mun['Main Roads Accessibility'].values[0])
with col15:
    local_road_acc = st.number_input(label='Local Roads Density', value = 0 if np.isnan(df_mun['Local Roads Density'].values[0]) else df_mun['Local Roads Density'].values[0])
with col16:
    prim_school_att = st.number_input(label='Primary School Attendance', value = 0 if np.isnan(df_mun['Primary School Attendance'].values[0]) else df_mun['Primary School Attendance'].values[0])
with col17:
    sec_school_att = st.number_input(label='Secondary School Attendance', value = 0 if np.isnan(df_mun['Secondary School Attendance'].values[0]) else df_mun['Secondary School Attendance'].values[0])
with col18:
    assistance_share = st.number_input(label='Assistance and Care Allowance Share', value = 0 if np.isnan(df_mun['Assistance and Care Allowance Share'].values[0]) else df_mun['Assistance and Care Allowance Share'].values[0])
with col19:
    poverty_share = st.number_input(label='Poverty Share', value = 0 if np.isnan(df_mun['Poverty Share'].values[0]) else df_mun['Poverty Share'].values[0])
with col110:
    doctors_acc = st.number_input(label='Doctors Accessibility', value = 0 if np.isnan(df_mun['Doctors Accessibility'].values[0]) else df_mun['Doctors Accessibility'].values[0])

with col21:
    st.markdown('This area shows the value of the attribute for the selected municipality in 2021')

with col22:
    st.markdown('Vehicles Density:')
    if df_mun['Vehicles Density'].values[0] <= 10.272:
        annotated_text((f"{df_mun['Vehicles Density'].values[0]}", '', 'red'))
    elif df_mun['Vehicles Density'].values[0] > 21.286:
        annotated_text((f"{df_mun['Vehicles Density'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Vehicles Density'].values[0]) else df_mun['Motorcycle Density'].values[0]}")

with col23:
    st.markdown('Motorcycle Density:')
    if df_mun['Motorcycle Density'].values[0] <= 0.281:
        annotated_text((f"{df_mun['Motorcycle Density'].values[0]}", '', 'red'))
    elif df_mun['Motorcycle Density'].values[0] > 0.762:
        annotated_text((f"{df_mun['Motorcycle Density'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Motorcycle Density'].values[0]) else df_mun['Motorcycle Density'].values[0]}")

with col24:
    st.markdown('Main Roads Accessibility:')
    if df_mun['Main Roads Accessibility'].values[0] > 1.676:
        annotated_text((f"{df_mun['Main Roads Accessibility'].values[0]}", '', 'red'))
    elif df_mun['Main Roads Accessibility'].values[0] <= 0.693:
        annotated_text((f"{df_mun['Main Roads Accessibility'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Main Roads Accessibility'].values[0]) else df_mun['Main Roads Accessibility'].values[0]}")
    
with col25:
    st.markdown('Local Roads Density:')
    if df_mun['Local Roads Density'].values[0] <= 107.736:
        annotated_text((f"{df_mun['Local Roads Density'].values[0]}", '', 'red'))
    elif df_mun['Local Roads Density'].values[0] > 450.854:
        annotated_text((f"{df_mun['Local Roads Density'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Local Roads Density'].values[0]) else df_mun['Local Roads Density'].values[0]}")

with col26:
    st.markdown('Primary School Attendance:')
    if df_mun['Primary School Attendance'].values[0] <= 365.412:
        annotated_text((f"{df_mun['Primary School Attendance'].values[0]}", '', 'red'))
    elif df_mun['Primary School Attendance'].values[0] > 590.641:
        annotated_text((f"{df_mun['Primary School Attendance'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Primary School Attendance'].values[0]) else df_mun['Primary School Attendance'].values[0]}")
    
with col27:
    st.markdown('Secondary School Attendance:')
    if df_mun['Secondary School Attendance'].values[0] <= 269.804:
        annotated_text((f"{df_mun['Secondary School Attendance'].values[0]}", '', 'red'))
    elif df_mun['Secondary School Attendance'].values[0] > 435.033:
        annotated_text((f"{df_mun['Secondary School Attendance'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Secondary School Attendance'].values[0]) else df_mun['Secondary School Attendance'].values[0]}")
    
with col28:
    st.markdown('Assistance and Care Allowance Share:')
    if df_mun['Assistance and Care Allowance Share'].values[0] > 0.6:
        annotated_text((f"{df_mun['Assistance and Care Allowance Share'].values[0]}", '', 'red'))
    elif df_mun['Assistance and Care Allowance Share'].values[0] <= 0.5:
        annotated_text((f"{df_mun['Assistance and Care Allowance Share'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Assistance and Care Allowance Share'].values[0]) else df_mun['Assistance and Care Allowance Share'].values[0]}")
    
with col29:
    st.markdown('Poverty Share:')
    if df_mun['Poverty Share'].values[0] > 12.9:
        annotated_text((f"{df_mun['Poverty Share'].values[0]}", '', 'red'))
    elif df_mun['Poverty Share'].values[0] <= 8.1:
        annotated_text((f"{df_mun['Poverty Share'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Poverty Share'].values[0]) else df_mun['Poverty Share'].values[0]}")
    
with col210:
    st.markdown('Doctors Accessibility:')
    if df_mun['Doctors Accessibility'].values[0] <= 1.3:
        annotated_text((f"{df_mun['Doctors Accessibility'].values[0]}", '', 'red'))
    elif df_mun['Doctors Accessibility'].values[0] > 2.1:
        annotated_text((f"{df_mun['Doctors Accessibility'].values[0]}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(df_mun['Doctors Accessibility'].values[0]) else df_mun['Doctors Accessibility'].values[0]}")

with col31:
    st.markdown(
        '''
        Below are the descriptions of the attributes.
        '''
    )

with col32:
    st.markdown(
        '''
        **Vehicles density** - The number of registered passenger vehicles per square meter of the municipality.
        '''
    )

with col33:
    st.markdown(
        '''
        **Motorcycles density** - The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.
        '''
    )

with col34:
    st.markdown(
        '''
        **Main roads accessibility** - The total length (in kilometers) of 1st-class road segments without highways divided by 1,000 inhabitants of the municipality.
        '''
    )

with col35:
    st.markdown(
        '''
        **Local roads density** - The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of the municipality.
        '''
    )

with col36:
    st.markdown(
        '''
        **Primary school's attendance** - The average number of children in primary school that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 7 and 14.
        '''
    )

with col37:
    st.markdown(
        '''
        **Secondary school's attendance** - The average number of children in secondary school (both vocational and general) that are within the municipality.  To calculate the number of children, we used an estimate of the number of people between the ages of 15 and 18.
        '''
    )

with col38:
    st.markdown(
        '''
        **Assistance and care allowance share** - Percentage of people within one municipality that use an increased allowance for assistance and care of another person.
        '''
    )

with col39:
    st.markdown(
        '''
        **Poverty share** - Percentage of people within one municipality that uses social protection.
        '''
    )

with col310:
    st.markdown(
        '''
        **Doctors accessibility** - The total number of doctors within a municipality divided by the 1,000 inhabitants.
        '''
    )

v = '2'
if veh_dens <= 10.272:
    v = '1'
elif veh_dens > 21.286:
    v = '3'

m = '2'
if motor_dens <= 0.281:
    m = '1'
elif motor_dens > 0.762:
    m = '3'

mra = '2'
if main_road_acc <= 0.693:
    mra = '1'
elif main_road_acc > 1.676:
    mra = '3'

lra = '2'
if local_road_acc <= 107.736:
    lra = '1'
elif local_road_acc > 450.854:
    lra = '3'

psa = '2'
if prim_school_att <= 365.412:
    psa = '1'
elif prim_school_att > 590.641:
    psa = '3'

ssa = '2'
if sec_school_att <= 269.804:
    ssa = '1'
elif sec_school_att > 435.033:
    ssa = '3'

ashare = '2'
if assistance_share <= 0.5:
    ashare = '1'
elif assistance_share > 0.6:
    ashare = '3'

pshare = '2'
if poverty_share <= 8.1:
    pshare = '1'
elif poverty_share > 12.9:
    pshare = '3'

pshare = '2'
if poverty_share <= 8.1:
    pshare = '1'
elif poverty_share > 12.9:
    pshare = '3'

da = '2'
if doctors_acc <= 1.3:
    da = '1'
elif doctors_acc > 2.1:
    da = '3'

st.markdown('---')
st.subheader('Prediction Outcome')

st.markdown('The prediction of the Net Internal Migrations based on the inserted values is:')

to_predict = [v, m, mra, lra, psa, ssa, ashare, pshare, da]
to_predict = pd.DataFrame([to_predict], columns=['Vehicles Density', 'Motorcycle Density', 'Main Roads Accessibility', 'Local Roads Density', 'Primary School Attendance', 'Secondary School Attendance', 'Assistance and Care Allowance Share', 'Poverty Share', 'Doctors Accessibility'])

prediction = dm.d.predict(to_predict).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
if prediction == 'Poor':
    annotated_text((prediction, '', 'red'))
    st.markdown('This value signals the **very high emigration** (greater than 4 persons on 1,000 inhabitants) from the municipality')
elif prediction == 'Good':
    annotated_text((prediction, '', 'lightgreen'))
    st.markdown('This value signals the **immigration** to the municipality')
else:
    annotated_text((prediction, '', 'gray'))
    st.markdown('This value signals the emigration (between 0 and 4 persons on 1,000 inhabitants) from the municipality')