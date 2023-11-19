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
df_year = df_im.loc[(df_im['Time Period'] == 2021), :]
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
    st.markdown(
        '''
        This area shows the value of the attribute for the selected municipality in 2021. 
        
        (Values within the brackets represent the average value of municipalities for the selected attribute)
        '''
    )

with col22:
    st.markdown('Vehicles Density:')
    if df_mun['Vehicles Density'].values[0] <= 10.272:
        annotated_text((f"{round(df_mun['Vehicles Density'].values[0], 3)}", '', 'red'))
    elif df_mun['Vehicles Density'].values[0] > 21.286:
        annotated_text((f"{round(df_mun['Vehicles Density'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Vehicles Density'].values[0], 3)) else round(df_mun['Vehicles Density'].values[0], 3)}")
    annotated_text((f'({round(df_year["Vehicles Density"].mean(), 3)})', 'Serbia', ''))

with col23:
    st.markdown('Motorcycle Density:')
    if df_mun['Motorcycle Density'].values[0] <= 0.281:
        annotated_text((f"{round(df_mun['Motorcycle Density'].values[0], 3)}", '', 'red'))
    elif df_mun['Motorcycle Density'].values[0] > 0.762:
        annotated_text((f"{round(df_mun['Motorcycle Density'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Motorcycle Density'].values[0], 3)) else round(df_mun['Motorcycle Density'].values[0], 3)}")
    annotated_text((f'({round(df_year["Motorcycle Density"].mean(), 3)})', 'Serbia', ''))

with col24:
    st.markdown('Main Roads Accessibility:')
    if df_mun['Main Roads Accessibility'].values[0] > 1.676:
        annotated_text((f"{round(df_mun['Main Roads Accessibility'].values[0], 3)}", '', 'red'))
    elif df_mun['Main Roads Accessibility'].values[0] <= 0.693:
        annotated_text((f"{round(df_mun['Main Roads Accessibility'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Main Roads Accessibility'].values[0], 3)) else round(df_mun['Main Roads Accessibility'].values[0], 3)}")
    annotated_text((f'({round(df_year["Main Roads Accessibility"].mean(), 3)})', 'Serbia', ''))
    
with col25:
    st.markdown('Local Roads Density:')
    if df_mun['Local Roads Density'].values[0] <= 107.736:
        annotated_text((f"{round(df_mun['Local Roads Density'].values[0], 3)}", '', 'red'))
    elif df_mun['Local Roads Density'].values[0] > 450.854:
        annotated_text((f"{round(df_mun['Local Roads Density'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Local Roads Density'].values[0], 3)) else round(df_mun['Local Roads Density'].values[0], 3)}")
    annotated_text((f'({round(df_year["Local Roads Density"].mean(), 3)})', 'Serbia', ''))

with col26:
    st.markdown('Primary School Attendance:')
    if df_mun['Primary School Attendance'].values[0] <= 365.412:
        annotated_text((f"{round(df_mun['Primary School Attendance'].values[0], 3)}", '', 'red'))
    elif df_mun['Primary School Attendance'].values[0] > 590.641:
        annotated_text((f"{round(df_mun['Primary School Attendance'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Primary School Attendance'].values[0], 3)) else round(df_mun['Primary School Attendance'].values[0], 3)}")
    annotated_text((f'({round(df_year["Primary School Attendance"].mean(), 3)})', 'Serbia', ''))
    
with col27:
    st.markdown('Secondary School Attendance:')
    if df_mun['Secondary School Attendance'].values[0] <= 269.804:
        annotated_text((f"{round(df_mun['Secondary School Attendance'].values[0], 3)}", '', 'red'))
    elif df_mun['Secondary School Attendance'].values[0] > 435.033:
        annotated_text((f"{round(df_mun['Secondary School Attendance'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Secondary School Attendance'].values[0], 3)) else round(df_mun['Secondary School Attendance'].values[0], 3)}")
    annotated_text((f'({round(df_year["Secondary School Attendance"].mean(), 3)})', 'Serbia', ''))
    
with col28:
    st.markdown('Assistance and Care Allowance Share:')
    if df_mun['Assistance and Care Allowance Share'].values[0] > 0.6:
        annotated_text((f"{round(df_mun['Assistance and Care Allowance Share'].values[0], 3)}", '', 'red'))
    elif df_mun['Assistance and Care Allowance Share'].values[0] <= 0.5:
        annotated_text((f"{round(df_mun['Assistance and Care Allowance Share'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Assistance and Care Allowance Share'].values[0], 3)) else round(df_mun['Assistance and Care Allowance Share'].values[0], 3)}")
    annotated_text((f'({round(df_year["Assistance and Care Allowance Share"].mean(), 3)})', 'Serbia', ''))
    
with col29:
    st.markdown('Poverty Share:')
    if df_mun['Poverty Share'].values[0] > 12.9:
        annotated_text((f"{round(df_mun['Poverty Share'].values[0], 3)}", '', 'red'))
    elif df_mun['Poverty Share'].values[0] <= 8.1:
        annotated_text((f"{round(df_mun['Poverty Share'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Poverty Share'].values[0], 3)) else round(df_mun['Poverty Share'].values[0], 3)}")
    annotated_text((f'({round(df_year["Poverty Share"].mean(), 3)})', 'Serbia', ''))
    
with col210:
    st.markdown('Doctors Accessibility:')
    if df_mun['Doctors Accessibility'].values[0] <= 1.3:
        annotated_text((f"{round(df_mun['Doctors Accessibility'].values[0], 3)}", '', 'red'))
    elif df_mun['Doctors Accessibility'].values[0] > 2.1:
        annotated_text((f"{round(df_mun['Doctors Accessibility'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Doctors Accessibility'].values[0], 3)) else round(df_mun['Doctors Accessibility'].values[0], 3)}")
    annotated_text((f'({round(df_year["Doctors Accessibility"].mean(), 3)})', 'Serbia', ''))

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

pred = dm.d.predict(to_predict, return_intermediate=True)

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
    st.markdown('Vehicles Density and Motorcycle Density lead to - **Vehicles**')
with col14:
    st.markdown('Main Roads Accessibility')
with col15:
    st.markdown('Local Roads Density')
with col16:
    st.markdown('Main Roads Accessibility and Local Roads Density lead to - **Roads**')
with col17:
    st.markdown('Primary School Attendance')
with col18:
    st.markdown('Secondary School Attendance')
with col19:
    st.markdown('Primary School Attendance and Secondary School Attendance lead to - **School**')
with col110:
    st.markdown('Assistance and Care Allowance Share')
with col111:
    st.markdown('Poverty Share')
with col112:
    st.markdown('Assistance and Care Allowance Share and Poverty Share lead to - **Social Factors**')
with col113:
    st.markdown('Doctors Accessibility')
with col114:
    st.markdown('Social Factors and Doctors Accessibility lead to - **Health and Social Factors**')
with col115:
    st.markdown('Schools and Health and Social Factors lead to - **Social Determinants**')

with col21:
    annotated_text((pred['Vehicles Density'], '', pred['Vehicles Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col22:
    annotated_text((pred['Motorcycle Density'], '', pred['Motorcycle Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col23:
    annotated_text((pred['Vehicles'], '', pred['Vehicles'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col24:
    annotated_text((pred['Main Roads Accessibility'], '', pred['Main Roads Accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
with col25:
    annotated_text((pred['Local Roads Density'], '', pred['Local Roads Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col26:
    annotated_text((pred['Roads'], '', pred['Roads'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col27:
    annotated_text((pred['Primary School Attendance'], '', pred['Primary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col28:
    annotated_text((pred['Secondary School Attendance'], '', pred['Secondary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col29:
    annotated_text((pred['School'], '', pred['School'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col210:
    annotated_text((pred['Assistance and Care Allowance Share'], '', pred['Assistance and Care Allowance Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
with col211:
    annotated_text((pred['Poverty Share'], '', pred['Poverty Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
with col212:
    annotated_text((pred['Social Factors'], '', pred['Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col213:
    annotated_text((pred['Doctors Accessibility'], '', pred['Doctors Accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col214:
    annotated_text((pred['Health and Social Factors'], '', pred['Health and Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col215:
    annotated_text((pred['Social Determinants'], '', pred['Social Determinants'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))


prediction = dm.d.predict(to_predict).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
st.markdown('**The predicted outcome is:**')
if prediction == 'Poor':
    annotated_text((prediction, '', 'red'))
    st.markdown('This value signals the **very high emigration** (greater than 4 persons on 1,000 inhabitants) from the municipality')
elif prediction == 'Good':
    annotated_text((prediction, '', 'lightgreen'))
    st.markdown('This value signals the **immigration** to the municipality')
else:
    annotated_text((prediction, '', 'gray'))
    st.markdown('This value signals the emigration (between 0 and 4 persons on 1,000 inhabitants) from the municipality')

st.markdown('---')
st.subheader('Possible Interventions')
st.markdown("This part of the page discuss the potential possibility of interventions on Net Internal migrations. We show potential interventions by changing the values of input attributes for one degree in improvement and one degree in deterioration. If a single intervention improves or deteriorates the outcome it will be presented in an appropriate color - red for the very negative net internal migrations (-4 or bigger emigration per 1,000 inhabitants), gray for negative (between -4 and 0 emigrations per 1,000 inhabitants), and green (positive migrations, or immigration).")

df_mun = to_predict.copy()
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