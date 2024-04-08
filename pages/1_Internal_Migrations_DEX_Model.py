import streamlit as st

import numpy as np
import pandas as pd

from annotated_text import annotated_text

import dex_model as dm
import page_util
import page_util_im

st.set_page_config(layout='wide', 
                   page_title = 'Net Internal Migration DEX Model')

page_util.hide_table_index()

st.title('Net Internal Migration DEX Model')

tab__1, tab__2, tab__3, tab__4, tab__5 = st.tabs(["Internal Migrations DEX Model", "Data Exploration", "Visualize Data", "Historical Policy Outcomes", "Simulate Policy Potential"])

with tab__1:
    with st.expander('Introduction'):
        page_util_im.introduction_text()

    st.markdown('---')
    st.subheader('Attribute Description')
    page_util_im.attribute_description()

    st.markdown('---')
    st.subheader('Model Hierarchy')
    page_util_im.model_hierarchy()

    st.markdown('---')
    st.subheader('Attribute Values')
    page_util_im.attribute_values()

    st.markdown('---')
    st.subheader('Decision Rules')
    page_util_im.decision_rules()

with tab__2:
    page_util_im.data_exploration()

with tab__3:
    page_util_im.data_visualization()
with tab__4:
    page_util_im.historical_info()
with tab__5:
    st.markdown(
        '''
        This page aims to helping you see the effects of the policy on the municipality.
        '''
    )

    df_im = pd.read_csv('data/im.csv')
    LSGs = np.unique(df_im['Area Name'])

    selected_lsg = st.selectbox(label='Please select municipality', options=LSGs, index=89)
    df_year = df_im.loc[(df_im['Time Period'] == 2021), :]
    df_mun = df_im.loc[(df_im['Area Name'] == selected_lsg) & (df_im['Time Period'] == 2021), :]

    st.markdown('---')
    st.subheader('Please insert appropriate values')

    col11, col21, col31 = st.columns(3)
    st.markdown('---')
    col12, col22, col32 = st.columns(3)
    st.markdown('---')
    col13, col23, col33 = st.columns(3)
    st.markdown('---')
    col14, col24, col34 = st.columns(3)
    st.markdown('---')
    col15, col25, col35 = st.columns(3)
    st.markdown('---')
    col16, col26, col36 = st.columns(3)
    st.markdown('---')
    col17, col27, col37 = st.columns(3)
    st.markdown('---')
    col18, col28, col38 = st.columns(3)
    st.markdown('---')
    col19, col29, col39 = st.columns(3)
    st.markdown('---')
    col110, col210, col310 = st.columns(3)
    st.markdown('---')

    with col11:
        st.markdown('Insert needed values to estimate Net Interal Migration Policy Potential')
    with col21:
        st.markdown(
            '''
            This area shows the value of the attribute for the selected municipality in 2021. 
            
            (Values within the brackets represent the average value of municipalities for the selected attribute)
            '''
        )
    with col31:
        st.markdown(
            '''
            Below are the descriptions of the attributes.
            '''
        )

    with col12:
        veh_dens = st.number_input(label='Vehicles Density', value = 0 if np.isnan(df_mun['Vehicles Density'].values[0]) else df_mun['Vehicles Density'].values[0])
    with col22:
        if df_mun['Vehicles Density'].values[0] <= 10.272:
            annotated_text((f"{round(df_mun['Vehicles Density'].values[0], 3)}", '', 'red'))
        elif df_mun['Vehicles Density'].values[0] > 21.286:
            annotated_text((f"{round(df_mun['Vehicles Density'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Vehicles Density'].values[0], 3)) else round(df_mun['Vehicles Density'].values[0], 3)}")
        annotated_text((f'({round(df_year["Vehicles Density"].mean(), 3)})', 'Serbia', ''))
    with col32:
        st.markdown(
            '''
            **Vehicles density** - The number of registered passenger vehicles per square meter of the municipality.
            '''
        )
    
    with col13:
        motor_dens = st.number_input(label='Motorcycle Density', value = 0 if np.isnan(df_mun['Motorcycle Density'].values[0]) else df_mun['Motorcycle Density'].values[0])
    with col23:
        if df_mun['Motorcycle Density'].values[0] <= 0.281:
            annotated_text((f"{round(df_mun['Motorcycle Density'].values[0], 3)}", '', 'red'))
        elif df_mun['Motorcycle Density'].values[0] > 0.762:
            annotated_text((f"{round(df_mun['Motorcycle Density'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Motorcycle Density'].values[0], 3)) else round(df_mun['Motorcycle Density'].values[0], 3)}")
        annotated_text((f'({round(df_year["Motorcycle Density"].mean(), 3)})', 'Serbia', ''))
    with col33:
        st.markdown(
            '''
            **Motorcycles density** - The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.
            '''
        )
    
    with col14:
        main_road_acc = st.number_input(label='Main Roads Accessibility', value = 0 if np.isnan(df_mun['Main Roads Accessibility'].values[0]) else df_mun['Main Roads Accessibility'].values[0])
    with col24:
        if df_mun['Main Roads Accessibility'].values[0] > 1.676:
            annotated_text((f"{round(df_mun['Main Roads Accessibility'].values[0], 3)}", '', 'red'))
        elif df_mun['Main Roads Accessibility'].values[0] <= 0.693:
            annotated_text((f"{round(df_mun['Main Roads Accessibility'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Main Roads Accessibility'].values[0], 3)) else round(df_mun['Main Roads Accessibility'].values[0], 3)}")
        annotated_text((f'({round(df_year["Main Roads Accessibility"].mean(), 3)})', 'Serbia', ''))
    with col34:
        st.markdown(
            '''
            **Main roads accessibility** - The total length (in kilometers) of 1st-class road segments without highways divided by 1,000 inhabitants of the municipality.
            '''
        )
    
    with col15:
        local_road_acc = st.number_input(label='Local Roads Density', value = 0 if np.isnan(df_mun['Local Roads Density'].values[0]) else df_mun['Local Roads Density'].values[0])
    with col25:
        if df_mun['Local Roads Density'].values[0] <= 107.736:
            annotated_text((f"{round(df_mun['Local Roads Density'].values[0], 3)}", '', 'red'))
        elif df_mun['Local Roads Density'].values[0] > 450.854:
            annotated_text((f"{round(df_mun['Local Roads Density'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Local Roads Density'].values[0], 3)) else round(df_mun['Local Roads Density'].values[0], 3)}")
        annotated_text((f'({round(df_year["Local Roads Density"].mean(), 3)})', 'Serbia', ''))
    with col35:
        st.markdown(
            '''
            **Local roads density** - The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of the municipality.
            '''
        )
    
    with col16:
        prim_school_att = st.number_input(label='Primary School Attendance', value = 0 if np.isnan(df_mun['Primary School Attendance'].values[0]) else df_mun['Primary School Attendance'].values[0])
    with col26:
        if df_mun['Primary School Attendance'].values[0] <= 365.412:
            annotated_text((f"{round(df_mun['Primary School Attendance'].values[0], 3)}", '', 'red'))
        elif df_mun['Primary School Attendance'].values[0] > 590.641:
            annotated_text((f"{round(df_mun['Primary School Attendance'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Primary School Attendance'].values[0], 3)) else round(df_mun['Primary School Attendance'].values[0], 3)}")
        annotated_text((f'({round(df_year["Primary School Attendance"].mean(), 3)})', 'Serbia', ''))
    with col36:
        st.markdown(
            '''
            **Primary school's attendance** - The average number of children in primary school that are within the municipality. To calculate the number of children, we used an estimate of the number of people between the ages of 7 and 14.
            '''
        )
    
    with col17:
        sec_school_att = st.number_input(label='Secondary School Attendance', value = 0 if np.isnan(df_mun['Secondary School Attendance'].values[0]) else df_mun['Secondary School Attendance'].values[0])
    with col27:
        if df_mun['Secondary School Attendance'].values[0] <= 269.804:
            annotated_text((f"{round(df_mun['Secondary School Attendance'].values[0], 3)}", '', 'red'))
        elif df_mun['Secondary School Attendance'].values[0] > 435.033:
            annotated_text((f"{round(df_mun['Secondary School Attendance'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Secondary School Attendance'].values[0], 3)) else round(df_mun['Secondary School Attendance'].values[0], 3)}")
        annotated_text((f'({round(df_year["Secondary School Attendance"].mean(), 3)})', 'Serbia', ''))
    with col37:
        st.markdown(
            '''
            **Secondary school's attendance** - The average number of children in secondary school (both vocational and general) that are within the municipality.  To calculate the number of children, we used an estimate of the number of people between the ages of 15 and 18.
            '''
        )
    
    with col18:
        assistance_share = st.number_input(label='Assistance and Care Allowance Share', value = 0 if np.isnan(df_mun['Assistance and Care Allowance Share'].values[0]) else df_mun['Assistance and Care Allowance Share'].values[0])
    with col28:
        if df_mun['Assistance and Care Allowance Share'].values[0] > 0.6:
            annotated_text((f"{round(df_mun['Assistance and Care Allowance Share'].values[0], 3)}", '', 'red'))
        elif df_mun['Assistance and Care Allowance Share'].values[0] <= 0.5:
            annotated_text((f"{round(df_mun['Assistance and Care Allowance Share'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Assistance and Care Allowance Share'].values[0], 3)) else round(df_mun['Assistance and Care Allowance Share'].values[0], 3)}")
        annotated_text((f'({round(df_year["Assistance and Care Allowance Share"].mean(), 3)})', 'Serbia', ''))
    with col38:
        st.markdown(
            '''
            **Assistance and care allowance share** - Percentage of people within one municipality that use an increased allowance for assistance and care of another person.
            '''
        )
    
    with col19:
        poverty_share = st.number_input(label='Poverty Share', value = 0 if np.isnan(df_mun['Poverty Share'].values[0]) else df_mun['Poverty Share'].values[0])
    with col29:
        if df_mun['Poverty Share'].values[0] > 12.9:
            annotated_text((f"{round(df_mun['Poverty Share'].values[0], 3)}", '', 'red'))
        elif df_mun['Poverty Share'].values[0] <= 8.1:
            annotated_text((f"{round(df_mun['Poverty Share'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Poverty Share'].values[0], 3)) else round(df_mun['Poverty Share'].values[0], 3)}")
        annotated_text((f'({round(df_year["Poverty Share"].mean(), 3)})', 'Serbia', ''))
    with col39:
        st.markdown(
            '''
            **Poverty share** - Percentage of people within one municipality that uses social protection.
            '''
        )
    
    with col110:
        doctors_acc = st.number_input(label='Doctors Accessibility', value = 0 if np.isnan(df_mun['Doctors Accessibility'].values[0]) else df_mun['Doctors Accessibility'].values[0])
    with col210:
        if df_mun['Doctors Accessibility'].values[0] <= 1.3:
            annotated_text((f"{round(df_mun['Doctors Accessibility'].values[0], 3)}", '', 'red'))
        elif df_mun['Doctors Accessibility'].values[0] > 2.1:
            annotated_text((f"{round(df_mun['Doctors Accessibility'].values[0], 3)}", '', 'lightgreen'))
        else:
            st.markdown(f"{0 if np.isnan(round(df_mun['Doctors Accessibility'].values[0], 3)) else round(df_mun['Doctors Accessibility'].values[0], 3)}")
        annotated_text((f'({round(df_year["Doctors Accessibility"].mean(), 3)})', 'Serbia', ''))
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

    st.subheader('Prediction Outcome')

    st.markdown('The prediction of the Net Internal Migrations based on the inserted values is:')

    to_predict = [v, m, mra, lra, psa, ssa, ashare, pshare, da]
    to_predict = pd.DataFrame([to_predict], columns=['Vehicles Density', 'Motorcycle Density', 'Main Roads Accessibility', 'Local Roads Density', 'Primary School Attendance', 'Secondary School Attendance', 'Assistance and Care Allowance Share', 'Poverty Share', 'Doctors Accessibility'])

    pred = dm.d.predict(to_predict, return_intermediate=True)

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

    col10, col20, col30, col40 = st.columns(4)
    st.markdown('---')
    col11, col21, col31, col41 = st.columns(4)
    st.markdown('---')
    col12, col22, col32, col42 = st.columns(4)
    st.markdown('---')
    col13, col23, col33, col43 = st.columns(4)
    st.markdown('---')
    col14, col24, col34, col44 = st.columns(4)
    st.markdown('---')
    col15, col25, col35, col45 = st.columns(4)
    st.markdown('---')
    col16, col26, col36, col46 = st.columns(4)
    st.markdown('---')
    col17, col27, col37, col47 = st.columns(4)
    st.markdown('---')
    col18, col28, col38, col48 = st.columns(4)
    st.markdown('---')
    col19, col29, col39, col49 = st.columns(4)
    st.markdown('---')
    col110, col210, col310, col410 = st.columns(4)
    st.markdown('---')
    col111, col211, col311, col411 = st.columns(4)
    st.markdown('---')
    col112, col212, col312, col412 = st.columns(4)
    st.markdown('---')
    col113, col213, col313, col413 = st.columns(4)
    st.markdown('---')
    col114, col214, col314, col414 = st.columns(4)
    st.markdown('---')
    col115, col215, col315, col415 = st.columns(4)

    with col10:
        st.markdown('**Attribute**')
    with col20:
        st.markdown('**-1**')
    with col30:
        st.markdown('**Value**')
    with col40:
        st.markdown('**+1**')

    with col11:
        st.markdown('Vehicles Density (&#8593;)')
    with col12:
        st.markdown('Motorcycle Density (&#8593;)')
    with col13:
        st.markdown('Vehicles Density and Motorcycle Density lead to - **Vehicles** (&#8593;)')
    with col14:
        st.markdown('Main Roads Accessibility (&#8595;)')
    with col15:
        st.markdown('Local Roads Density (&#8593;)')
    with col16:
        st.markdown('Main Roads Accessibility and Local Roads Density lead to - **Roads** (&#8593;)')
    with col17:
        st.markdown('Primary School Attendance (&#8593;)')
    with col18:
        st.markdown('Secondary School Attendance (&#8593;)')
    with col19:
        st.markdown('Primary School Attendance and Secondary School Attendance lead to - **School** (&#8593;)')
    with col110:
        st.markdown('Assistance and Care Allowance Share (&#8595;)')
    with col111:
        st.markdown('Poverty Share (&#8595;)')
    with col112:
        st.markdown('Assistance and Care Allowance Share and Poverty Share lead to - **Social Factors** (&#8593;)')
    with col113:
        st.markdown('Doctors Accessibility (&#8593;)')
    with col114:
        st.markdown('Social Factors and Doctors Accessibility lead to - **Health and Social Factors** (&#8593;)')
    with col115:
        st.markdown('Schools and Health and Social Factors lead to - **Social Determinants** (&#8593;)')

    with col31:
        annotated_text((pred['Vehicles Density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Vehicles Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col32:
        annotated_text((pred['Motorcycle Density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Motorcycle Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col33:
        annotated_text((pred['Vehicles'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Vehicles'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col34:
        annotated_text((pred['Main Roads Accessibility'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'}), '', pred['Main Roads Accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
    with col35:
        annotated_text((pred['Local Roads Density'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Local Roads Density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col36:
        annotated_text((pred['Roads'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Roads'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col37:
        annotated_text((pred['Primary School Attendance'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Primary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col38:
        annotated_text((pred['Secondary School Attendance'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Secondary School Attendance'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col39:
        annotated_text((pred['School'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['School'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col310:
        annotated_text((pred['Assistance and Care Allowance Share'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'}), '', pred['Assistance and Care Allowance Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
    with col311:
        annotated_text((pred['Poverty Share'].map({'1': 'Good', '2': 'Medium', '3': 'Poor'}), '', pred['Poverty Share'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
    with col312:
        annotated_text((pred['Social Factors'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col313:
        annotated_text((pred['Doctors Accessibility'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Doctors Accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col314:
        annotated_text((pred['Health and Social Factors'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Health and Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
    with col315:
        annotated_text((pred['Social Determinants'].map({'1': 'Poor', '2': 'Medium', '3': 'Good'}), '', pred['Social Determinants'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))

    df_mun = to_predict.copy()
    ### Vehicles Density
    to_write = True
    if (df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 1]) | (df_mun['Vehicles Density'].values[0] == dm.d._levels[0][1][len(dm.d._levels[0][1]) - 2]):
        to_write = False

        df_mun_v = df_mun.copy()
        down_v = dm.d._levels[0][1].index(df_mun['Vehicles Density'].values[0]) - 1
        df_mun_v['Vehicles Density'] = dm.d._levels[0][1][down_v]
        prediction_n = dm.d.predict(df_mun_v).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
        if prediction != prediction_n:
            with col21:
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
            with col41:
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
            with col21:
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
            with col41:
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
            with col23:
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
            with col43:
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
            with col23:
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
            with col43:
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
            with col24:
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
            with col44:
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
            with col24:
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
            with col44:
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
            with col25:
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
            with col45:
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
            with col25:
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
            with col45:
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
            with col27:
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
            with col47:
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
            with col27:
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
            with col47:
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
            with col28:
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
            with col48:
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
            with col28:
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
            with col48:
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
            with col210:
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
            with col410:
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
            with col210:
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
            with col410:
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
            with col211:
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
            with col411:
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
            with col211:
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
            with col411:
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
            with col213:
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
            with col413:
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
            with col213:
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
            with col413:
                if prediction_n == 'Poor':
                    annotated_text((prediction_n, '', 'red'))
                elif prediction_n == 'Good':
                    annotated_text((prediction_n, '', 'lightgreen'))
                else:
                    annotated_text((prediction_n, '', 'gray'))