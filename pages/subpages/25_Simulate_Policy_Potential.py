import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np

import dex_model_gva as dm

st.set_page_config(layout='wide', page_title = 'Simulate Gross Value Added per capita Policy Potential')

st.title('Simulate Gross Value Added per capita Policy Potential')

st.markdown(
    '''
    This page aims to helping you see the effects of the policy on the municipality.
    '''
)

df_gva = pd.read_csv('data/gva.csv')
df_gva = df_gva.loc[df_gva['GVA Per Capita Normalized'].notna(), :]
LSGs = np.unique(df_gva['Area Name'])

selected_lsg = st.selectbox(label='Please select municipality', options=LSGs)
df_year = df_gva.loc[(df_gva['Time Period'] == 2021), :]
df_mun = df_gva.loc[(df_gva['Area Name'] == selected_lsg) & (df_gva['Time Period'] == 2021), :]

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
col111, col211, col311 = st.columns(3)
col112, col212, col312 = st.columns(3)

with col11:
    st.markdown('Insert needed values to estimate Gross Value Added per capita Policy Potential')
with col12:
    a_mra = st.number_input(label='Main road accessibility', value = 0 if np.isnan(df_mun['Main road accessibility'].values[0]) else df_mun['Main road accessibility'].values[0])
with col13:
    a_lrd = st.number_input(label='Local roads density', value = 0 if np.isnan(df_mun['Local roads density'].values[0]) else df_mun['Local roads density'].values[0])
with col14:
    a_md = st.number_input(label='Motorcycles density', value = 0 if np.isnan(df_mun['Motorcycles density'].values[0]) else df_mun['Motorcycles density'].values[0])
with col15:
    a_vd = st.number_input(label='Vehicles density', value = 0 if np.isnan(df_mun['Vehicles density'].values[0]) else df_mun['Vehicles density'].values[0])
with col16:
    a_acr = st.number_input(label='Active companies rate', value = 0 if np.isnan(df_mun['Active companies rate'].values[0]) else df_mun['Active companies rate'].values[0])
with col17:
    a_mer = st.number_input(label='Municipality employment rate', value = 0 if np.isnan(df_mun['Municipality employment rate'].values[0]) else df_mun['Municipality employment rate'].values[0])
with col18:
    a_er = st.number_input(label='Unemployed rate', value = 0 if np.isnan(df_mun['Unemployed rate'].values[0]) else df_mun['Unemployed rate'].values[0])
with col19:
    a_tsir = st.number_input(label='Transport and storage investments rate', value = 0 if np.isnan(df_mun['Transport and storage investments rate'].values[0]) else df_mun['Transport and storage investments rate'].values[0])
with col110:
    a_da = st.number_input(label='Doctors accessibility', value = 0 if np.isnan(df_mun['Doctors accessibility'].values[0]) else df_mun['Doctors accessibility'].values[0])
with col111:
    a_pcer = st.number_input(label='Preschool children enrollment rate', value = 0 if np.isnan(df_mun['Preschool children enrollment rate'].values[0]) else df_mun['Preschool children enrollment rate'].values[0])
with col112:
    a_ta = st.number_input(label='Tourists Arrivals', value = 0 if np.isnan(df_mun['Tourists Arrivals'].values[0]) else df_mun['Tourists Arrivals'].values[0])


with col21:
    st.markdown(
        '''
        This area shows the value of the attribute for the selected municipality in 2021. 
        
        (Values within the brackets represent the average value of municipalities for the selected attribute)
        '''
    )

with col22:
    st.markdown('Main road accessibility:')
    if df_mun['Main road accessibility'].values[0] > 13.025:
        annotated_text((f"{round(df_mun['Main road accessibility'].values[0], 3)}", '', 'red'))
    elif df_mun['Main road accessibility'].values[0] <= 5.461:
        annotated_text((f"{round(df_mun['Main road accessibility'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Main road accessibility'].values[0], 3)) else round(df_mun['Main road accessibility'].values[0], 3)}")
    annotated_text((f'({round(df_year["Main road accessibility"].mean(), 3)})', 'Serbia', ''))

with col23:
    st.markdown('Local roads density:')
    if df_mun['Local roads density'].values[0] > 0.382:
        annotated_text((f"{round(df_mun['Local roads density'].values[0], 3)}", '', 'red'))
    elif df_mun['Local roads density'].values[0] <= 0.213:
        annotated_text((f"{round(df_mun['Local roads density'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Local roads density'].values[0], 3)) else round(df_mun['Local roads density'].values[0], 3)}")
    annotated_text((f'({round(df_year["Local roads density"].mean(), 3)})', 'Serbia', ''))

with col24:
    st.markdown('Motorcycles density:')
    if df_mun['Motorcycles density'].values[0] <= 0.281:
        annotated_text((f"{round(df_mun['Motorcycles density'].values[0], 3)}", '', 'red'))
    elif df_mun['Motorcycles density'].values[0] > 0.762:
        annotated_text((f"{round(df_mun['Motorcycles density'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Motorcycles density'].values[0], 3)) else round(df_mun['Motorcycles density'].values[0], 3)}")
    annotated_text((f'({round(df_year["Motorcycles density"].mean(), 3)})', 'Serbia', ''))
    
with col25:
    st.markdown('Vehicles density:')
    if df_mun['Vehicles density'].values[0] > 21.286:
        annotated_text((f"{round(df_mun['Vehicles density'].values[0], 3)}", '', 'red'))
    elif df_mun['Vehicles density'].values[0] <= 10.272:
        annotated_text((f"{round(df_mun['Vehicles density'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Vehicles density'].values[0], 3)) else round(df_mun['Vehicles density'].values[0], 3)}")
    annotated_text((f'({round(df_year["Vehicles density"].mean(), 3)})', 'Serbia', ''))

with col26:
    st.markdown('Active companies rate:')
    if df_mun['Active companies rate'].values[0] <= 7.755:
        annotated_text((f"{round(df_mun['Active companies rate'].values[0], 3)}", '', 'red'))
    elif df_mun['Active companies rate'].values[0] > 10.985:
        annotated_text((f"{round(df_mun['Active companies rate'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Active companies rate'].values[0], 3)) else round(df_mun['Active companies rate'].values[0], 3)}")
    annotated_text((f'({round(df_year["Active companies rate"].mean(), 3)})', 'Serbia', ''))
    
with col27:
    st.markdown('Municipality employment rate:')
    if df_mun['Municipality employment rate'].values[0] <= 0.291:
        annotated_text((f"{round(df_mun['Municipality employment rate'].values[0], 3)}", '', 'red'))
    elif df_mun['Municipality employment rate'].values[0] > 0.378:
        annotated_text((f"{round(df_mun['Municipality employment rate'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Municipality employment rate'].values[0], 3)) else round(df_mun['Municipality employment rate'].values[0], 3)}")
    annotated_text((f'({round(df_year["Municipality employment rate"].mean(), 3)})', 'Serbia', ''))
    
with col28:
    st.markdown('Unemployed rate:')
    if df_mun['Unemployed rate'].values[0] > 122.0:
        annotated_text((f"{round(df_mun['Unemployed rate'].values[0], 3)}", '', 'red'))
    elif df_mun['Unemployed rate'].values[0] <= 77.0:
        annotated_text((f"{round(df_mun['Unemployed rate'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Unemployed rate'].values[0], 3)) else round(df_mun['Unemployed rate'].values[0], 3)}")
    annotated_text((f'({round(df_year["Unemployed rate"].mean(), 3)})', 'Serbia', ''))
    
with col29:
    st.markdown('Transport and storage investments rate:')
    if df_mun['Transport and storage investments rate'].values[0] < 0.562:
        annotated_text((f"{round(df_mun['Transport and storage investments rate'].values[0], 3)}", '', 'red'))
    elif df_mun['Transport and storage investments rate'].values[0] > 0.562:
        annotated_text((f"{round(df_mun['Transport and storage investments rate'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Transport and storage investments rate'].values[0], 3)) else round(df_mun['Transport and storage investments rate'].values[0], 3)}")
    annotated_text((f'({round(df_year["Transport and storage investments rate"].mean(), 3)})', 'Serbia', ''))
    
with col210:
    st.markdown('Doctors accessibility:')
    if df_mun['Doctors accessibility'].values[0] <= 1.3:
        annotated_text((f"{round(df_mun['Doctors accessibility'].values[0], 3)}", '', 'red'))
    elif df_mun['Doctors accessibility'].values[0] > 2.1:
        annotated_text((f"{round(df_mun['Doctors accessibility'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Doctors accessibility'].values[0], 3)) else round(df_mun['Doctors accessibility'].values[0], 3)}")
    annotated_text((f'({round(df_year["Doctors accessibility"].mean(), 3)})', 'Serbia', ''))

with col211:
    st.markdown('Preschool children enrollment rate:')
    if df_mun['Preschool children enrollment rate'].values[0] <= 2.086:
        annotated_text((f"{round(df_mun['Preschool children enrollment rate'].values[0], 3)}", '', 'red'))
    elif df_mun['Preschool children enrollment rate'].values[0] > 2.682:
        annotated_text((f"{round(df_mun['Preschool children enrollment rate'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Preschool children enrollment rate'].values[0], 3)) else round(df_mun['Preschool children enrollment rate'].values[0], 3)}")
    annotated_text((f'({round(df_year["Preschool children enrollment rate"].mean(), 3)})', 'Serbia', ''))

with col212:
    st.markdown('Tourists Arrivals:')
    if df_mun['Tourists Arrivals'].values[0] <= 978.0:
        annotated_text((f"{round(df_mun['Tourists Arrivals'].values[0], 3)}", '', 'red'))
    elif df_mun['Tourists Arrivals'].values[0] > 7629.333:
        annotated_text((f"{round(df_mun['Tourists Arrivals'].values[0], 3)}", '', 'lightgreen'))
    else:
        st.markdown(f"{0 if np.isnan(round(df_mun['Tourists Arrivals'].values[0], 3)) else round(df_mun['Tourists Arrivals'].values[0], 3)}")
    annotated_text((f'({round(df_year["Tourists Arrivals"].mean(), 3)})', 'Serbia', ''))


with col31:
    st.markdown(
        '''
        Below are the descriptions of the attributes.
        '''
    )

with col32:
    st.markdown(
        '''
        **Main road accessibility** - The total length (in kilometers) of all road segments (I level without highways, II, and local road segments) per square meter of the municipality.
        '''
    )

with col33:
    st.markdown(
        '''
        **Local roads density** - The total length (in kilometers) of local road segments without highways divided by 1,000 inhabitants of an LSG.
        '''
    )

with col34:
    st.markdown(
        '''
        **Motorcycles density** - The number of registered mopeds, motorcycles, and other two-wheel or three-wheel vehicles per square meter of the municipality.
        '''
    )

with col35:
    st.markdown(
        '''
        **Vehicles density** - The number of registered passenger vehicles per square meter of the municipality.
        '''
    )

with col36:
    st.markdown(
        '''
        **Active companies rate** - The total number of active companies per 1,000 inhabitants.
        '''
    )

with col37:
    st.markdown(
        '''
        **Municipality employment rate** - Percentage of people that are registered to work in an LSG. This number is obtained by dividing the number of employed people within an LSG by the estimated working-age population in an LSG.
        '''
    )

with col38:
    st.markdown(
        '''
        **Unemployment rate** - The total number of registered unemployed people per 1,000 inhabitants in an LSG.
        '''
    )

with col39:
    st.markdown(
        '''
        **Transport and storage investments rate** - The total amount (in RSD thousands) an LSG invested in new fixed assets in transport and storage per capita.
        '''
    )

with col310:
    st.markdown(
        '''
        **Doctors accessibility** - The total number of doctors within a municipality divided by the 1,000 inhabitants.
        '''
    )

with col311:
    st.markdown(
        '''
        **Preschool children enrollment rate** - Percentage of children attending the obligatory preschool education within an LSG. To calculate the number of children, we used an estimate of the number of children having 6 years.
        '''
    )

with col312:
    st.markdown(
        '''
        **Tourist Arrivals** - The total number of tourist arrivals within one year in an LSG.
        '''
    )

v_mra = '2'
if a_mra <= 5.461:
    v_mra = '1'
elif a_mra > 13.025:
    v_mra = '3'

v_lrd = '2'
if a_lrd <= 0.213:
    v_lrd = '1'
elif a_lrd > 0.382:
    v_lrd = '3'

v_md = '2'
if a_md <= 0.281:
    v_md = '1'
elif a_md > 0.762:
    v_md = '3'

v_vd = '2'
if a_vd <= 10.272:
    v_vd = '1'
elif a_vd > 21.286:
    v_vd = '3'

v_acr = '2'
if a_acr <= 7.755:
    v_acr = '1'
elif a_acr > 10.985:
    v_acr = '3'

v_mer = '2'
if a_mer <= 0.291:
    v_mer = '1'
elif a_mer > 0.378:
    v_mer = '3'

v_er = '2'
if a_er <= 77.0:
    v_er = '1'
elif a_er > 122.0:
    v_er = '3'

v_tsir = '2'
if a_tsir < 0.562:
    v_tsir = '1'
elif a_tsir > 0.562:
    v_tsir = '3'

v_da = '2'
if a_da <= 1.3:
    v_da = '1'
elif a_da > 2.1:
    v_da = '3'

v_pcer = '2'
if a_pcer <= 2.086:
    v_pcer = '1'
elif a_pcer > 2.682:
    v_pcer = '3'

v_ta = '2'
if a_ta <= 978.0:
    v_ta = '1'
elif a_ta > 7629.333:
    v_ta = '3'

st.markdown('---')
st.subheader('Prediction Outcome')

st.markdown('The prediction of the Gross Value Added per capita (constant prices) based on the inserted values is:')

to_predict = [v_mra, v_lrd, v_md, v_vd, v_acr, v_mer, v_er, v_tsir, v_da, v_pcer, v_ta]
to_predict = pd.DataFrame([to_predict], columns=['Main road accessibility', 'Local roads density', 'Motorcycles density', 'Vehicles density', 'Active companies rate', 'Municipality employment rate', 'Unemployed rate', 'Transport and storage investments rate', 'Doctors accessibility', 'Preschool children enrollment rate', 'Tourists Arrivals'])

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
col116, col216 = st.columns(2)
col117, col217 = st.columns(2)

with col11:
    st.markdown('Main road accessibility')
with col12:
    st.markdown('Local roads density')
with col13:
    st.markdown('Main road accessibility and Local roads density lead to - **Local Traffic**')
with col14:
    st.markdown('Motorcycles density')
with col15:
    st.markdown('Vehicles density')
with col16:
    st.markdown('Motorcycles density and Vehicles density lead to - **General Traffic**')
with col17:
    st.markdown('Local Traffic and General Traffic lead to - **Road and Traffic**')
with col18:
    st.markdown('Tourists Arrivals')
with col19:
    st.markdown('Road and Traffic and Tourists Arrivals lead to - **Tourism and Traffic**')
with col110:
    st.markdown('Preschool children enrollment rate')
with col111:
    st.markdown('Doctors accessibility')
with col112:
    st.markdown('Preschool children enrollment rate and Doctors accessibility lead to - **Social Factors**')
with col113:
    st.markdown('Municipality employment rate')
with col114:
    st.markdown('Unemployed rate')
with col115:
    st.markdown('Municipality employment rate and Unemployed rate lead to - **Employment State**')
with col116:
    st.markdown('Employment State and Active companies rate lead to - **Economy State**')
with col117:
    st.markdown('Economy State and Transport and storage investments rate lead to - **Economy and Investments**')

with col21:
    annotated_text((pred['Main road accessibility'], '', pred['Main road accessibility'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col22:
    annotated_text((pred['Local roads density'], '', pred['Local roads density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col23:
    annotated_text((pred['Local Traffic'], '', pred['Local Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col24:
    annotated_text((pred['Motorcycles density'], '', pred['Motorcycles density'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
with col25:
    annotated_text((pred['Vehicles density'], '', pred['Vehicles density'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col26:
    annotated_text((pred['General Traffic'], '', pred['General Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col27:
    annotated_text((pred['Road and Traffic'], '', pred['Road and Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col28:
    annotated_text((pred['Tourists Arrivals'], '', pred['Tourists Arrivals'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col29:
    annotated_text((pred['Tourism and Traffic'], '', pred['Tourism and Traffic'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col210:
    annotated_text((pred['Preschool children enrollment rate'], '', pred['Preschool children enrollment rate'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
with col211:
    annotated_text((pred['Doctors accessibility'], '', pred['Doctors accessibility'].map({'1': 'lightgreen', '2': 'gray', '3': 'red'})[0]))
with col212:
    annotated_text((pred['Social Factors'], '', pred['Social Factors'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col213:
    annotated_text((pred['Municipality employment rate'], '', pred['Municipality employment rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col214:
    annotated_text((pred['Unemployed rate'], '', pred['Unemployed rate'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col215:
    annotated_text((pred['Employment State'], '', pred['Employment State'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col216:
    annotated_text((pred['Economy State'], '', pred['Economy State'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))
with col217:
    annotated_text((pred['Economy and Investments'], '', pred['Economy and Investments'].map({'1': 'red', '2': 'gray', '3': 'lightgreen'})[0]))


st.markdown('**The predicted outcome is:**')
df_mun = to_predict.copy()
prediction = dm.d.predict(to_predict).values[0].replace('1', 'Poor').replace('2', 'Medium').replace('3', 'Good')
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