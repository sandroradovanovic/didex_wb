import streamlit as st
from page_util import apply_style

st.set_page_config(layout='wide', 
                   page_title = 'Contact')
apply_style()

st.title('Contact')

st.markdown(
    '''
    The authors of the web page and DEX models are:
    - [PhD Boris Delibašić](mailto:boris.delibasic@fon.bg.ac.rs), Full Professor, University of Belgrade
    - [PhD Sandro Radovanović](mailto:sandro.radovanovic@fon.bg.ac.rs), Assistant Professor, University of Belgrade

    The project supervisors were:
    - [PhD Svetlana Vukanović](mailto:svukanovic@worldbank.org), Senior Transport Specialist, World Bank Group
    - [Igor Miščević](mailto:imiscevic@worldbank.org), Urban Development Expert, World Bank Group
    - [Lazar Šestović](mailto:lsestovic@worldbank.org), Senior Country Economist, World Bank Group
    '''
)