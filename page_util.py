import streamlit as st

def hide_table_index():
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

def apply_style():
    with open('util/style-overrides.css') as f:
        css_styles = f.read()
    
    # Inject CSS into the Streamlit app
    st.markdown(f'<style>{css_styles}</style>', unsafe_allow_html=True)