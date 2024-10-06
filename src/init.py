import streamlit as st
def initialize_session_state():
    st.session_state.setdefault('segmented_image', None)
    st.session_state.setdefault('new_colors', {})