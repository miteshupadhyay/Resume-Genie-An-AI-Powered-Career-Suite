# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:18:29 2026

@author: User
"""

import streamlit as st
from PIL import Image

logo = Image.open("ResumeGenieLogo.png")
st.sidebar.image(logo,width=80)

st.sidebar.markdown("**Resume Genie**")

st.title("Hello World!!")
st.write("Welcome to my first Streamlit App")

