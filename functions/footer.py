# Usefull libriaries
from PIL import Image
import streamlit as st
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px

# Other functions for the app footer
def image_render(src_as_string, **style):
    return img(src = src_as_string, style = styles(**style))

def link_render(link, text, **style):
    return a(_href = link, _target = "_blank", style = styles(**style))(text)

def footer(*args):
    style = """<style> # MainMenu {visibility: hidden;} footer {visibility: hidden;} .stApp { bottom: 85px; } </style>"""
    style_div = styles(position = "fixed", 
                       left = 0, 
                       bottom = 0, 
                       margin = px(0, 0, 0, 0), 
                       width = percent(100), 
                       color = "black", 
                       text_align = "center",
                       height = px(30),
                       opacity = 1,
                       display="flex",
                       justify_content="center"
                       )
    style_hr = styles(display = "block",
                      margin = px(8, 8, "auto", "auto"),
                      border_style = "inset",
                      border_width = px(2)
                      )
    body = p()
    foot = div( style = style_div )( hr( tyle = style_hr ), body )
    st.markdown(style, unsafe_allow_html = True)
    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)
    
    st.markdown(str(foot), unsafe_allow_html = True)
