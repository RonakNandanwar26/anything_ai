import streamlit as st
import requests
from summarize_doc import summarize_doc


def main():
    with st.sidebar:
        st.title("Anything.ai")

        page_options = ['Summarize Document']
        selected_page = st.radio("",page_options)

    if selected_page == 'Summarize Document':
        summarize_doc()



if __name__ == '__main__':
    main()