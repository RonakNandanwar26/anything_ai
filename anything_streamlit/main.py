import streamlit as st
import requests
from summarize_doc import summarize_doc
from docQnA import doc_qna


def main():
    with st.sidebar:
        st.title("Anything.ai")

        page_options = ['Summarize Document','Document Q & A']
        selected_page = st.radio("",page_options)

    if selected_page == 'Summarize Document':
        summarize_doc()
    elif selected_page == 'Document Q & A':
        doc_qna()



if __name__ == '__main__':
    main()