import streamlit as st
from helper import api_call

def summarize_doc():
    st.title("Summarize Document")
    with st.form("Doument Summarization form"):
        model = st.selectbox("Model",
                             options=['ChatGPT','Gemini-Pro'])
        api_key = st.text_input("Api Key")
        doc_length = st.selectbox("Select Document Length",
                    options=['Short','Medium','Large (Recommended for document with 30 or more pages)'])
        
        file = st.file_uploader("Choose a File",type=['txt','pdf'])
        num_words = st.slider("Summary in number of words",min_value=50,max_value=500)
        
        if st.form_submit_button("Summarize"):
            data={
                    "model":model,
                    "api_key":api_key,
                    "doc_length":doc_length,
                    "num_words":num_words
                }
            files = {
                        "file":(file.name,file,'application/pdf')
                    }
            # response = requests.post('http://localhost:8000/doc_summary/summarize',data=data,files=files)
            api_url = 'http://3.110.92.222/doc/summarize'
            response = api_call(api_url,'post',data=data,files=files)
            
            if response.status_code == 200:
                response = eval(response.content.decode('utf-8'))['data']
                message  = eval(response.content.decode('utf-8'))['message']
            else:
                print(response)
                response = eval(response.content.decode('utf-8'))['message']
            st.write(response)