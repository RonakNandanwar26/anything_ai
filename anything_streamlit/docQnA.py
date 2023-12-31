# First
import streamlit as st
from helper import api_call

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
def doc_qna():
    st.title("Document Question Answer")
    with st.form("Document Question Answer"): 
        model = st.selectbox("Choose Model: ",
                            options=['ChatGPT']) # Gemini-Pro'
        api_key = st.text_input("API Key",type='password')
        file = st.file_uploader("Choose a File:",type=['txt','pdf'])
        question = st.text_area("Question:") 
        
        # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        if st.form_submit_button("submit"):
            if api_key == "":
                st.error("Please enter API Key...")
                st.stop()
            if file is None:
                st.error("Please select file...")
                st.stop()
            with st.spinner("Please wait..."):
                api_url = 'http://3.110.92.222/doc/qna'
                data = {
                    'model' : model,
                    'api_key' : api_key,
                    'question': question
                }
                files = {
                    "file":(file.name,file,'application/pdf')
                }
                response = api_call(api_url,'post',data=data,files=files)
                if response.status_code == 200:
                    st.write(eval(response.content.decode('utf-8'))['data'])
                else:
                    st.error(eval(response.content.decode('utf-8'))['message'])