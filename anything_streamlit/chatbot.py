# First
import openai 
import streamlit as st
from helper import api_call

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
def doc_qna():
    st.title("💬 Document Q&A") 
    model = st.selectbox("Choose Model: ",
                         options=['ChatGPT','Gemini-Pro'])
    api_key = st.text_input("API Key",type='password')
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        openai.api_key = api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        print(st.session_state.messages)
        # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        api_url = 'http://localhost:8000/doc/qna'
        data = {
            'model' : model,
            'api_key' : api_key,
            'chat_history': st.session_state.messages
        }
        response = api_call(api_url,'post',data=data)
        # msg = response.choices[0].message
        msg = response
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg.content)