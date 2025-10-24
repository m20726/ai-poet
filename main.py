# from dotenv import load_dotenv
# load_dotenv()
import streamlit as st
import time
# from langchain.chat_models import ChatOpenAI
# chat_model = ChatOpenAI()

st.title("인공지능 시인")

content = st.text_input("시의 주제를 제시해주세요")

if st.button("시 작성 요청하기"):
    with st.spinner("시 작성 중...", show_time=True):
        time.sleep(1)
        # result = chat_model.predict(content + "에 대한 시를 작성해줘")
        result = content + "에 대한 시"
        st.write(result)