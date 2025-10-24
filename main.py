# from dotenv import load_dotenv
# load_dotenv()
import streamlit as st
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate

model_q2 = "llama-2-7b-chat.ggmlv3.q2_K.bin"
model_q8 = "llama-2-7b-chat.ggmlv3.q8_0.bin"

# 모델이 한 번만 로드되고, 다시 실행할 때 빠르게 불러옵니다.
@st.cache_resource
def load_model():
    return CTransformers(model=model_q8, model_type="llama")

try:
    llm = load_model()
except Exception as e:
    st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

template = """
You are a poet with rich emotions.  
Write a lyrical poem in English that fits the theme '{question}'.  
If possible, make it a short poem with at least four lines.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

st.title("인공지능 시인")

content = st.text_input("시의 주제를 제시해주세요")

if st.button("시 작성 요청하기"):
    final_prompt = prompt.format(question=f"Write a poem about {content}")
    with st.spinner("시 작성 중...", show_time=True):
        result = llm.invoke(final_prompt)
        st.write(result)