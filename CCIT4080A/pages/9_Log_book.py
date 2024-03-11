import streamlit as st


st.set_page_config(page_title="Log Book", page_icon="For_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_ASS.jpeg")
col2.title("Log Book")
st.header("", divider="red")
with st.sidebar:
    st.image("For_ASS.jpeg")
    st.title("∀ ASS Team members")
    st.header("", divider="red")
    mem1, mem2, mem3, mem4 = st.columns([1,1,1,1])
    mem1.write("Angus Li")
    mem2.write("Alex Lau")
    mem3.write("Sunny Yau")
    mem4.write("Sunny Chen")
    st.header("", divider="red")


