import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

def kod(data):
    st.write(data*2)

data = st.number_input("sayı gir:")
result = kod(data)
st.write(f"Sonuç: {result}")
