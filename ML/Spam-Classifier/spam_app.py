import streamlit as st
import pickle 

mod=pickle.load(open("model.pkl","rb"))
vec=pickle.load(open("vectorizer.pkl","rb"))

st.title("✉️ Spam Message Classifier 📤")

message=st.text_area("Enter a message:")

if st.button("predict"):
    if message.strip()=="":
        st.warning("please enter the message")
    else :
        message_vec=vec.transform([message])
        pred=mod.predict(message_vec)[0]
        if pred==1:
            st.error("🚨 Spam Message")
        else :
            st.success("✅ Not Spam(Ham)")
