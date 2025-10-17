import streamlit as st
from dotenv import load_dotenv
import os
from orchestrator import orchestrate
 
load_dotenv()
 
st.set_page_config(page_title='HealthPair Prototype', layout='centered')
 
st.title('HealthPair — Simple Multi-Agent Chat (Streamlit)')
st.markdown('Enter user context (age, conditions, medications, lifestyle) and press **Generate**.')
 
context = st.text_area('User context', height=200, placeholder='e.g. 45-year-old male, type 2 diabetes, BMI 29, sedentary, hypertensive...')
 
col1, col2 = st.columns(2)
with col1:
    if st.button('Generate Recommendation'):
        if not context.strip():
            st.error('Please provide user context.')
        else:
            with st.spinner('Calling agents...'):
                try:
                    result = orchestrate(context)
                except Exception as e:
                    st.exception(e)
                    result = None
            if result:
                st.subheader('Final Recommendation')
                st.code(result['final_recommendation'])
                st.subheader('Risk Report (raw)')
                st.text(result['risk_report'])
                st.subheader('Lifestyle Advice (raw)')
                st.text(result['lifestyle_advice'])
 
with col2:
    st.info('Notes')
    st.write('- This is a prototype using your GEMINI API key.')
    st.write('- Not a substitute for professional medical advice.')
    st.write('- Customize prompts in agents/ for tone and length.')