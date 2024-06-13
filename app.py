import time
import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from langchain_core.language_models.llms import LLM
from typing import Any, Optional, List
import re
import plotly.express as px

# Function to get or create a Snowpark session
def get_or_create_session():
    if 'snowpark_session' not in st.session_state:
        st.session_state.snowpark_session = get_active_session()
    return st.session_state.snowpark_session

class SnowflakeCortexLLM(LLM):
    sp_session: Session
    model: str = 'mixtral-8x7b'
    cortex_function: str = 'complete'

    @property
    def _llm_type(self) -> str:
        return "snowflake_cortex"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        sql_stmt = f'''
            SELECT snowflake.cortex.{self.cortex_function}(
                '{self.model}', '{prompt}'
            ) AS llm_response;
        '''
        l_rows = self.sp_session.sql(sql_stmt).collect()
        llm_response = l_rows[0]['LLM_RESPONSE']
        return llm_response

def extract_python_code(text):
    pattern = r"```(?:python\s*)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    else:
        start_pattern = r"```(?:python\s*)?(.*)"
        start_match = re.search(start_pattern, text, re.DOTALL)
        if start_match:
            return start_match.group(1).strip()
    return None

def get_response_from_llm(prompt):
    llm = SnowflakeCortexLLM(sp_session=get_or_create_session())
    return llm(prompt)

# Ensure df exists
database = "NBA"
schema = "PUBLIC"
table = "BOXSCORES"
df = None

# Streamlit app
st.set_page_config(page_title="Historic Boxscore Data Interface", page_icon="📊", layout="wide")
st.title("Historic Boxscore Data Interface 📊")

# User input
prompt = st.text_area('Enter your query about historic boxscore data:')

# Initialize session state variables if they do not exist
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ''
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False

if st.button('Submit'):
    st.session_state.generated_code = None
    st.session_state.show_chart = False
    st.session_state.edit_mode = False
    with st.spinner("Waiting for LLM to generate code..."):
        df = get_or_create_session().table(f"{database}.{schema}.{table}").to_pandas()
        response = get_response_from_llm(f"""
        You are a Python developer that writes code using Plotly to visualize data.
        Your data input is a global variable, pandas dataframe named df with the following columns: {', '.join(df.columns)}.
        Do not attempt to read in, create, or utilize any additional data or methods.
        Generate Python code only to visualize the following query:
        {prompt}
        Ensure the code includes all necessary imports, data aggregation, and sorting steps.
        Be careful to re-index if grouping and then presenting a chart with an axis that matches the groupby col.
        Make sure the code can be executed as is to generate the requested plot.
        """)
        code = extract_python_code(response)
        
        if not code:
            st.error("No valid Python code could be parsed from the LLM response.")
            st.text(response)
        else:
            st.session_state.generated_code = code
            st.session_state.show_chart = True

if st.session_state.show_chart:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader('Generated Code:')
        if st.session_state.edit_mode:
            st.text_area('Edit the code if needed:', st.session_state.generated_code, height=300, key='code_editor')
            if st.button('Save and Refresh Chart'):
                st.session_state.generated_code = st.session_state.code_editor
                st.session_state.edit_mode = False
        else:
            st.code(st.session_state.generated_code, language="python")
            if st.button('Edit Code'):
                st.session_state.edit_mode = True
                st.session_state.show_chart = False

    with col2:
        st.subheader('Chart:')
        with st.spinner("Plotting ..."):
            try:
                if df is None:
                    df = get_or_create_session().table(f"{database}.{schema}.{table}").to_pandas()
                exec_globals = {"st": st, "df": df, "px": px}
                exec(st.session_state.generated_code, exec_globals)
                fig = exec_globals.get('fig')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
if st.session_state.edit_mode:
    st.subheader('Generated Code:')
    with st.container():
        st.text_area('Edit the code if needed:', st.session_state.generated_code, height=300, key='code_editor')
        if st.button('Save and Refresh Chart'):
            st.session_state.generated_code = st.session_state.code_editor
            st.session_state.edit_mode = False
            st.session_state.show_chart = True
            
