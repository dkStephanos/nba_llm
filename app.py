import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from langchain_core.language_models.llms import LLM
from typing import Any, Optional, List
import re
import plotly.express as px

session = get_active_session()

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
    # Match content between triple backticks (```), optionally with "python" after the first backtick
    pattern = r"```(?:python\s*)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Return the first matched code block
        return matches[0].strip()
    else:
        # Handle the case where no complete code block is found
        # Look for the last incomplete code block starting with ```
        start_pattern = r"```(?:python\s*)?(.*)"
        start_match = re.search(start_pattern, text, re.DOTALL)
        if start_match:
            return start_match.group(1).strip()
    return None

def get_response_from_llm(prompt):
    llm = SnowflakeCortexLLM(sp_session=session)
    return llm(prompt)

# Ensure df exists
database = "NBA"  
schema = "PUBLIC"  
table = "BOXSCORES"

# Streamlit app
st.set_page_config(page_title="Historic Boxscore Data Interface", page_icon="📊", layout="wide")
st.title("Historic Boxscore Data Interface 📊")

# User input
prompt = st.text_area('Enter your query about historic boxscore data:')

# Session state for the generated code and edit mode
st.session_state.generated_code = ''
st.session_state.edit_mode = False
st.session_state.show_chart = False

if st.button('Submit'):
    with st.spinner("Waiting for LLM to generate code..."):
        # Fetch the data and create a pandas dataframe
        df = session.table(f"{database}.{schema}.{table}").to_pandas()
        # Update the prompt to ask for Python code explicitly
        llm_prompt = f"""
        You are a Python developer that writes code using Plotly to visualize data.
        Your data input is a pandas dataframe named df with the following columns: {', '.join(df.columns)}.
        Do not attempt to read in or utilize any additional data or methods.
        Generate Python code only to visualize the following query:
        {prompt}
        Ensure the code includes all necessary imports, data aggregation, and sorting steps.
        Make sure the code can be executed as is to generate the requested plot.
        """
        response = get_response_from_llm(llm_prompt)
        code = extract_python_code(response)
        
        if not code:
            st.error("No valid Python code could be parsed from the LLM response.")
            st.text(response)
        else:
            st.session_state.generated_code = code  # Store the generated code in session state
            st.session_state.show_chart = True

            st.subheader('Generated Code:')
            if st.session_state.edit_mode:
                st.text_area('Edit the code if needed:', st.session_state.generated_code, height=300, key='code_editor')
            else:
                st.code(st.session_state.generated_code, language="python")

# Edit button to toggle edit mode
if st.session_state.show_chart:
    if st.session_state.edit_mode:
        if st.button('Save and Refresh Chart'):
            st.session_state.generated_code = st.session_state.code_editor
            st.session_state.edit_mode = False
    else:
        if st.button('Edit Code'):
            st.session_state.edit_mode = True

        st.subheader('Chart:')
        with st.spinner("Plotting ..."):
            # Re-execute the modified code
            try:
                exec_globals = {"st": st, "df": session.table(f"{database}.{schema}.{table}").to_pandas(), "px": px}
                exec(st.session_state.generated_code, exec_globals)
                fig = exec_globals.get('fig')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
