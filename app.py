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
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def get_response_from_llm(prompt):
    llm = SnowflakeCortexLLM(sp_session=session)
    return llm(prompt)

# Ensure df exists
database = "NBA"  
schema = "PUBLIC"  
table = "BOXSCORES"  

# Fetch the data and create a pandas dataframe
df = session.table(f"{database}.{schema}.{table}").to_pandas()

# Streamlit app
st.set_page_config(page_title="Historic Boxscore Data Interface", page_icon="📊", layout="wide")
st.title("Historic Boxscore Data Interface 📊")

# User input
prompt = st.text_area('Enter your query about historic boxscore data:')
if st.button('Submit'):
    with st.spinner("Waiting for LLM to generate code..."):
        # Update the prompt to ask for Python code explicitly
        llm_prompt = f"""
        You are a Python developer that writes code using Plotly to visualize data.
        Your data input is a pandas dataframe named df with the following columns: {', '.join(df.columns)}.
        Generate Python code only to visualize the following query:
        {prompt}
        Make sure the code performs any necessary aggregation and sorting, and can be executed as is to generate the requested plot.
        """
        response = get_response_from_llm(llm_prompt)
    
        code = extract_python_code(response)
        
        if not code:
            st.error("No valid Python code could be parsed from the LLM response.")
            st.text(response)
        else:
            st.subheader('Generated Code:')
            st.code(code, language="python", line_numbers=True)
            
            st.subheader('Chart:')
            with st.spinner("Plotting ..."):
                # Execute the generated code
                try:
                    exec_globals = {"st": st, "df": df, "px": px}
                    exec(code, exec_globals)
                    fig = exec_globals.get('fig')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
