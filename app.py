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
    model: str = 'mistral-large'
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

# Streamlit app
st.set_page_config(page_title="Historic Boxscore Data Interface", page_icon="📊", layout="wide")
st.title("Historic Boxscore Data Interface 📊")

# Initialize session state variables
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ''
if 'show_code' not in st.session_state:
    st.session_state.show_code = False
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'updating_chart' not in st.session_state:
    st.session_state.updating_chart = False
if 'error_occurred' not in st.session_state:
    st.session_state.error_occurred = False

# User input
prompt = st.text_area('Enter your query about historic boxscore data:')

def generate_code():
    st.session_state.generated_code = None
    st.session_state.show_chart = False
    st.session_state.edit_mode = False
    st.session_state.updating_chart = True
    with st.spinner("Waiting for LLM to generate code..."):
        if st.session_state.df is None:
            st.session_state.df = get_or_create_session().table(f"{database}.{schema}.{table}").to_pandas()
        response = get_response_from_llm(f"""
        You are a Python developer that writes code using Plotly to visualize data.
        Your data input is a global variable, pandas dataframe named df that is available to you.
        The df has the following columnal structure: {', '.join([f"{col} ({dtype})" for col, dtype in st.session_state.df.dtypes.items()])}.
        Do not attempt to read in, create, or utilize any additional data or methods (or the provided df).
        Do not make any assumptions when writing the code. 
        Generate Python code only to visualize the following query:
        {prompt}
        Ensure the code includes all necessary imports, data aggregation, and sorting steps.
        Be careful to re-index if grouping and then presenting a chart with an axis that matches the groupby col.
        Make sure the code can be executed as is to generate the requested plot.

        The data we are interacting with is a series of boxscore entries for players. each row represents a single player for a single game.
        When asked for career stats, these need to be aggregatted across the entire data set (or specified year range) on a player or team basis (depending on context).
        Do not use GAME_ID in any calculations, when getting per-game averages, you must use the rows count.

        When crafting your python code, make sure you are using keys that actually exist in the dataframe. For example, if I ask for data around a specific team, i.e. Lakers, you cannot just use that to filter the col. 
        In this case, our team name takes the structure: Los Angeles Lakers, so you need to coerce any nl prompts to fit the underlying data in the df.

        Unless specified via per minute, or historical, or totals, assume that any requested stat should be on a per-game basis. Per game stats should be calculated as an aggregation accross the rows DO NOT use MP to calculate per game avgs.
        Be sure to inspect dtypes as well to avoid operations with bad operands like dividing a float by a str.

        Here is a list of common mistakes you SHOULD avoid making.

        1) # Calculate per game scoring average
            nuggets_df["PTS_AVG"] = nuggets_df["PTS"] / nuggets_df["MP"] * 48

            --> This makes no sense, minutes played has no impact on per game totals or averages
        """)
        code = extract_python_code(response)
        
        if not code:
            st.error("No valid Python code could be parsed from the LLM response.")
            st.text(response)
        else:
            st.session_state.generated_code = code
            st.session_state.show_chart = True

if st.button('Submit'):
    generate_code()

@st.cache_data
def get_chart(code, df):
    exec_globals = {"st": st, "df": df, "px": px}
    exec(code, exec_globals)
    return exec_globals.get('fig')

# Chart display logic
if st.session_state.show_chart:
    st.subheader('Chart:')
    chart_placeholder = st.empty()
    
    if st.session_state.updating_chart:
        with st.spinner("Updating chart..."):
            try:
                fig = get_chart(st.session_state.generated_code, st.session_state.df)
                if fig:
                    time.sleep(0.5)  # Add a small delay to ensure the spinner is visible
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                st.session_state.error_occurred = False
            except Exception as e:
                st.session_state.error_occurred = True
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
            finally:
                st.session_state.updating_chart = False
    else:
        try:
            fig = get_chart(st.session_state.generated_code, st.session_state.df)
            if fig:
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            st.session_state.error_occurred = False
        except Exception as e:
            st.session_state.error_occurred = True
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Code display and editing logic
if st.session_state.show_chart or st.session_state.error_occurred:
    if st.button('Toggle Code View'):
        st.session_state.show_code = not st.session_state.show_code
        st.session_state.edit_mode = False  # Reset edit mode when toggling

if st.session_state.show_code or st.session_state.error_occurred:
    st.subheader('Code:')
    
    if not st.session_state.edit_mode:
        st.code(st.session_state.generated_code, language="python")
        if st.button('Edit Code'):
            st.session_state.edit_mode = True
    else:
        edited_code = st.text_area('Edit the code:', st.session_state.generated_code, height=300, key='code_editor')
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('Save and Update Chart'):
                st.session_state.generated_code = edited_code
                st.session_state.edit_mode = False
                st.session_state.updating_chart = True
                st.experimental_rerun()
        with col2:
            if st.button('Cancel'):
                st.session_state.edit_mode = False
