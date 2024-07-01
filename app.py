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

def get_response_from_llm(prompt, conversation_history):
    llm = SnowflakeCortexLLM(sp_session=get_or_create_session())
    
    starter_prompt = """
    You are a Python developer that writes code using Plotly to visualize data.
    Your data input is a global variable, pandas dataframe named df that is available to you.
    The df has the following columnal structure: {columns}.
    Do not attempt to read in, create, or utilize any additional data or methods (or the provided df).
    Do not make any assumptions when writing the code.
    Generate Python code only to visualize the following query.
    Ensure the code includes all necessary imports, data aggregation, and sorting steps.
    Be careful to re-index if grouping and then presenting a chart with an axis that matches the groupby col.
    Make sure the code can be executed as is to generate the requested plot.
    The data we are interacting with is a series of boxscore entries for players. Each row represents a single player for a single game.
    When asked for career stats, these need to be aggregated across the entire data set (or specified year range) on a player or team basis (depending on context).
    Do not use GAME_ID in any calculations, when getting per-game averages, you must use the rows count.
    When crafting your python code, make sure you are using keys that actually exist in the dataframe. For example, if I ask for data around a specific team, i.e. Lakers, you cannot just use that to filter the col.
    In this case, our team name takes the structure: Los Angeles Lakers, so you need to coerce any nl prompts to fit the underlying data in the df.
    Unless specified via per minute, or historical, or totals, assume that any requested stat should be on a per-game basis. Per game stats should be calculated as an aggregation across the rows DO NOT use MP to calculate per game avgs.
    Be sure to inspect dtypes as well to avoid operations with bad operands like dividing a float by a str.
    Here is a list of common mistakes you SHOULD avoid making:
    1) # Calculate per game scoring average
       nuggets_df["PTS_AVG"] = nuggets_df["PTS"] / nuggets_df["MP"] * 48
       --> This makes no sense, minutes played has no impact on per game totals or averages
    """
    
    columns = ', '.join([f"{col} ({dtype})" for col, dtype in st.session_state.df.dtypes.items()])
    starter_prompt = starter_prompt.format(columns=columns)
    
    context = "\n".join([f"Human: {turn['human']}\nAI: {turn['ai']}" for turn in conversation_history[-3:]])  # Include last 3 turns
    full_prompt = f"{starter_prompt}\n\nConversation history:\n{context}\n\nHuman: {prompt}\nAI:"
    
    return llm(full_prompt)

@st.cache_data
def get_chart(code, df):
    exec_globals = {"st": st, "df": df, "px": px}
    exec(code, exec_globals)
    return exec_globals.get('fig')

# Streamlit app
st.set_page_config(page_title="Historic Boxscore Data Chat Interface", page_icon="📊", layout="wide")
st.title("Historic Boxscore Data Chat Interface 📊")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Load data if not already loaded
if st.session_state.df is None:
    database = "NBA"
    schema = "PUBLIC"
    table = "BOXSCORES"
    st.session_state.df = get_or_create_session().table(f"{database}.{schema}.{table}").to_pandas()

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chart" in message:
            st.plotly_chart(message["chart"], use_container_width=True)
        if "code" in message:
            show_code_key = f"show_code_{i}"
            edit_mode_key = f"edit_mode_{i}"
            
            if show_code_key not in st.session_state:
                st.session_state[show_code_key] = False
            if edit_mode_key not in st.session_state:
                st.session_state[edit_mode_key] = False

            if st.button("Show Code", key=f"show_code_button_{i}"):
                st.session_state[show_code_key] = not st.session_state[show_code_key]

            if st.session_state[show_code_key]:
                st.code(message["code"], language="python")
                if st.button("Edit Code", key=f"edit_code_button_{i}"):
                    st.session_state[edit_mode_key] = True

            if st.session_state[edit_mode_key]:
                edited_code = st.text_area("Edit the code:", message["code"], height=300, key=f"code_editor_{i}")
                if st.button("Update Chart", key=f"update_chart_{i}"):
                    try:
                        new_fig = get_chart(edited_code, st.session_state.df)
                        if new_fig:
                            st.plotly_chart(new_fig, use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "content": "Here's the updated chart based on your edited code:", "code": edited_code, "chart": new_fig})
                        else:
                            st.warning("The edited code executed successfully, but no chart was generated.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    st.session_state[edit_mode_key] = False
                    st.experimental_rerun()


# Chat input
if prompt := st.chat_input("Ask about historic boxscore data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"human": prompt, "ai": ""})  # Add empty AI response for now

    response = get_response_from_llm(prompt, st.session_state.conversation_history)
    
    code = extract_python_code(response)
    if not code:
        st.session_state.messages.append({"role": "assistant", "content": "I couldn't generate valid Python code. Here's the raw response:", "error": response})
    else:
        try:
            fig = get_chart(code, st.session_state.df)
            if fig:
                st.session_state.messages.append({"role": "assistant", "content": "Here's the chart based on your query:", "code": code, "chart": fig})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "The code executed successfully, but no chart was generated.", "code": code})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}", "code": code, "error": str(e)})

    # Update the AI response in the conversation history
    st.session_state.conversation_history[-1]["ai"] = response
    
    st.experimental_rerun()

# Optionally, add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.experimental_rerun()

st.markdown("""
<script>
    var body = window.parent.document.querySelector(".main");
    body.scrollTop = body.scrollHeight;
</script>
""", unsafe_allow_html=True)