import streamlit as st
import pandas as pd
import math
from pathlib import Path

st.title('Carbon Emission Predictor App')

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Carbon Emission Predictor',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    #DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    #raw_gdp_df = pd.read_csv(DATA_FILENAME)

    csv_url = "https://raw.githubusercontent.com/shawnburris98/carbonfootprint/main/Carbon%20Emission.csv"
    df = pd.read_csv(csv_url)



   
