import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Carbon Emission Predictor',
    page_icon=':earth_americas:',	
)# This is an emoji shortcode. Could be a URL too.

st.title('Carbon Emission Predictor App')


# radio button
# first argument is the title of the radio button
# second argument is the options for the radio button
status = st.radio("Select Gender: ", ('Male', 'Female'))

# conditional statement to print 
# Male if male is selected else print female
# show the result using the success function
if (status == 'Male'):
	st.success("Male")
else:
	st.success("Female")

# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Body_Type = st.selectbox("Body Type: ",
					['Underweight', 'Normal', 'Overweight', 'Obese'])

# print the selected hobby
st.write("Body Type: ", Body_Type)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Diet = st.selectbox("Diet: ",
					['Omnivore', 'Vegetarian', 'Pescatarian', 'Vegan'])

# print the selected hobby
st.write("Diet: ", Diet)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
How_often_shower = st.selectbox("How Often Do You Shower?: ",
					['Daily', 'Less Frequently', 'More Frequently', 'Twice A Day'])

# print the selected hobby
st.write("How Often Do You Shower?: ", How_often_shower)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Heating = st.selectbox("Your Home Heating Source: ",
					['Coal', 'Natural Gas', 'Wood', 'Electricity'])

# print the selected hobby
st.write("Your Home Heating Source: ", Heating)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Trans = st.selectbox("What Is Your Main Means of Transportation?: ",
					['Public', 'Walk/Bicycle', 'Private'])

# print the selected hobby
st.write("What Is Your Main Means Of Transport?: ", Trans)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Social = st.selectbox("How Often Do You Participate In Social Activities?: ",
					['Often', 'Never', 'Sometimes'])

# print the selected hobby
st.write("How Often Do You Participate In Social Activities?: ", Social)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Waste_Size = st.selectbox("Waste Bag Size: ",
					['Small', 'Medium', 'Large', 'Extra Large'])

# print the selected hobby
st.write("Your Home Primiarily uses : ", Waste_Size)
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
Air = st.selectbox("Frequency of using aircraft in the last month.: ",
					['Frequently', 'Rarely', 'Never', 'Very Frequently'])

# print the selected hobby
st.write("Frequency of using aircraft in the last month.: ", Air )
# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
Car_Dist = st.slider("Montly Distance(km) By car ", 0, 9999)

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(Car_Dist))

# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
Waste = st.slider("Waste Bags Weekly", 1, 7)

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(Waste))
# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
How_longtv = st.slider("Number Of Daily Hours You Spend Infront Of The TV or Monitor", 1, 24)

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(How_longtv))
# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
Newclothes = st.slider("New Clothes Montly", 0,50 )

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(Newclothes))
# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
Internet = st.slider("Daily Hours Spent On Internet", 0, 24)

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(Internet))
# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
EnergyEff = st.selectbox("Are Your Purchases Energy Efficient? ",
					['Yes', 'No', 'Sometimes'])

# print the selected hobby
st.write("You Are Energy Aware", EnergyEff) 
# Selection box

# first argument takes the box title
# second argument takes the options to show
Recyc = st.multiselect("What Do You Recycle?: ",
						['Nothing', 'Paper', 'Metal', 'Glass', 'Plastic'])

# write the selected options
st.write("You selected", len(Recyc))
# multi select box

# first argument takes the box title
# second argument takes the options to show
Cook_With = st.multiselect("What Do You Use to Cook? ",
						['Microwave', 'Oven', 'Stove', 'Grill', 'AirFryer', '' ])

# write the selected options
st.write("You cook with", len(Cook_With)) 
# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
Bill = st.slider("Montly Grocery Bill", 50, 299)

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(Bill))




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



   
