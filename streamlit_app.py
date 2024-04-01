import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import math
from pathlib import Path
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot

LOGGER = get_logger(__name__)


def run():
      st.set_page_config(
      page_title='Carbon Emission Predictor',
      page_icon=':earth_americas:',	
      )# This is an emoji shortcode. Could be a URL too.
      st.write('# Carbon Emission Predictor App')
      st.subheader('Raw data')
      csv_url = './Carbon_Emission.csv'
      df = pd.read_csv(csv_url)
      st.write(df)


      status = st.radio("Select Gender: ", ('Male', 'Female'))

      # conditional statement to print 
      # Male if male is selected else print female
      # show the result using the success function
      if (status == 'Male'):
            st.success("Male")
      else:
            st.success("Female")

      st.write("--------------------------------------------------------------------------")

      # Selection box

      # first argument takes the titleof the selectionbox
      # second argument takes options
      Body_Type = st.selectbox("Body Type: ",
                                    ['Underweight', 'Normal', 'Overweight', 'Obese'])

      # print the selected hobby
      st.write("Body Type: ", Body_Type)
      # Selection box

      st.write("--------------------------------------------------------------------------")

      # first argument takes the titleof the selectionbox
      # second argument takes options
      Diet = st.selectbox("Diet: ",
                                    ['Omnivore', 'Vegetarian', 'Pescatarian', 'Vegan'])

      # print the selected hobby
      st.write("Diet: ", Diet)
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      How_often_shower = st.selectbox("How Often Do You Shower?: ",
                                    ['Daily', 'Less Frequently', 'More Frequently', 'Twice A Day'])

      # print the selected hobby
      st.write("How Often Do You Shower?: ", How_often_shower)
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      Heating = st.selectbox("Your Home Heating Source: ",
                                    ['Coal', 'Natural Gas', 'Wood', 'Electricity'])

      # print the selected hobby
      st.write("Your Home Heating Source: ", Heating)
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      Trans = st.selectbox("What Is Your Main Means of Transportation?: ",
                                    ['Public', 'Walk/Bicycle', 'Private'])

      # print the selected hobby
      st.write("What Is Your Main Means Of Transport?: ", Trans)
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      Social = st.selectbox("How Often Do You Participate In Social Activities?: ",
                                    ['Often', 'Never', 'Sometimes'])

      # print the selected hobby
      st.write("How Often Do You Participate In Social Activities?: ", Social)
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      Waste_Size = st.selectbox("Waste Bag Size: ",
                                    ['Small', 'Medium', 'Large', 'Extra Large'])

      # print the selected hobby
      st.write("Your Home Primiarily uses : ", Waste_Size)
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      Air = st.selectbox("Frequency of using aircraft in the last month.: ",
                                    ['Frequently', 'Rarely', 'Never', 'Very Frequently'])

      # print the selected hobby
      st.write("Frequency of using aircraft in the last month.: ", Air )
      # slider
      st.write("--------------------------------------------------------------------------")
      # first argument takes the title of the slider
      # second argument takes the starting of the slider
      # last argument takes the end number
      Car_Dist = st.slider("Montly Distance(km) By car ", 0, 9999)

      # print the level
      # format() is used to print value 
      # of a variable at a specific position
      st.text('Selected: {}KM'.format(Car_Dist))
      st.write("--------------------------------------------------------------------------")
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
      st.write("--------------------------------------------------------------------------")
      # first argument takes the title of the slider
      # second argument takes the starting of the slider
      # last argument takes the end number
      How_longtv = st.slider("Number Of Daily Hours You Spend Infront Of The TV or Monitor", 1, 24)

      # print the level
      # format() is used to print value 
      # of a variable at a specific position
      st.text('Selected: {}Hours'.format(How_longtv))
      # slider
      st.write("--------------------------------------------------------------------------")
      # first argument takes the title of the slider
      # second argument takes the starting of the slider
      # last argument takes the end number
      Newclothes = st.slider("New Clothes Montly", 0,50 )

      # print the level
      # format() is used to print value 
      # of a variable at a specific position
      st.text('Selected: {}'.format(Newclothes))
      # slider
      st.write("--------------------------------------------------------------------------")
      # first argument takes the title of the slider
      # second argument takes the starting of the slider
      # last argument takes the end number
      Internet = st.slider("Daily Hours Spent On Internet", 0, 24)

      # print the level
      # format() is used to print value 
      # of a variable at a specific position
      st.text('Selected: {} Hours'.format(Internet))
      # Selection box
      st.write("--------------------------------------------------------------------------")
      # first argument takes the titleof the selectionbox
      # second argument takes options
      EnergyEff = st.selectbox("Are Your Purchases Energy Efficient? ",
                                    ['Yes', 'No', 'Sometimes'])

      # print the selected hobby
      st.write("You Are Energy Aware", EnergyEff) 
      # Selection box
      st.write("--------------------------------------------------------------------------")

      # slider

      # first argument takes the title of the slider
      # second argument takes the starting of the slider
      # last argument takes the end number
      Bill = st.slider("Montly Grocery Bill", 50, 299)

      # print the level
      # format() is used to print value 
      # of a variable at a specific position
      st.text('Selected: {}$'.format(Bill), )




      # -----------------------------------------------------------------------------
      # Declare some useful functions.

      #@st.cache_data
      #def get_data():




      #deleting the column that has missing values
      df. __delitem__('Vehicle Type')
      df. __delitem__('Recycling')
      df. __delitem__('Cooking_With')

      # Encode(changing categorical values to numerical values)
      #df['recycling_encode'] = LabelEncoder().fit_transform(df['Recycling'])
      df['travelingByAir_encode'] = LabelEncoder().fit_transform(df['Frequency of Traveling by Air'])
      df['howOftenShower_encode'] = LabelEncoder().fit_transform(df['How Often Shower'])
      df['heating_encode'] = LabelEncoder().fit_transform(df['Heating Energy Source'])
      df['bodytype_encode'] = LabelEncoder().fit_transform(df['Body Type'])
      df['sex_encode'] = LabelEncoder().fit_transform(df['Sex'])
      df['diet_encode'] = LabelEncoder().fit_transform(df['Diet'])
      df['transport_encode'] = LabelEncoder().fit_transform(df['Transport'])
      df['socialAct_encode'] = LabelEncoder().fit_transform(df['Social Activity'])
      df['energyEfficiency_encode'] = LabelEncoder().fit_transform(df['Energy efficiency'])
      df['wasteBag_encode'] = LabelEncoder().fit_transform(df['Waste Bag Size'])
      #df['cookingWith_encode'] = LabelEncoder().fit_transform(df['Cooking_With'])
      # Transform the 'carbon emission' variable using Box-Cox transformation
      df['carbonEmission_transform'] = stats.boxcox(df['CarbonEmission'])[0]

      # Define X (features) and y (target) and remove duplicate features that will not be used in the model
      X = df.drop(['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source',
            'Transport', 'Social Activity',
            'Frequency of Traveling by Air',
            'Waste Bag Size','Energy efficiency','CarbonEmission',
                  'carbonEmission_transform'], axis=1)
      y = df['carbonEmission_transform']

      # Split the dataset into X_train, X_test, y_train, and y_test, 10% of the data for testing
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

      # Instantiate a linear regression model
      linear_model = LinearRegression()
      # Fit the model using the training data
      linear_model.fit(X_train, y_train)
      # For each record in the test set, predict the y value (transformed value of charges)
      # The predicted values are stored in the y_pred array
      y_pred = linear_model.predict(X_test)


      st.write('Predict your own Carbon Emission')

      #Body Type
      if  Body_Type== "underweight":
            bodytype_encode = 3
      elif Body_Type== "normal":
            bodytype_encode = 0
      elif Body_Type== "overweight":
            bodytype_encode = 2
      #for obese
      else:
            bodytype_encode = 1

      #Sex    
      if status == "Male":
            sex_encode=1
      else:
            sex_encode=0

      #Diet
      if Diet == "Omnivore":
            diet_encode = 0
      elif Diet == "Pescatarian":
            diet_encode = 1
      elif Diet == "Vegetarian":
            diet_encode = 3
      #vegan
      else:
            diet_encode = 2

      if How_often_shower == "Daily":
            howOftenShower_encode = 0
      elif How_often_shower == "Less Frequently":
            howOftenShower_encode = 1
      elif How_often_shower == "More Frequently":
            howOftenShower_encode = 2
      #twice a day
      else:
            howOftenShower_encode = 3
      #heating source
      if Heating == "Coal":
            heating_encode = 0
      elif Heating == "Natural Gas":
            heating_encode = 2
      elif Heating == "Wood":
            heating_encode = 3
      #elecricity 
      else:
            heating_encode = 1
      #Transport
      if Trans == "Public":
            transport_encode = 1
      elif Trans == "Walk/Bicycle":
            transport_encode = 2
      #private
      else:
            transport_encode = 0

      #Social activity
      if Social == "Often":
            socialAct_encode = 1
      elif Social == "Never":
            socialAct_encode = 0
      #sometimes
            socialAct_encode = 2

      #Frequency of traveling by air
      if Air == "Frequently":
            travelingByAir_encode = 0
      elif Air == "Rarely":
            travelingByAir_encode = 2
      elif Air == "Never":
            travelingByAir_encode = 1
      #very frequently
      else:
            travelingByAir_encode = 3

      #waste bag size 
      if Waste == "Large":
            wasteBag_encode = 1
      elif Waste == "Extra Large":
            wasteBag_encode = 0
      elif Waste == "Small":
            wasteBag_encode = 3
      #medium
      else:
            wasteBag_encode = 2

      #energy efficiency
      if EnergyEff == "No":
            energyEfficiency_encode = 0
      elif EnergyEff == "Sometimes":
            energyEfficiency_encode = 1
      #yes
      else:
            energyEfficiency_encode = 2



      predicted_carbon_emission_transformed = linear_model.predict([['Monthly Grocery Bill',
            'Vehicle Monthly Distance Km',
            'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
            'How Many New Clothes Monthly', 'How Long Internet Daily Hour',
            'travelingByAir_encode',
            'howOftenShower_encode', 'heating_encode', 'bodytype_encode',
            'sex_encode', 'diet_encode', 'transport_encode', 'socialAct_encode',
            'energyEfficiency_encode', 'wasteBag_encode']])

      predicted_CE = inv_boxcox(predicted_carbon_emission_transformed)

      st.write('Predicted Carbon Emission: ',round(predicted_CE[0], 0))

if __name__== "__main__":
            run()

