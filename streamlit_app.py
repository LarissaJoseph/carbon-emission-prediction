import streamlit as st
#from streamlit.logger import get_logger
import pandas as pd
#import math
#from pathlib import Path
#import seaborn as sns
#import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
#from sklearn import metrics
#from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression#, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
#from scipy.stats import boxcox_normplot

#LOGGER = get_logger(__name__)


def run():
      st.set_page_config(
      page_title='Carbon Emission Predictor',
      page_icon=':earth_americas:',	
      )
      st.write('# Carbon Emission Predictor App')
      st.subheader('Predict your own Carbon Emission')
      csv_url = './Carbon_Emission.csv'
      df = pd.read_csv(csv_url)
      


      Sex = st.radio("Select Gender: ", ('Male', 'Female'))
      if (Sex == 'Male'):
            st.success("Male")
      else:
            st.success("Female")

      st.write("--------------------------------------------------------------------------")


      Body_Type = st.selectbox("Body Type: ",
                                    ['Underweight', 'Normal', 'Overweight', 'Obese'])

    
      st.write("Body Type: ", Body_Type)
      

      st.write("--------------------------------------------------------------------------")


      Diet = st.selectbox("Diet: ",
                                    ['Omnivore', 'Vegetarian', 'Pescatarian', 'Vegan'])

      
      st.write("Diet: ", Diet)
      
      st.write("--------------------------------------------------------------------------")
      
      How_often_shower = st.selectbox("How Often Do You Shower?: ",
                                    ['Daily', 'Less Frequently', 'More Frequently', 'Twice A Day'])

      
      st.write("How Often Do You Shower?: ", How_often_shower)
      
      st.write("--------------------------------------------------------------------------")
     
      Heating = st.selectbox("Your Home Heating Source: ",
                                    ['Coal', 'Natural Gas', 'Wood', 'Electricity'])

      st.write("Your Home Heating Source: ", Heating)
      
      st.write("--------------------------------------------------------------------------")
      
      Trans = st.selectbox("What Is Your Main Means of Transportation?: ",
                                    ['Public', 'Walk/Bicycle', 'Private'])

      
      st.write("What Is Your Main Means Of Transport?: ", Trans)
     
      st.write("--------------------------------------------------------------------------")
      
      Social = st.selectbox("How Often Do You Participate In Social Activities?: ",
                                    ['Often', 'Never', 'Sometimes'])

     
      st.write("How Often Do You Participate In Social Activities?: ", Social)
      
      st.write("--------------------------------------------------------------------------")
      
      Waste_Size = st.selectbox("Waste Bag Size: ",
                                    ['Small', 'Medium', 'Large', 'Extra Large'])

      
      st.write("Your Home Primiarily uses : ", Waste_Size)
      
      st.write("--------------------------------------------------------------------------")
      
      Air = st.selectbox("Frequency of using aircraft in the last month.: ",
                                    ['Frequently', 'Rarely', 'Never', 'Very Frequently'])

      
      st.write("Frequency of using aircraft in the last month.: ", Air )
     
      st.write("--------------------------------------------------------------------------")
      
      Car_Dist = st.slider("Montly Distance(km) By car ", 0, 9999)

     
      st.text('Selected: {}KM'.format(Car_Dist))
      st.write("--------------------------------------------------------------------------")
      
      Waste = st.slider("Waste Bags Weekly", 1, 7)

      
      st.text('Selected: {}'.format(Waste))
  
      st.write("--------------------------------------------------------------------------")
     
      How_longtv = st.slider("Number Of Daily Hours You Spend Infront Of The TV or Monitor", 1, 24)

      
      st.text('Selected: {}Hours'.format(How_longtv))
    
      st.write("--------------------------------------------------------------------------")
      
      Newclothes = st.slider("New Clothes Montly", 0,50 )

  
      st.text('Selected: {}'.format(Newclothes))
      
      st.write("--------------------------------------------------------------------------")
   
      Internet = st.slider("Daily Hours Spent On Internet", 0, 24)

      st.text('Selected: {} Hours'.format(Internet))
      
      st.write("--------------------------------------------------------------------------")
    
      EnergyEff = st.selectbox("Are Your Purchases Energy Efficient? ",
                                    ['Yes', 'No', 'Sometimes'])

      st.write("You Are Energy Aware", EnergyEff) 
      
      st.write("--------------------------------------------------------------------------")

      
      Bill = st.slider("Montly Grocery Bill", 50, 299)

      st.text('Selected: {}$'.format(Bill), )


      df = df.drop(['Vehicle Type', 'Recycling', 'Cooking_With'], axis=1)
      
      # Encode(changing categorical values to numerical values)
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
      df['carbonEmission_transform'] = stats.boxcox(df['CarbonEmission'])[0]

     

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
      
      #shower
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

       #Body Type
      if  Body_Type== "Underweight":
            bodytype_encode = 3
      elif Body_Type== "Normal":
            bodytype_encode = 0
      elif Body_Type== "Overweight":
            bodytype_encode = 2
      #for obese
      else:
            bodytype_encode = 1
      
      #Sex    
      if Sex == "Male":
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
      else:
            socialAct_encode = 2

      #energy efficiency
      if EnergyEff == "No":
            energyEfficiency_encode = 0
      elif EnergyEff == "Sometimes":
            energyEfficiency_encode = 1
      #yes
      else:
            energyEfficiency_encode = 2

      #waste bag size 
      if Waste_Size == "Large":
            wasteBag_encode = 1
      elif Waste_Size == "Extra Large":
            wasteBag_encode = 0
      elif Waste_Size == "Small":
            wasteBag_encode = 3
      #medium
      else:
            wasteBag_encode = 2

      
      input_data = [[Bill, Car_Dist, Waste, How_longtv, Newclothes, Internet,
                  travelingByAir_encode,howOftenShower_encode,heating_encode,bodytype_encode,
                  sex_encode,diet_encode,transport_encode,socialAct_encode,
                  energyEfficiency_encode,wasteBag_encode]]
      input_data
      
      
      
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
      linear_model.fit(X_train, y_train)
      linear_model.predict(X_test)
      
      predicted_carbon_emission_transformed = linear_model.predict(input_data)
      #predicted_CE = inv_boxcox(predicted_carbon_emission_transformed, stats.boxcox(df['CarbonEmission'])[1])



      predicted_CE = inv_boxcox(predicted_carbon_emission_transformed,stats.boxcox(df['CarbonEmission'])[1])

      st.write('Predicted Carbon Emission: ',round(predicted_CE[0], 0))

      

if __name__== "__main__":
           run()

