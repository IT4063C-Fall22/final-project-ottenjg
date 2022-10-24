#!/usr/bin/env python
# coding: utf-8

# # Project Title

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# <br>
# The effect education has on crime rates in Cincinnati
# 
# 
# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# <br>
# What is the coorelation education has to rising or falling crime rates in Cincinnati?
# <br>
# Which areas show the biggest coorelation between education and crime rates, and which ones show the least?
# 
# 
# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# <br>
# I would still be using geographical charts and models like heat maps. I would use line graphs to compare education level/investment amounts to crime rates. Bar charts would be useful as well. 
# <br>
# An answer I want to see or atleast an outcome would be to help create solutions to crime rates using education. 
# 
# 
# ## Data Sources
# *What 3 data sources have you identified for this project?*
# <h3>Crimes committed in Cincinnati. Includes location data in particular. API</h3>
# <br>
# https://data.cincinnati-oh.gov/safety/PDI-Police-Data-Initiative-Crime-Incidents/k59e-2pvf
# <br>
# <h3>Hamilton Count School Locations. API</h3>
# <br>
# https://data-cagisportal.opendata.arcgis.com/datasets/countywide-school-locations/explore?showTable=true
# <br>
# <h3>Hamilton county school ratings.</h3>
# <br>
# https://infogram.com/ohio-report-card-2022-1hd12yx1ykkow6k
# <h3>Cincinnati Census Data. database</h3>
# <br>
# https://www.cincinnati-oh.gov/planning/maps-and-data/census-demographics/2020-census-data/
# <br>
# 
# *How are you going to relate these datasets?*
# <br>
# i will relate crime to school ratings and education levels using common location data.
# 
# *How will you use this data to answer your project question?*
# <br>
# i will use location data to show maps (heat maps). I will answer the question by showing a connection between crime and education using location data. I will see how scores for schools could be impacted, and will try to see if improving school performance can lower crime rates.

# In[4]:


# Python ≥3.10 is required
import sys
assert sys.version_info >= (3, 10)

# Common imports
import numpy as np
import pandas as pd
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

# other imports
from wordcloud import WordCloud


# In[45]:


# Importing the data
crime_data = pd.read_csv('./datasets/PDI__Police_Data_Initiative__Crime_Incidents.csv')
traffic_stops = pd.read_csv('./datasets/PDI__Police_Data_Initiative__Traffic_Stops__Drivers_.csv')
shootings = pd.read_csv('./datasets/PDI__Police_Data_Initiative__CPD_Shootings.csv')


# In[70]:


traffic_stops.sample(10)


# In[59]:


crime_data.sample(10)


# In[50]:


shootings.sample(10)


# In[8]:


crime_data.drop(crime_data.index[crime_data['COMMUNITY_COUNCIL_NEIGHBORHOOD'] == 'nan'], inplace = True)
crime_data['DATE_REPORTED'] = pd.to_datetime(crime_data['DATE_REPORTED']).dt.normalize()
crime_data.drop(crime_data.index[crime_data['SUSPECT_RACE'] == 'UNKNOWN'], inplace = True)
crime_data['COMMUNITY_COUNCIL_NEIGHBORHOOD'] = crime_data['COMMUNITY_COUNCIL_NEIGHBORHOOD'].astype("str")


# In[33]:


traffic_stops.drop(traffic_stops.index[traffic_stops['COMMUNITY_COUNCIL_NEIGHBORHOOD'] == 'NaN'], inplace = True)
traffic_stops.drop(traffic_stops.index[traffic_stops['LATITUDE_X'] == 'NaN'], inplace = True)
traffic_stops['DATE_REPORTED'] = pd.to_datetime(traffic_stops['DATE_REPORTED']).dt.normalize()


# In[40]:


traffic_stops.sample()


# In[76]:


traffic_stops_sample = traffic_stops.sample(1000)


# In[91]:


# top 10 total crime
crime_data['COMMUNITY_COUNCIL_NEIGHBORHOOD'].value_counts()[:10].plot(kind='barh')


# In[15]:


# counts of crime by race
crime_data['SUSPECT_RACE'].value_counts()[:10].plot(kind='barh')


# In[20]:


crime_data['COMMUNITY_COUNCIL_NEIGHBORHOOD'].where(crime_data['SUSPECT_RACE'] == 'BLACK').value_counts()[:10].plot(kind='barh')


# In[21]:


crime_data['COMMUNITY_COUNCIL_NEIGHBORHOOD'].where(crime_data['SUSPECT_RACE'] == 'WHITE').value_counts()[:10].plot(kind='barh')


# In[42]:


traffic_stops['ACTIONTAKENCID'].where(traffic_stops['RACE'] == 'WHITE').value_counts()[:10].plot(kind='barh')


# In[44]:


traffic_stops['ACTIONTAKENCID'].where(traffic_stops['RACE'] == 'BLACK').value_counts()[:10].plot(kind='barh')


# In[77]:


shootings['COMMUNITY_COUNCIL_NEIGHBORHOOD'].value_counts()[:10].plot(kind='barh')


# In[88]:


crime_data['DATE_REPORTED'].value_counts()[:].plot()


# In[89]:


crime_data['DATE_REPORTED'].dt.month.value_counts()[:].plot(kind = "barh")


# In[91]:


shootings['MONTHOCCURED'].value_counts()[:].plot(kind = "barh")


# <h2>Machine Learning Plan</h2>
# <br>
# <p>My goal is to use machine learning to answer my new question of how crime affects different areas and demographics within Cincinnati. What type of machine learning model are you planning to use?
# i plan to use supeervised regression. This is because i want to use factors that could play into the likelyhood of a crime happening and then predict the outcome or likelihood of a crime.
# What are the challenges have you identified/are you anticipating in building your machine learning model?
# I will have to work on identifying different columns that are related for example different columns in traffic stops indicate an arrest. I'll have to identify those if i want to make a simple prediction of whether someone will be arrested or not. 
# How are you planning to address these challenges? I will combine columns depending on the case. </p>

# In[92]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

