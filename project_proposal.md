# Predicting economic recovery from Covid-pandemic

## Elevator pitch
xxx

## Data
Data will be gathered from global free-of-charge databanks having economic and other relevant indicator data. These include:
- [Our world in data](https://ourworldindata.org/)
- [Iternational Monetary Fund](https://www.imf.org/external/datamapper)
- [World Bank](https://databank.worldbank.org/source/world-development-indicators)

These inlcude wast amounts of differnet kind of indicators which could be used in project. During project also other sources will be searched for.

Data will be gathered in formats provided and changed to csv locally. Data will be stored locally, as there is need to update data from datasources rather inrequently. Most of the indicators are annual, although it is hoped that we would find monthly indicators as well. 

Main challenge is to merge different kind of data to uniform format. Data is provied in different formats (excel, json, csv) but also is structured differently. This certainly requires considerable datawrangling so that reliable data set is formed.

## Data analysis
Aim is to estimate how fast countries and world regions will recover for economic downturn caoused be Covid-pandmic itself and response to it. For this we need up-to-date indicators describing key features of the societys which influnece on eoconomic recovery. 

Initially several possible indicators to predict economic recovery are identified:
- vaccination rate
- realince on tourism
- financial stimulus pakcets
- death rate
- pre pandemic GDP growth rate
- unemployment rate
- bankruptcy rate during pandemic
- other concurrent economic stabilities
- prepandemic debt rate
- current debt rate

Epic challenge will be algorithm, which produces country specific recovery estimates from the data set.  

## Communication of results
Data will be available for users in web. Oone option for this is [heroku.com platform](https://www.heroku.com/). In addition to final country speciic estimates users can explore indicators in wich estimates are based on. 

## Operationalization
This service will provide considerebale added value for several groups. For economic playrs it is crucial to know when speicific countries will recover for economic crisis. This could be user for example in investment decision. Also policymakers in globan and national level can exploit the service for planning strategies how the expedite economic recovery.




## Template

A. TITLE
What should we call our mini-project? Should be cool and descriptive. 

B. ELEVATOR PITCH [max 400 characters]
Which is the target group of our mini-project? 
What needs does it address? 
Is there added value gained from the data we are planning to use?

C. DATA
Which data sources are we planning to use? 
Which is the data management plan? 
Is there data cleaning/wrangling involved?

D. DATA ANALYSIS
What will we try to learn? 
Which variables are we more likely to use? 
Which (machine/statistical) learning methods might be more beneficial and why?

E. COMMUNICATION OF RESULTS
How are we planning to summarize and visualize the results? 
Which visualization methods will be more useful to our defined target group?

F. OPERATIONALIZATION
What is the added value? 
Why is this application/blog post useful to our defined target group?