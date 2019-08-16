Titanic Data Science Project
===================
In this write-up, I will detail the why, what, and how of my first Data Science project. I tackled this project to apply what I have learned and gauge how powerful machine learning techniques can truly be.

My main objective was to analyze data about the Titanic's passengers from Kaggle (an online community for data scientists to find and publish data sets) and determine what sorts of people were more likely to survive. Below, I will recap my thought process and the steps I took towards my goal. 


----------

Data Description
-------------
I was given a training set and test set in the form of CSV files, filled with observed Titanic passengers, along with a variety of relevant information about them and whether they survived or not. The variables provided included: Passenger ID, Class, gender, age, number of parents and children, number of siblings and spouses, fare, cabin, and port of embarkation. 

My first thought was that some of these numerous variables were most likely largely inconsequential as to whether they survived or not. In addition, some observations were missing data. 

----------


Observations and Analysis
-------------------
**Observations**:
1. The Ticket variable contains a high ratio of duplicates and may not have a correlation with survival.
2.  The Cabin variable can be excluded as it is highly incomplete with many missing observations.
3.  PassengerID can also be excluded because it is not correlated to survival.
4.  The Name variable can be used to derive each observed person's title.

**Assumptions**
1.  Women and Children were more likely to have survived.
2.  Upper class passengers were more likely to survive.

After writing down these preliminary observations and assumptions, the next step was to verify their validity. 

*Pclass*
To observe the Pclass variable's significance, I compared all three classes and their corresponding survival rates, noting that there was a direct correlation between higher classes and higher survival rates.

*Gender*
Comparing male and female survival rates, I found that 74.2% of females survived while only 18.8% of males survived. This clearly indicates that females were much more likely to survive.

*SibSp*
SibSp was unique in that its correlation to survival was not linear, with values of 1 and 2 having the highest survival rates and 0 having the third highest.

*Parch*
This variable was similar to SibSp, with the only difference being that values of 3 had the highest survival rates. This could perhaps be due to a smaller number of observations, and also because people with more children were more likely to be chosen to survive.

*Embarked*
After comparing the three values of Embarked to one another, there was no clear correlation.

*Age*
I chose to analyze this variable in bands, meaning I grouped observations within a certain range into a category, then analyzed the categories as a whole. After this, I found a linear correlation between younger ages and higher survival rates.

At the end of all this, I concluded that age, gender, class, and size of family were all crucial elements in survival.
