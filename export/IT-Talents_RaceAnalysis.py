
# coding: utf-8

# # IT-Talents Race Analysis
# ### Moritz Göckel
# 
# ## Table of Content
# 
# ### 1. Loading and Cleaning
# 
# ### 2. Analyzing
# 2.1. Does the weather affect the fuel consumption?
# 
# 2.2. Which track is the most popular?
# 
# 2.3. Cont' fuel_consumption and weather
# 
# 2.4. Cont' popular tracks and pareto's law
# 
# 2.5. Which track gets finished more/less often?
# 
# 2.6. How does fuel_consumption differ on multiple tracks?
# 
# 2.7. How much money do the players pay on the different tracks?
# 
# 2.8. Lets find correlations
# 
# 2.9. Is there a correlation between the popularity of a track and the money?
# 
# 2.10. What is the most likely weather?3.11 On which weather do the player spend the most money?
# 
# 2.11. On which weather do the player spend the most money?
# 
# 2.12. Is the opponent or the challenger more likely to win?
# 
# 2.13. Do older accounts win over newer accouts?
# 
# 2.14. How many races get finished?
# 
# 2.15. On which weekdays happen the most races?
# 
# 2.16. At which times happen the most races?
# 
# 2.17. How did the amount of races per month develope?
# 
# ### 3. Predicting
# 3.1. Features
# 
# 3.2. Training and test data
# 
# 3.3. Machine Learning
# 
# 3.4. Predicting

# In[1]:

# We want inline charts
get_ipython().magic('matplotlib inline')


# Importing the basic libraries

# In[2]:

import pandas as pd # For data management
import numpy as np # For math and array functions
import matplotlib.pyplot as plt # For plotting


# ## 1. Loading and Cleaning
# We will load the data and check if we need to clean something up

# In[3]:

# Loading the dataset with pandas
allRaces = pd.read_csv('races.csv', sep=';', index_col='id')


# In[4]:

# having a quick peek at the header
list(allRaces)


# In[5]:

# checking out how the values could look like
allRaces.values[1]


# In[6]:

# Fuel consumption is a string but should be a number, so lets fix that
allRaces[['fuel_consumption']] = allRaces[['fuel_consumption']].apply(pd.to_numeric, errors='coerce')


# In[7]:

allRaces.values[1] # Now fuel consumption is a nice float variable


# There will be more to change and to clean later, but we will just do that as soon as we need to

# ## 2. Analyzing
# In this chapter we answer some questions about the data 

# ### 2.1 Does the weather affect the fuel consumption?
# This question can only be awnsered by completed races. Therefore we only include finished races

# In[8]:

finished = allRaces.loc[allRaces['status'] == 'finished'] # Select all the finished races
finished.groupby(['weather'], as_index=False).mean()[['weather', 'fuel_consumption']] # Get the mean of every weather


# There seems to be no correlation. The averages of the fuel_consumption are the same in all kinds of weather conditions.

# ### 2.2 Which track is the most popular?
# For this question we regard all races again

# In[9]:

#Lets get the most populat tracks
plt.hist(allRaces['track_id'].values)
plt.show()


# As you can see this is a very uneven distribution. Track 12 seems to be the most popular by far. 
# Lets check the absolute numbers

# In[10]:

allRaces['track_id'].value_counts()


# We will get back analyzing the track data so lets just save the amount of races per track and the IDs

# In[11]:

# Lets save that
tracks = pd.DataFrame(allRaces['track_id'].value_counts())
tracks.columns = ['races']
tracks.index.name = 'track_id'


# ### 2.3 Cont' fuel_consumption and weather
# As track 12 is the most popular, lets check wether there is an correlation between the fuel_consumption and the weather on this one

# In[12]:

# The most popular track seems to be track 12, so lets check our "weather affects fuel_consumption"-hypothesis only on track 12
finishedOnTrack12 = finished.loc[allRaces['track_id'] == 12]
finishedOnTrack12.groupby(['weather'], as_index=False).mean()[['weather', 'fuel_consumption']]


# There still seems to be no significant correlation between the weather and the fuel_consumption. But there is still a small effect. We can say that driving in the sun takes slightly less fuel than in thundery weather

# ### 2.4 Cont' popular tracks and pareto's law

# In[13]:

# Lets further investigate the track preferances of the players, this time in percentages
popularity = allRaces['track_id'].value_counts() / allRaces['track_id'].value_counts().values.sum() * 100

plt.figure(figsize = (16,4))
popularity.plot.bar()


# As you can see, track 12 and 3 account to over 84% of the races. This is parettos law: 2/12 tracks account for more than 80% of the races. The remaining 10 tracks only accout for about 14% of the races.

# In[14]:

# Lets save that
tracks['pupularity'] = popularity


# ### 2.5 Which track gets finished more/less often?
# Not all races get finished, many of them get canceled before and some after start. Lets check wether some tracks have more canceled races than others.  

# In[15]:

# Are there any maps that get finished more often?
finishedRatio = finished['track_id'].value_counts() / allRaces['track_id'].value_counts() * 100
finishedRatio = finishedRatio.sort_values()

plt.figure(figsize = (16,4))
finishedRatio.plot.bar(logy=True)


# The most finished track is track 14, the track that gets finished the least is track 3. As track 3 and 12 are the most popular tracks, it is interesting to see that they have quite a big gap regarding the finished to unfinished races ratio. Track 3 gets finished a lot less than track 12.

# In[16]:

# Lets save that
tracks['finishedRatio'] = finishedRatio


# ### 2.6 How does fuel_consumption differ on multiple tracks?
# As the tracks are probably different in length, it is likely that they also differ in average fuel_consumption. Lets check that, for obvious reasons lets only consider finished races

# In[17]:

meanFuelConsumption = pd.DataFrame(finished[['track_id', 'fuel_consumption']].dropna().groupby(['track_id'], as_index=False).mean())
meanFuelConsumption = meanFuelConsumption.set_index(['track_id'])
meanFuelConsumption.sort_values('fuel_consumption').plot.bar()


# There is quite a range of neccessary fuel for the different tracks. Probably they differ a lot in length. We can assume that probably track 4 is the shortest and track 14 the longest. By this meassure track 14 is probably more than 20 times longer than track 4. The most popular track 12 is pretty much in the middle with its needed fuel of 9.48 units on average.

# In[18]:

# Lets save that
tracks['meanFuelConsumption'] = meanFuelConsumption['fuel_consumption']


# ### 2.7 How much money do the players pay on the different tracks?
# Lets take the averages for all tracks

# In[19]:

# Lets also include the average money per track
meanMoney = pd.DataFrame(finished[['track_id', 'fuel_consumption', 'money']].dropna().groupby(['track_id'], as_index=False).mean())
meanMoney = meanMoney.set_index(['track_id'])

plt.figure(figsize = (16,4))
meanMoney.sort_values('money')['money'].plot.bar()


# Track 12, which is the most popular has ranks very high when we compare the money. An explenation for that could be that only new players who dont have a lot of money play the other maps and the more experienced and 'richer' players play only track 12 and 6. But this is just a theory.

# In[20]:

# Lets save that
tracks['meanMoney'] = meanMoney['money']


# ### 2.8 Lets find correlations
# We got quite some knowladge about the different tracks now. Take a look at the table bellow to see what we got. Lets find some correlations between the values.

# In[21]:

tracks


# That is how our tracks table looks like, now lets take a look on the correlation table. First the numbers and then the heatmap

# In[22]:

tracks.corr()


# In[23]:

# Lets see if there is any correlations between the different attributes of the tracks
import seaborn as sns

plt.figure(figsize = (16,9))

corr = tracks.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# There is quite a correlation between popularity and meanMoney. We stated an explenation for that in the previous chapter. The other correlations are not very strong.

# ### 2.9 Is there a correlation between the popularity of a track and the money?
# Lets investigate this by visualising

# In[24]:

# There seems to be a correlation between the Money and the popularity of the track, lets investigate
x, y = tracks[['pupularity', 'meanMoney']].values.T
plt.scatter(x, y)
plt.ylabel("Popularity")
plt.xlabel("Mean Money")
plt.show()


# Track 12 is an outlayer. It seems to be a litte far fatched to talk of a correlation here, but it is interesting that the most popular track is also one of the ones with the heighest mean money. Maybe only casual players with little money race on the other maps.

# ### 2.10 What is the most likely weather?

# In[25]:

plt.figure(figsize = (7,7))
(allRaces['weather'].value_counts() / allRaces['weather'].value_counts().sum() * 100).plot.pie()


# Sunny is the most likely. How about on the different tracks?

# In[26]:

tracksWeather = pd.DataFrame(columns=('rainy', 'snowy', 'sunny', 'thundery'))
tracksWeather.index.name = 'track_id'

for x in range(3, 15):
    onlyTrack = allRaces.loc[allRaces['track_id'] == x]
    tracksWeather.loc[x] = onlyTrack['weather'].value_counts(normalize=True).sort_index().values

tracksWeather


# The distribution of the probabilites for the different weathers seem to be quite similar on the different tracks. Eighter is is random or players allways choose the weather with always the same chances on every track.

# In[27]:

# Lets save that
tracks = tracks.join(tracksWeather)


# ### 2.11 On which weather do the player spend the most money?

# In[28]:

weatherMoneyData = finished.groupby(['weather'], as_index=False).mean()[['weather', 'money']].sort_values("money")
weatherMoneyData.plot.bar(x = weatherMoneyData['weather'].values, logy=True)


# They spend the most money in the sun and the least in the thunder

# ### 2.12 Is the opponent or the challenger more likely to win?

# In[29]:

challenger = 0
opponent = 0

for row in finished.itertuples():
    if getattr(row, 'winner') == getattr(row, 'challenger'):
        challenger = challenger + 1
    elif getattr(row, 'winner') == getattr(row, 'opponent'):
        opponent = opponent + 1
    else:
        raise Exception('Winner is nighter opponent nor challenger')
        
challenger / (challenger + opponent) * 100


# The challenger is slightly more likely to win. Challengers in 56% of the time. We can use that later as a feature for our prediction of races.

# ### 2.13 Do older accounts win over newer accouts?
# Older accounts should have a lower id-number, while newer accounts should have a heigher one. Therefore could lower id drivers have more experience and win more often. Lets check if that is true.

# In[30]:

lowerId = 0
higherId = 0

for row in finished.itertuples():
    winnerId = getattr(row, 'winner')
    loserId = -1
    if winnerId == getattr(row, 'challenger'):
        loserId = getattr(row, 'opponent')
    elif winnerId == getattr(row, 'opponent'):
        loserId = getattr(row, 'challenger')
        
    if loserId == -1:
        raise Exception('Something went wrong')
    
    if winnerId > loserId:
        higherId = higherId + 1
    elif winnerId < loserId:
        lowerId = lowerId + 1
    else:
        raise Exception("Something went wrong")
    
olderAccountsWin = lowerId / (lowerId + higherId) * 100

plt.figure(figsize = (7,7))
plt.pie([olderAccountsWin, 100 - olderAccountsWin], labels=["Lower ID wins", "Higher ID wins"])


# This hypothesis can assumed to be true, lowerIds (probably older accounts) win significantly more often than higher ids (66%). We can also use that later as a feature for our prediction of races.

# ### 2.14 How many races get finished?

# In[31]:

allRaces['status'].value_counts(normalize=True) * 100


# In[32]:

plt.figure(figsize = (7,7))
allRaces['status'].value_counts(normalize=True).plot.pie()


# Only 65% of all races get finished. All the others are declined, retired or waiting.

# ### 2.15 On which weekdays happen the most races? 

# In[33]:

# First we convert the string of race_driven into datetimes
allRaces['race_driven'] = pd.to_datetime(allRaces['race_driven'], errors='coerce')


# In[34]:

# Lets create a field for weekdays
allRaces['weekday'] = allRaces.apply(lambda row: row['race_driven'].isoweekday(), axis=1)


# In[35]:

plt.figure(figsize = (16,4))
allRaces['weekday'].value_counts().sort_index().plot.line()


# Most races are done on sunday and monday. Thursday and friday are the days in which the least races happen.

# ### 2.16 At which times happen the most races?

# In[36]:

allRaces['hour'] = allRaces.apply(lambda row: row['race_driven'].hour, axis=1)


# In[51]:

plt.figure(figsize = (16,4))
allRaces['hour'].value_counts().sort_index().plot.line()


# The most popular hours for races are 21:00 and 20:00. Between 3:00 and 5:00 are the least popular. Just as expected.

# ### 2.17 How did the amount of races per month develope? 

# In[38]:

import datetime
import math
import time

firstDate = allRaces.values[0][1].timestamp();
allRaces['time_number'] = allRaces.apply(lambda row: (row['race_driven'].timestamp() - firstDate), axis=1, ignore_failures=True)
dates = allRaces.dropna().apply(lambda row: int(row['time_number'] / 60 / 60 / 24 / 30), axis=1, ignore_failures=True)


# In[39]:

y = dates.value_counts().values
x = dates.value_counts().index

plt.figure(figsize = (16,4))
plt.scatter(x, y)
plt.ylabel("Races")
plt.xlabel("Month")
plt.show()


# As you can see in the chart, the first couple of months there was a spike in players and then a steady decline after the 20th month. There are little spikes in regular intervals but it looks quite smooth.

# ## 3 Prediction
# Given two players we want to predict the outcome of the race. First of all we have to engineer some features.
# 
# ### 3.1 Features
# 
# #### Feature ideas
# 1. Who is the challanger? As seen before, this has some predictive value.
# 2. Who had more games? I think the more games a player had, the more experieced and skilled he should be.
# 3. Who has the older account? We have proofen before that this has predictive value.
# 4. Who won more games? People who win more probably have a higher chance to win again.
# 5. Who has a higher win/loos ratio? Not only absolute wins but also wins in relation to losses are important.
# 6. Who won more games agains the same opponend? If those two players played against each other before, we should know about that.
# 
# We are developing a function now that returns these features in relation between the two players. We only regard races that have been completed before the current time, this makes sure our algorithm cant look into the future.  

# In[40]:

rp = allRaces.dropna()
def getPredictionFeatures(time, challenger, opponent):
    challenger_games = rp[(rp.time_number < time) & ((rp.challenger == challenger) | (rp.opponent == challenger))]
    challenger_wins = challenger_games[challenger_games.winner == challenger]
    
    opponent_games = rp[(rp.time_number < time) & ((rp.challenger == opponent) | (rp.opponent == opponent))]
    opponent_wins = opponent_games[opponent_games.winner == opponent]
    
    samesetup_games_ids = opponent_games.index.intersection(challenger_games.index)
    samesetup_games = rp[rp.index.isin(samesetup_games_ids)]
        
    if opponent_games.shape[0] + challenger_games.shape[0] != 0:
        gamesindex = challenger_games.shape[0] / (opponent_games.shape[0] + challenger_games.shape[0])
    else:
        gamesindex = 0.5
    
    if opponent_wins.shape[0] + challenger_wins.shape[0] != 0:
        winindex = challenger_wins.shape[0] / (opponent_wins.shape[0] + challenger_wins.shape[0])
    else:
        winindex = 0.5
    
    if challenger_games.shape[0] != 0 and opponent_games.shape[0] != 0:
        challengerWinCount = challenger_wins.shape[0] / challenger_games.shape[0]
        if (challengerWinCount + (opponent_wins.shape[0] / opponent_games.shape[0])) != 0:
            winCountIndex = challengerWinCount / (challengerWinCount + (opponent_wins.shape[0] / opponent_games.shape[0]))
        else:
            winCountIndex = 0.5
    else:
        winCountIndex = 0.5
    
    if samesetup_games.shape[0] != 0:
        samesetupWinIndex = samesetup_games[samesetup_games.winner == challenger].shape[0] / samesetup_games.shape[0]
    else:
        samesetupWinIndex = 0.5
    
    accountAgeIndex = int(challenger > opponent)
    
    return gamesindex, winindex, winCountIndex, samesetupWinIndex, accountAgeIndex


# Lets now see how the feature array would look like

# In[41]:

getPredictionFeatures(1666080, 5, 4)


# All these values can range from 0 (bad for the challenger) to 1 (good for the challenger). If the algorithm runs into a division by zero (means there are no samples) it will return the variable as 0.5 
# 
# We calculate now all the features and the outcomes from all the finished games and put them in our x (features) and y (outcome) array. We always provide the current timestamp to ensure the algorithm cant look into the future.

# In[42]:

x = []
y = []

for index, row in rp.iterrows():
    pred = getPredictionFeatures(row.time_number, row.challenger, row.opponent)
    truth = int(row.winner == row.challenger)
    x.append(pred)
    y.append(truth)


# ### 3.2 Training and test data
# 
# Now we split our x and y data into training and test datasets (80%/20%). We use sklearn for machine learning

# In[43]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  


# ### 3.3 Machine Learning
# 
# Now we train the machine learning algorithm of our choice with the training data and print out how he scores with our testing data. We treat this as a classification problem and therefore predict "Is the challenger going to win?". If the awnser is 0, then our prediction is "No", if the awnser is 1, then our prediction is "Yes, he is going to"

# First we give all the features to the GradientBoostingClassifier. I tried a lot of different algorithms, this one seems to work quite well.

# In[44]:

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(learning_rate=0.5, n_estimators=10)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# A precition of 75% is quite good, lets see which features are the most important

# In[45]:

clf.feature_importances_


# Especially feature 3 (winCountIndex) and feature 4 (samesetupWinIndex) are useful for the algorithm. Makes sense, as they provide information about the experience of the players and about their previous matches against each other.
# 
# Lets see if these features are enough for the algorithm. Maybe the other features are just noise, so lets remove them and try again: 

# In[46]:

x_man = np.array(x)[:,1:4]
x_man_train, x_man_test, y_man_train, y_man_test = train_test_split(x_man, y, test_size=0.2)


# In[47]:

clf_man = GradientBoostingClassifier(learning_rate=0.5, n_estimators=10)
clf_man.fit(x_man_train, y_man_train)
clf_man.score(x_man_test, y_man_test)


# The performance is slightly worse but quite similar. While these features are enough for the algorithm to perform well, the others still seem to help a little.
# 
# Lets now automatically select the best two features with the chi2 algorithm and check how the model performes with these:

# In[48]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

x_selected = SelectKBest(chi2, k=2).fit_transform(x, y)
x_selected_train, x_selected_test, y_selected_train, y_selected_test = train_test_split(x_selected, y, test_size=0.2)


# In[49]:

clf_auto = GradientBoostingClassifier(learning_rate=0.5, n_estimators=10)
clf_auto.fit(x_selected_train, y_selected_train)
clf_auto.score(x_selected_test, y_selected_test)


# As we can see the precition gets even worse. Therefore we can conclude that all our features are useful for the algorithm.
# Our final precition is 75% with the first (and complete) set of features.

# ### 3.4 Predicting
# 
# As you can se we got a precition of 75%, I guess that is decent. We can might that score by trying other algorithms and other settings.
# 
# Now lets use our model to predict the 200ed race:

# In[50]:

print("Prediction: " + str(clf.predict([x[200]])))
print("Truth: " + str([y[200]]))


# In this case we are lucky and the prediction and the truth are equal. As the precition is 75% it will be correct 75% of the time.
# 
# Thank you for reading :)
