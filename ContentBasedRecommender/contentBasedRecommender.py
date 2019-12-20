# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 12/12/2019 
# Recommender system tutorial: content-based approach

import pandas as pd #Dataframe manipulation library
from math import sqrt #sqrt math function is needed
import numpy as np
import matplotlib.pyplot as plt

# acquiring data (execute only once)
#!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
#print('unziping ...')
#!unzip -o -j moviedataset.zip 

print("***********************")
print("CONTENT BASED")
print("***********************")

# read each file into their Dataframes
movies_df = pd.read_csv('movies.csv') #Storing the movie information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')#Storing the user information into a pandas dataframe

# create years column keeping the year from the title column (just data manipulation)
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False) #Using regular expressions to find a year stored between parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False) #Removing the parentheses
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '') 	  #Removing the years from the 'title' column
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip()) #Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['genres'] = movies_df.genres.str.split('|') # genres column values splitted into a list :  Adventure|Children|Fantasy -> [Adventure, Children, Fantasy]

# Convert the list of genres to a vector where each column corresponds to one possible value of the feature.

moviesWithGenres_df = movies_df.copy() #Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
moviesWithGenres_df = moviesWithGenres_df.fillna(0) #Filling in the NaN values with 0 to show that a movie doesn't have that column's genre

ratings_df = ratings_df.drop('timestamp', 1) # removes (drop()) timestamp column from a dataframe

"""
moviesWithGenres_df:

       movieId                        title  ... Film-Noir (no genres listed)
0            1                    Toy Story  ...       0.0                0.0
1            2                      Jumanji  ...       0.0                0.0
2            3             Grumpier Old Men  ...       0.0                0.0
"""

"""
	Content-Based recommendation system: figure out what a user's favourite aspects of an item is, and then recommends items that present those aspects
	in this case figure out the input's favorite genres from the movies and ratings given.
	
	ADVANTAGES: Highly personalized for the user, leanr user preferences
	DISADVANTAGES: Determining what characteristics of the item the user dislikes or likes is not always obvious, low quality item recommendations might happen
"""

userInput = [
            {'title':'Akira', 'rating':4},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Trainspotting', 'rating':4.5}
         ] 

inputMovies = pd.DataFrame(userInput)
# print(inputMovies)

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())] # filtering out the movies by title rated by user
inputMovies = pd.merge(inputId, inputMovies) # get the movieId (it's implicitly merging it by title)
inputMovies = inputMovies.drop('genres', 1).drop('year', 1) # remove information we won't use from the input dataframe

#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
"""
print(inputMovies)

   movieId          title  rating
0        1      Toy Story     3.5
1        2        Jumanji     2.0
2      296   Pulp Fiction     5.0
3      778  Trainspotting     4.5
4     1274          Akira     4.0
"""
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())] #Filtering out the movies from the input
"""
print(userMovies)
      movieId          title  ... Film-Noir (no genres listed)
0           1      Toy Story  ...       0.0                0.0
1           2        Jumanji  ...       0.0                0.0
293       296   Pulp Fiction  ...       0.0                0.0
765       778  Trainspotting  ...       0.0                0.0
1246     1274          Akira  ...       0.0                0.0
"""

userMovies = userMovies.reset_index(drop=True) #Resetting the index to avoid future issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1) #Dropping unnecessary issues due to save memory and to avoid issues
"""
print(userGenreTable)

   Adventure  Animation  Children  ...  Western  Film-Noir  (no genres listed)
0        1.0        1.0       1.0  ...      0.0        0.0                 0.0
1        1.0        0.0       1.0  ...      0.0        0.0                 0.0
2        0.0        0.0       0.0  ...      0.0        0.0                 0.0
3        0.0        0.0       0.0  ...      0.0        0.0                 0.0
4        1.0        1.0       0.0  ...      0.0        0.0                 0.0

"""

"""
start learning the input's preferences
turn each genre into weights: using the input's reviews and multiplying them into the input's genre table and then summing up the resulting table by column. 
This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.
"""
#Dot produt to get Weights Genre Matrix = input User Rating x Movie Matrix
userProfile = userGenreTable.transpose().dot(inputMovies['rating']) # User Profile -> you can recommend movies that satisfy the user's preferences
"""
print(userProfile)
Adventure              9.5
Animation              7.5
Children               5.5
Comedy                13.0
Fantasy                5.5
Romance                0.0
Drama                  9.5
Action                 4.0
Crime                  9.5
Thriller               5.0
Horror                 0.0
Mystery                0.0
Sci-Fi                 4.0
"""

# extracting the genre table from the original dataframe:
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId']) # Now let's get the genres of every movie in our original dataframe
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1) #And drop the unnecessary information
print("genreTable")
print(genreTable.head())
"""
genreTable
         Adventure  Animation  Children  ...  Western  Film-Noir  (no genres listed)
movieId                                  ...                                    
1              1.0        1.0       1.0  ...      0.0        0.0                 0.0
2              1.0        0.0       1.0  ...      0.0        0.0                 0.0
3              0.0        0.0       0.0  ...      0.0        0.0                 0.0
4              0.0        0.0       0.0  ...      0.0        0.0                 0.0
5              0.0        0.0       0.0  ...      0.0        0.0                 0.0
"""


# With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average 
# of every movie based on the input profile and recommend the top twenty movies that most satisfy it.

#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
print("Recommendation table:")
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(10).keys())] ) #The final recommendation table

"""
Recommendation table:
       movieId  ...  year
2902      2987  ...  1988
4625      4719  ...  2001
4923      5018  ...  1991
8605     26093  ...  1962
9296     27344  ...  1999
13250    64645  ...  1968
15001    75408  ...  2008
15073    76153  ...  2002
16055    81132  ...  2010
26442   122787  ...  1959
"""

"""
	COLLABORATIVE FILTERING (User-User Filtering): algo used to recommend items to users based on information taken from the user	
	There are several methods of finding similar users (Correlation Function)
	
	- Select a user with the movies the user has watched
	- Based on his rating to movies, find the top X neighbours
	- Get the watched movie record of the user for each neighbour.
	- Calculate a similarity score using some formula
	- Recommend the items with the highest score
	
	Advantages: 
	- takes consideration of other user's ratings  
	- do not need to study/extract info from the recommended item
	- the system adapts to the user's interests which might change over time
	
	Disadvantages:
	- maybe slow approximation function 
	- privacy issues when trying to learn the user's preferences
"""

print("***********************")
print("COLLABORATIVE FILTERING")
print("***********************")

# Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
print("user subset table:")
print(userSubset.head())

# group up the rows by user ID
userSubsetGroup = userSubset.groupby(['userId']) #. groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
print("userID=1130:")
print(userSubsetGroup.get_group(1130))

# sort these groups so the users that share the most movies in common with the input have higher priority (we won't go through every single user)
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True) #Sorting it so users with movie most in common with the input will have priority

# Now lets look at the first user
print(userSubsetGroup[0:3])

# now compare subset of users to our specified user and find the one that is most similar: using Pearson Correlation Coefficient
# we select a subset of users to iterate through
userSubsetGroup = userSubsetGroup[0:100]

# calculate the Pearson Correlation between input user and subset group
pearsonCorrelationDict = {} # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient

for name, group in userSubsetGroup: # for every user group in our subset
    group = group.sort_values(by='movieId')     #  sort the input and current user group so the values aren't mixed up later on
    inputMovies = inputMovies.sort_values(by='movieId')
    nRatings = len(group)     #Get the N for the formula
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]     # get the review scores for the movies that they both have in common
    tempRatingList = temp_df['rating'].tolist()                                       # then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempGroupList = group['rating'].tolist()     									  # put the current user group reviews in a list format
    
	#Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonCorrelationDict.items()
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

# Now let's get the top 50 USERS that are most similar to the input
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topUsers.head())

# start recommending MOVIES to the input user

# taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight.  
# 1 get the movies watched by the users in our pearsonDF from the ratings dataframe 
# 2 store their correlation in a new column called "_similarityIndex"
# 3 taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

# multiply the movie rating by its weight (The similarity index)
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

# sum up the new ratings 
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

# divide it by the sum of the weights
recommendation_df = pd.DataFrame() #Creates an empty dataframe
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex'] #Now we take the weighted average
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

# sort it and see the top 20 movies that the algorithm recommended
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print("RECOMMENDATION LIST:")
print(recommendation_df.head(10))
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
