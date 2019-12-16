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

print("Start script...")

# read each file into their Dataframes
movies_df = pd.read_csv('movies.csv') #Storing the movie information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')#Storing the user information into a pandas dataframe

# create years column keeping the year from the title column
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
	Content-Based recommendation system: 
	figure out what a user's favourite aspects of an item is, and then recommends items that present those aspects
	
	figure out the input's favorite genres from the movies and ratings given.
"""

"""
userInput = [
            {'title':'Trainspotting', 'rating':5},
            {'title':'War Z', 'rating':3.5},
            {'title':'Ghost', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Apocalipse Now', 'rating':4.5}
         ] 
"""

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 

inputMovies = pd.DataFrame(userInput)
# print(inputMovies)

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())] #Filtering out the movies by title

inputMovies = pd.merge(inputId, inputMovies) #Then merging it so we can get the movieId. It's implicitly merging it by title.

inputMovies = inputMovies.drop('genres', 1).drop('year', 1) #Remove information we won't use from the input dataframe

#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
# print(inputMovies)

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())] #Filtering out the movies from the input
# print(userMovies)

userMovies = userMovies.reset_index(drop=True) #Resetting the index to avoid future issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1) #Dropping unnecessary issues due to save memory and to avoid issues
# print(userGenreTable)

"""
start learning the input's preferences
turn each genre into weights: using the input's reviews and multiplying them into the input's genre table and then summing up the resulting table by column. 
This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.
"""
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating']) # User Profile -> you can recommend movies that satisfy the user's preferences
# print(userProfile)

# extracting the genre table from the original dataframe:
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId']) # Now let's get the genres of every movie in our original dataframe

genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1) #And drop the unnecessary information
genreTable.head()


# With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average 
# of every movie based on the input profile and recommend the top twenty movies that most satisfy it.

#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()

#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()

print("Recommendation table:")
#The final recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())] )
