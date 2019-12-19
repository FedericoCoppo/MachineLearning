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
