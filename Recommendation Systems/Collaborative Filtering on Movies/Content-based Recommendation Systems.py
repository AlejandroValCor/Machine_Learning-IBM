# Importing needed packages

# await removed due to only use inside async function error
# Acquiring data
!wget -O moviedataset.zip https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%205/data/moviedataset.zip
print('unziping ...')
!unzip -o -j moviedataset.zip 

# First, let's get all of the imports out of the way:
#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Now let's read each file into their Dataframes:
#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

# Let's also take a peek at how each of them are organized:
#Head is a function that gets the first N rows of a dataframe. N's default is 5.
movies_df.head()

# So each movie has a unique ID, a title with its release year along with it (Which may contain unicode characters) and several different genres
# in the same field. Let's remove the year from the title column and place it into its own one by using the handy extract function that Pandas
# has.
# Let's remove the year from the title column by using pandas' replace function and store it in a new year column.
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# Let's look at the result!
movies_df.head()

# With that, let's also drop the genres column since we won't need it for this particular recommendation system.
#Dropping the genres column
movies_df = movies_df.drop('genres', 1)

# Here's the final movies dataframe:
movies_df.head()

# Next, let's look at the ratings dataframe.
ratings_df.head()

# Every row in the ratings dataframe has a user id associated with at least one movie, a rating and a timestamp showing when they reviewed it.
# We won't be needing the timestamp column, so let's drop it to save on memory.
#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)

# Here's how the final ratings Dataframe looks like:
ratings_df.head()

# Collaborative Filtering
# The process for creating a User Based recommendation system is as follows:

# Select a user with the movies the user has watched
# Based on his rating of the movies, find the top X neighbours
# Get the watched movie record of the user for each neighbour
# Calculate a similarity score using some formula
# Recommend the items with the highest score
# Let's begin by creating an input user to recommend movies to:

# Notice: To add more movies, simply increase the amount of elements in the userInput. Feel free to add more in! Just be sure to write it in with capital letters and if a movie starts with a "The", like "The Matrix" then write it in like this: 'Matrix, The' .
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies

# Add movieId to input user
# With the input complete, let's extract the input movies's ID's from the movies dataframe and add them into it.

# We can achieve this by first filtering out the rows that contain the input movies' title and then merging this subset with the input
# dataframe. We also drop unnecessary columns for the input to save memory space.
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies

# The users who has seen the same movies
# Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.
#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()

# We now group up the rows by user ID.
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

# Let's look at one of the users, e.g. the one with userID=1130.
userSubsetGroup.get_group(1130)

# Let's also sort these groups so the users that share the most movies in common with the input have higher priority. This provides a richer
# recommendation since we won't go through every single user.
# Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

# Now let's look at the first user.
userSubsetGroup[0:3]

# Similarity of users to input user
# We're going to find out how similar each user is to the input through the Pearson Correlation Coefficient.
# We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every
# single user.

# Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and
# the value is the coefficient.
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonCorrelationDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()

# The top x similar users to input user
# Now let's get the top 50 users that are most similar to the input.
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()

# Now, let's start recommending movies to the input user.

# Rating of selected users to all movies
# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do
# this, we first need to get the movies watched by the users in our pearsonDF from the ratings dataframe and then store their correlation in a
# new column called _similarityIndex". This is achieved below by merging of these two tables.
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()

# Now all we need to do is simply multiply the movie rating by its weight (the similarity index), then sum up the new ratings and divide it by
# the sum of the weights.

# We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

# It shows the idea of all similar users to candidate movies for the input user:
#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

# Now let's sort it and see the top 20 movies that the algorithm recommended!
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

# Advantages and Disadvantages of Collaborative Filtering

# Advantages
# Takes other user's ratings into consideration
# Doesn't need to study or extract information from the recommended item
# Adapts to the user's interests which might change over time

# Disadvantages
# Approximation function can be slow
# There might be a low amount of users to approximate
# Privacy issues when trying to learn the user's preferences