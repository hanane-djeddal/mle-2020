from content_based_filtering.helpers.movies import *
from content_based_filtering.helpers.users import *
from content_based_filtering.helpers.scalability_tools import *
import numpy as np
import pandas as pd


def get_movies_similarity_matrix(movies_genre):
    """
    movies_genre : a pandas dataframe containing for each movie its genre
    
    Returns the matrix of similarities between movies
    """
    return movies_genre.values.dot(movies_genre.values.T)

def get_users_similarity_matrix(users_info):
    """
    users_info : a pandas dataframe containing for each user, the features needed for similarity computation
    
    Returns a similarity Matrix 
    Similarity is computed as the sum of absolute value of the differences between each feature
    
    """
    entry_matrix=users_info.values
    similarity_matrix=np.empty([entry_matrix.shape[0],entry_matrix.shape[0]])
    for i in range(entry_matrix.shape[0]):
        for j in range(entry_matrix.shape[0]):
            similarity_matrix[i][j]= np.sum(abs(entry_matrix[i]-entry_matrix[j]))
            
    return similarity_matrix

def get_most_similar_movies(movies_similarity, movies, movie_name, year=None, top=10):
    """
    Returns top similar movies to the movie given by 'movie_name' and 'year'
    """
    index_movie = get_movie_id(movies, movie_name, year)  
    best = movies_similarity[index_movie].argsort()[::-1]
    return [(ind, get_movie_name(movies, ind), movies_similarity[index_movie, ind]) for ind in best[:top] if ind != index_movie]

def get_most_similar_users(user_similarity,users,user_id,top=10):
    """
    Returns top similar users to the user having user_id=user_id
    """
    index_user=get_user_index(users,user_id)
    best = (-1*user_similarity[index_user]).argsort()[::-1]
    return [(ind, get_user_ID(users, ind), user_similarity[index_user, ind]) for ind in best[:top] if ind != index_user]
    

def get_content_based_recommendations(dataframe,movies,movies_similarity,user_id,rating_col='rating',user_id_col='user_id',
                                      movie_id_col='movie_id',top=10,nb_recommendations=5):
    """
    dataframe: pandas dataframe containing information about users, their ratings and the movies they rate
    movies : dataframe reduced to the columns about the movies
    movies_similarity : movies similarity matrix
    user_id : user ID 
    user_id_col, movie_id_col : names of the IDs columns
    top: number of rated movies to coonsider for the recommendation
    nb_recommendations: number of recommended movies
    
    Returns top recommendations for the user having the ID 'user_id' when considering the top rated movies they have watched.
    """
    top_movies = dataframe[dataframe[user_id_col] == user_id].sort_values(by=rating_col,ascending=False).head(top)[movie_id_col]
    index=['movie_id', 'title', 'similarity']

    most_similars = []
    for top_movie in top_movies:
        most_similars += get_most_similar(movies_similarity, get_movie_name(movies, top_movie), get_movie_year(movies, top_movie))

    return pd.DataFrame(most_similars, columns=index).drop_duplicates().sort_values(by='similarity', 
                                                                                    ascending=False).head(nb_recommendations)

def get_collaborative_recommendations(dataframe,movies,users,users_similarity,user_id,rating_col='rating',user_id_col='user_id',
                        movie_id_col='movie_id',top=5,nb_recommendations=5):
    """
    dataframe: pandas dataframe containing information about users, their ratings and the movies they rate
    movies : dataframe reduced to the columns about the movies
    users: dataframe reduced to the columns about the users
    users_similarity : users similarity matrix
    user_id : user ID 
    user_id_col, movie_id_col : names of the IDs columns
    top: number of similar users to coonsider for the recommendation
    nb_recommendations: number of recommended movies
    
    Returns top recommendations for the user having the ID 'user_id' when considering the top rated movies they have watched.
    """
    most_similar_users = get_most_similar_users(users_similarity,users,user_id,5)
    top_movies=[]
    for top_users in  most_similar_users:
        user_id=top_users[1]
        if(top_users != np.nan):
            movies_by_user=dataframe[dataframe[user_id_col] == user_id].sort_values(by=rating_col,
                                                                          ascending=False).head(top)[movie_id_col]
            for movie in movies_by_user:
                top_movies.append((movie,top_users[2]))
    
    index=['movie_id', 'title', 'similarity']

    most_similars = []
    for top_movie,similarity_value in top_movies:
        most_similars.append( (top_movie,get_movie_name(movies, top_movie), similarity_value))
    

    return pd.DataFrame(most_similars, columns=index).drop_duplicates().sort_values(by='similarity', 
                                                                                    ascending=True).head(nb_recommendations)


