def similarity_score (similarities):
    return np.mean(similarities)
    
def recommendation_score(recommended_movies, movies,weights=np.array([]), norm=False, max_weight=5, min_weight=0):
    """
    recommended_movies : numpy 1D vector of IDs of the recommended movies 
    movies : numpy 1D  vector of IDs of the movies watched by the user (random or with high rating)
    weights : numpy 1D  vector of ratings corresponding the IDS of  the 'movies' vector
    
    if weights == None : 
    returns the ratio of the number of common movies in 'recommended_movies' and 'movies' 
    to the size of the 'movies' array 
    
    if weights != None :
    returns the sum of the ratings of common movies 
    If norm : We scale the ratings so that max_weight -> 1 and min_weight->0 (max_weight and min_weight) should be provided
    """
    common,r_indices,m_indices=np.intersect1d(recommended_movies, movies,return_indices=True)
    if weights.size == 0:
        return (np.size(common)/ np.size(movies))
    else :
        if(norm):
            weights= (weights - min_weight) /(max_weight - min_weight)
        return (sum([weights[i] for i in m_indices]))/np.size(movies)
        
        
    
        
    
