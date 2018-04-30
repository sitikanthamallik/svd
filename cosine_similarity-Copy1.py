import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

#Load ml-100k movie data into pandas with labels.
df = pd.read_csv("c:\\users\\nEW u\\Desktop\\u.data", sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

#Declare number of users and movies.
n_users = df.user_id.unique().shape[0]  #943
n_movies = df.movie_id.unique().shape[0]  #1682
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies)  )

#Creates a train test split of 75/25.
train_data, test_data = train_test_split(df, test_size=0.25)

#Populates a train and test matrix (user_id x movie_id), containing ratings.
train_data_matrix = np.zeros((n_users, n_movies))
for line in train_data.itertuples():
    #[user_id index, movie_id index] = given rating.
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_movies))
for line in test_data.itertuples():
    #[user_id index, movie_id index] = given rating.
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#Creates cosine similarity matrices for users and movies. 
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
movie_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
#Creates euclidean similarity matrices for users and movies. 
user_similarity1 = pairwise_distances(train_data_matrix, metric='euclidean')
movie_similarity1 = pairwise_distances(train_data_matrix.T, metric='euclidean')
#Creates manhattan similarity matrices for users and movies. 
user_similarity2 = pairwise_distances(train_data_matrix, metric='manhattan')
movie_similarity2 = pairwise_distances(train_data_matrix.T, metric='manhattan')
#Creates jaccard similarity matrices for users and movies. 
user_similarity3 = pairwise_distances(train_data_matrix, metric='jaccard')
movie_similarity3 = pairwise_distances(train_data_matrix.T, metric='jaccard')



## cosine Predictions 
#User-Movie Collaborative Filtering, difference from mean rating is a better indicator than absolute rating.
mean_user_rating = train_data_matrix.mean(axis=1)[:, np.newaxis] 
ratings_diff = (train_data_matrix - mean_user_rating) 
user_pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

#Movie-Movie Collaborative Filtering
movie_pred = train_data_matrix.dot(movie_similarity) / np.array([np.abs(movie_similarity).sum(axis=1)])

##euclidean Predictions 
#User-Movie Collaborative Filtering, difference from mean rating is a better indicator than absolute rating.
mean_user_rating1 = train_data_matrix.mean(axis=1)[:, np.newaxis] 
ratings_diff1 = (train_data_matrix - mean_user_rating1) 
user_pred1 = mean_user_rating1 + user_similarity1.dot(ratings_diff1) / np.array([np.abs(user_similarity1).sum(axis=1)]).T

#Movie-Movie Collaborative Filtering
movie_pred1 = train_data_matrix.dot(movie_similarity1) / np.array([np.abs(movie_similarity1).sum(axis=1)])
## manhattan Predictions 
#User-Movie Collaborative Filtering, difference from mean rating is a better indicator than absolute rating.
mean_user_rating2 = train_data_matrix.mean(axis=1)[:,np.newaxis]
ratings_diff2 = (train_data_matrix - mean_user_rating2) 
user_pred2 = mean_user_rating2 + user_similarity2.dot(ratings_diff2) / np.array([np.abs(user_similarity2).sum(axis=1)]).T

#Movie-Movie Collaborative Filtering
movie_pred2 = train_data_matrix.dot(movie_similarity2) / np.array([np.abs(movie_similarity2).sum(axis=1)])

##jaccard Predictions 
#User-Movie Collaborative Filtering, difference from mean rating is a better indicator than absolute rating.
mean_user_rating3 = train_data_matrix.mean(axis=1)[:, np.newaxis] 
ratings_diff3 = (train_data_matrix - mean_user_rating3) 
user_pred3 = mean_user_rating3 + user_similarity3.dot(ratings_diff3) / np.array([np.abs(user_similarity3).sum(axis=1)]).T

#Movie-Movie Collaborative Filtering
movie_pred3 = train_data_matrix.dot(movie_similarity3) / np.array([np.abs(movie_similarity3).sum(axis=1)])


#Root Mean Squared Error for validation.
def rmse(pred, test):
    pred = pred[test.nonzero()].flatten() 
    test = test[test.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, test))

print ("Collaborative Filtering RMSE(cosine based)")
print ('User-based: ', rmse(user_pred, test_data_matrix) )  # ~3.12584229228
print ('Movie-based: ', rmse(movie_pred, test_data_matrix))	  # ~3.45381500808

print ("Collaborative Filtering RMSE(euclidean based)")
print ('User-based: ', rmse(user_pred1, test_data_matrix) )  # ~3.12584229228
print ('Movie-based: ', rmse(movie_pred1, test_data_matrix))	  # ~3.45381500808

print ("Collaborative Filtering RMSE(manhattan based)")
print ('User-based: ', rmse(user_pred2, test_data_matrix) )  # ~3.12584229228
print ('Movie-based: ', rmse(movie_pred2, test_data_matrix))	  # ~3.45381500808

print ("Collaborative Filtering RMSE(jaccard based)")
print ('User-based: ', rmse(user_pred3, test_data_matrix) )  # ~3.12584229228
print ('Movie-based: ', rmse(movie_pred3, test_data_matrix))	  # ~3.45381500808


