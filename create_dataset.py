import numpy as np
from sklearn.preprocessing import StandardScaler

N = 943  # users
M = 1682  # movies

rating_dataset = np.empty((N, M))  # the part of the dataset with the ratings
onehot_dataset = np.eye(N)  # the part of the dataset with the one hot encoding
dataset = np.empty((N, N + M))  # the whole dataset

# import data from file
data = np.genfromtxt('u.data', delimiter='\t', dtype='int')

# fill the rating dataset
rating_dataset[:] = np.nan
for record in data:  # for every rating record in data, add a value to a cell in the dataset array
    user = record[0]  # get the user id
    movie = record[1]  # get the movie id
    rating = record[2]  # get the rating
    rating_dataset[user - 1, movie - 1] = rating  # add the rating in the cell of the array with indices: (user-1,movie-1)

# fill the missing values
for i in range(len(rating_dataset)):  # for every rating vector
    avg_rating = np.nanmean(rating_dataset[i])  # calculate average of rating vector and ignore missing values
    rating_dataset[i][np.isnan(rating_dataset[i])] = avg_rating  # fill missing values with average of rating vector

# preprocess data
sc = StandardScaler()
rating_dataset = np.transpose(sc.fit_transform(np.transpose(rating_dataset)))
# we used the scaler on the transposed dataset (movieXuser and not userXmovie) because the exercise instructs us to do centering
# on the user vector and not the movie vector, and the scaler performs vertically (per feature) and not horizontally (per record), so we had to transpose it

# create the final dataset
dataset = np.concatenate((onehot_dataset, rating_dataset), 1)

# save dataset in file
np.savetxt('dataset.data', dataset, delimiter='\t')
