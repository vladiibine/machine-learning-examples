from __future__ import print_function
from sklearn import svm

data = [
 # I manually made this data. 
 # We have basically 2 classes of input data
 # For the first class:
 # 1st number is in range [0..3], second [0, 4..6], third [0, 7..9]
 [1, 4, 7],
 [2, 0, 7],
 [0, 4, 7],
 [2, 5, 7],
 [0, 0, 7],
 [0, 0, 7],
 [3, 0, 8],
 [0, 5, 8],
 [1, 0, 8],
 [0, 0, 8],
 [0, 4, 8],
 [2, 0, 8],
 [0, 0, 8],
 [0, 6, 9],
 [0, 6, 9],
 [2, 4, 9],
 [1, 5, 9],
 [0, 4, 9],
 [3, 4, 8],
 [0, 6, 8],
 [0, 4, 8],
 [1, 6, 8],
 [2, 5, 8],
 [0, 5, 9],
 [0, 4, 9],
 [1, 6, 9],
 [0, 4, 9],
 [1, 4, 0],
 [1, 6, 0],
 [3, 5, 0],
 [0, 5, 0],
 [1, 6, 7],
 [0, 5, 7],
 [0, 6, 7],
 [0, 0, 0],
 # Second class:
 # 1st is in range [7..9], second [4..6], third [0..3]
 [7, 4, 0],
 [7, 5, 0],
 [8, 5, 0],
 [8, 6, 1],
 [8, 6, 2],
 [7, 5, 3],
 [8, 4, 2],
 [9, 5, 0],
 [7, 5, 0],
 [8, 6, 1],
 [8, 6, 2],
 [9, 4, 0],
 [8, 5, 2],
 [7, 4, 3],
 [8, 5, 1],
 [9, 6, 2],
 [7, 5, 2],
 [8, 4, 2],
 [9, 5, 2],
 [8, 4, 1],
 [8, 6, 1],
 [9, 5, 3],
 [7, 5, 2],
 [8, 4, 2],
 [7, 5, 2],
 [8, 6, 0],
 [8, 4, 0],
 [9, 5, 2],
 [8, 5, 2],
 [7, 6, 0]]


# The labels to apply to these classes
labels = [0] * 35 + [1] * 30

estimator = svm.SVC()

estimator.fit(data, labels)

result = estimator.predict([7, 7, 1])
print('\n\n')
print('Result: ', result[0])
print('Expected result: 1')
