import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

input_dir = "/home/anhoang/Basic_DL/cv/scikit-learn/image_classification/clf-data"

categories = ['empty', 'not_empty']

data =[]
labels = []

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)

        img = resize(img , (15, 15)) # Resize to 15x15 image
        data.append(img.flatten()) # Convert image into array
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)


best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print(score)

pickle.dump(best_estimator, open('./model.p', 'wb'))