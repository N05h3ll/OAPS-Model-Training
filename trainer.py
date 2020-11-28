import pandas as pd
from sklearn.svm import SVC
import joblib
from sklearn import metrics
print('Reading Data ...')
testQuestion = pd.read_csv('testQuestions.csv', sep=',', escapechar='\\', low_memory=False)
testSim = pd.read_csv('testSimilar.csv', sep=',', escapechar='\\', low_memory=False)
trainQuestions = pd.read_csv('trainQuestions.csv', sep=',', escapechar='\\', low_memory=False)
trainSim = pd.read_csv('trainSimilar.csv', sep=',', escapechar='\\', low_memory=False)
svclassifier = SVC(kernel='linear')
print('Training ...')
svclassifier.fit(trainQuestions, trainSim)
print('Predict ...')
predeciton = svclassifier.predict(testQuestion)
print("Accuracy:",metrics.accuracy_score(testSim, predeciton))
