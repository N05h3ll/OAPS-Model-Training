import pandas as pd
import nltk
import fasttext as ft
import fasttext.util as ftu
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
data = pd.read_csv('quora.csv', sep='\t', low_memory=False)
def textProcessing(questions) :
    pureQuestions = []
    stop_words = nltk.corpus.stopwords.words("english")
    for q in questions :
        words = nltk.word_tokenize(str(q))
        pureSentence = []
        for w in words:
            if w.isalpha() and w not in stop_words:
                pureSentence.append(lemma.lemmatize(w).lower())
        pureQuestions.append(' '.join(pureSentence))
        pureSentence.clear()
    return pureQuestions

question1 = textProcessing(data['question1'].head(10000))
question2 = textProcessing(data['question2'].head(10000))
isDuplicate = data['is_duplicate'].head(10000)
model = ft.load_model('cc.en.300.bin')
ftu.reduce_model(model, 100)
trainQuestions=[]
trainSimilar=[]
for index,q in enumerate(question1, start=0):
    v1 = model.get_sentence_vector(q)
    v2 = model.get_sentence_vector(question2[index])
    trainQuestions.append([v1,v2])
    trainSimilar.append(isDuplicate[index])

print('Training')
DfTrainQuestions = pd.DataFrame(trainQuestions)
DfSimilar = pd.DataFrame(trainSimilar)
DfTrainQuestions.to_csv('trainQuestions.csv', escapechar='\\', sep=',', index=False, header=['q1','q2'])
DfSimilar.to_csv('trainSimilar.csv', escapechar='\\', sep=',', index=False, header=['sim'])