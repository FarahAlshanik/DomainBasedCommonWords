
IMPORT Python3;
#option('outputLimit',100);


 

rec0 := RECORD
 set of unicode cell;
END;

rec := RECORD
DATASET(rec0) arow;
END;



IMPORT TextVectors AS tv;
IMPORT * from tv.Types;
IMPORT STD;
IMPORT tv.internal AS int;
IMPORT int.svUtils AS Utils;

 
 CSVRecord := RECORD
    
  string text; 
END;

Sentence := Types.Sentence;


corpus := DATASET('~thor::farah::cd',
                 CSVRecord,
                 CSV);
dataset(rec0) MatrixMultiply(dataset(CSVRecord) A) := embed(Python3)
# Python code that returns one more than the value passed to it
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pandas as pd
import nltk
from string import punctuation
#nltk.download()

def split_data(line):
  BOW_ = [line.split()]
  BOW = []
  for token in BOW_:
    BOW += token
    BOW = list(filter(lambda w: w not in punctuation, BOW))  
    filtered_words = list(map(lambda word: word.lower(), BOW))

  return filtered_words
s=[]
for n in A:
  s.append((n.text))
dfObj = pd.DataFrame(s, columns = ['content']) 
dfObj['tokens'] = dfObj['content'].map(split_data)
cvectorizer = CountVectorizer(min_df=4, analyzer='word', tokenizer=split_data,stop_words='english')
cvz = cvectorizer.fit_transform(list(dfObj['tokens'].map(lambda tokens: ' '.join(tokens))))
lda_model = LDA()
X_topics = lda_model.fit_transform(cvz)
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(list(dfObj['tokens'].map(lambda tokens: ' '.join(tokens))))  

#n_top_words = 12
#topic_summaries = []
m=[]
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        m.append([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return m
print("Topics found via LDA:")
#m.append(print_topics(lda_model, cvectorizer, 12))
#topic_word = lda_model.topic_word_  
#vocab = cvectorizer.get_feature_names()
#for i, topic_dist in enumerate(topic_word):
 #   topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
  #  topic_summaries.append(' '.join(topic_words))
   # print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    #m.append(topic_words)
#return dfObj['tokens']  #work
#s=print_topics(lda_model, cvectorizer, 20)

number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics)
lda.fit(count_data)
# Print the topics found by the LDA model
#print("Topics found via LDA:")
s=print_topics(lda, count_vectorizer, number_words)
return s
ENDEMBED;

//Second form: a function
OUTPUT(MatrixMultiply(corpus));