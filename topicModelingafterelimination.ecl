//EXPORT topicModelingafterelimination := 'todo';

IMPORT Python3;
#option('outputLimit',100);


CSVRecord3 := RECORD
    
  string term; 
END;


//
 


words := DATASET('~thor::farah::cd_stopss',
                 CSVRecord3,
                 CSV);


 
 domain_based:=words[1..250];
 output(domain_based,named('domainbasedwords'));
 
 

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

CSVRecord filter2(CSVRecord doc) := TRANSFORM

expr3:='[^A-Za-z0-9-]';	
SELF.text:=TRIM(REGEXREPLACE(expr3, doc.text, ' '));  
SELF := doc;
END;
 
result2:= PROJECT(corpus, filter2(LEFT));


output(result2,named('aftercleaning'));                
 
 
  wSIDD := PROJECT(corpus, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));


          
                
                 
Import Python3;
dataset(CSVRecord) MatrixMultiplyy(dataset(Sentence) A, dataset(CSVRecord3) db ) := embed(Python3)
import pandas as pd
import re

d=[]
for n in db:
  d.append((n.term))  

def split_data(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]',r'',text)
   # text = re.sub(r'[0-9]','',text)
    text = re.sub(r'\/', '', str(text))
    text = re.sub(r'didn\'t', 'did not', str(text))
    text = re.sub(r'\'s', 'is', str(text))
    text = re.sub(r'n\'t', 'not', str(text))
    text = re.sub(r'\.', '', str(text))
    text = re.sub(r'\(', ' ', str(text))
    text = re.sub(r'\)', ' ', str(text))
    text = re.sub(r'\]', ' ', str(text))
    text = re.sub(r'\[', ' ', str(text))
    text = re.sub(r'can\'t', 'can not', str(text))
    text = re.sub(r'\*', '', str(text))
    text = re.sub(r'\"', '', str(text))
    text = re.sub(r'\'', '', str(text))
    text = re.sub(r'\%', '', str(text))
    text = re.sub(r'\#', '', str(text))
    text = re.sub(r'\$', '', str(text))
    text = re.sub(r'\,', '', str(text))
    text = re.sub(r'\&', '', str(text))
    text = re.sub(r':', '', str(text))
    text = re.sub(r'\@', '', str(text))
    text = re.sub(r'\~', '', str(text))
    text = re.sub(r'\+', '', str(text))
    text = re.sub(r'<', '', str(text))
    text = re.sub(r'>', '', str(text))
    text = re.sub(r';', '', str(text))
    text = re.sub(r',', '', str(text))
    text = re.sub(r'=', '', str(text))
    text = text.strip()
    filtered_words = [word for word in text.split() if word not in d]
    text = " ".join(filtered_words)
    return text   
s=[]
for n in A:
  s.append((n.text))
dfObj = pd.DataFrame(s, columns = ['content']) 
dfObj['content'] = dfObj['content'].map(split_data)
return dfObj['content']
ENDEMBED;




//Second form: a function
//OUTPUT(MatrixMultiply(corpuss,domain_based));


 CSVRecord2 := RECORD
    integer id;
  string text; 
END;

vv:=MatrixMultiplyy(wSIDD,domain_based);
output(vv);

vvv := PROJECT(vv, TRANSFORM(CSVRecord2, SELF.id := COUNTER, SELF := LEFT));
output(vvv,named('vvv'));
 
 
 
 
 
 
 
 
 
 /*
 createRegExForm := project(domain_based, 
                            transform(recordof(left),
					             self.term := '(( |^)' + trim(STD.Str.ToLowerCase(LEFT.term)) +'( |$))'));
createRegExForm;						 
 createSinglexForm := rollup(createRegExForm,
                             true,
					    transform(recordof(left),
					              self.term := left.term + '|' + right.term));
 
 createSinglexForm;
 regEx := createSinglexForm[1].term;
 regEx;
 removeDomainTerms := project(result2,      //insted of result2
                              transform(recordof(left),
						          inSentence    := trim(STD.Str.ToLowerCase(LEFT.text));
								self.text := regexreplace(regEx, inSentence, ' '),
								self          := left));
 
 output(removeDomainTerms,named('aftereliminating'));                
                 
   */              
dataset(rec0) MatrixMultiply(dataset(CSVRecord2) A) := embed(Python3)
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
OUTPUT(MatrixMultiply(vvv));