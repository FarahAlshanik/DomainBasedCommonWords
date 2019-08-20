//EXPORT class_with_dom_2 := 'todo';

#option('outputLimit',1000);

IMPORT TextVectors AS tv;
IMPORT * from tv.Types;
IMPORT STD;
IMPORT tv.internal AS int;
IMPORT int.svUtils AS Utils;
IMPORT ML;
 IMPORT Examples.Sentilyze AS Sentilyze;
IMPORT ML_Core;
IMPORT LearningTrees AS LT;

 CSVRecord := RECORD
    
  string text; 
  integer jname;
END;


 CSVRecord2 := RECORD
   integer id; 
  string text; 
  integer jname;
  
END;

Sentence := Types.Sentence;
//Biochemical_physiology2
corpus := DATASET('~thor::farah::cd',
                 CSVRecord,
                 CSV);
 corpus;

//wSID := PROJECT(corpus, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));
 
 

corpus_jou_id := PROJECT(corpus, TRANSFORM(CSVRecord2, SELF.id := COUNTER, SELF := LEFT));
corpus_jou_id;
 
 
 
 
 wSID := PROJECT(corpus, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));
 //wSID_test := PROJECT(testSentences, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));


//wSID;

sentences := DISTRIBUTE(wSID, sentId);
//sentences_test := DISTRIBUTE(wSID_test, sentId);
//sentences;
                       
sv := tv.SentenceVectors(minOccurs := 1,wordNGrams := 1);

// := tv.SentenceVectors(minOccurs := 1, wordNGrams := 1, noProgressEpochs := 5, batchSize := 30);

// Train and return the model, given your set of sentences
model := sv.GetModel(sentences);
//model_test := sv.GetModel(sentences_test);

//model;
//output(model);


sentMod := model(typ = Types.t_ModRecType.sentence);
sentMod;

//sentMod_test := model_test(typ = Types.t_ModRecType.sentence);
//sentMod_test;
//wordvecs := model(typ = Types.t_ModRecType.word);
//wordvecs;

 

 
TempRec3 := record
	
integer typ;	
  integer id;
	string text;
  set of real vec;
  integer jid;
	
end;


numOfDimensions := 100;

abst_jou    := join(sentMod, corpus_jou_id,
                      left.id = right.id,
				  transform(TempRec3,
				            self := left,
                    self.jid:= right.jname;
                   // self.vec:=right.vec;
						));
abst_jou;


TempRec2 := record
        integer typ;
        integer id;
        string text;
        real vec1;
        integer dimension;
end;

//myDataES := SORT(abst_jou, id);
//5886..29407  train
//1..5886    test
myTrainData := abst_jou[1382..6910];  // Treat first 5000 as training data.  Transform back to the original format.

myTestData := abst_jou[1..1382]; // Treat next 2000 as test data



myTrainData;
myTestData;


corpus_jou_id;



myTrainData_dep := corpus_jou_id[1382..6910];  // Treat first 5000 as training data.  Transform back to the original format.

myTestData_dep := corpus_jou_id[1..1382]; // Treat next 2000 as test data

 
 
/*
normalizeRec := normalize(sentMod,
                          numOfDimensions,
					 transform(TempRec2,
					           self.dimension := counter,
							 self           := left,
							 self.vec1       := left.vec[counter]));
                          
					
sortedDim := group(sort(distribute(normalizeRec, hash32(dimension)), dimension,id,local), dimension, local);				          				         
sortedDim;


ddAverage2    := join(sortedDim, dSentences2,
                      left.id = right.rid,
				  transform(TempRec3,
				            self := left,
                    self.jid:= right.jid;
                   // self.vec:=right.vec;
						));
ddAverage2;



ddAverage    := join(sortedDim, dSentences2,
                      left.id = right.rid,
				  transform(TempRec3,
				            self := left,
                    self.jid:= right.jid;
                   // self.vec:=right.vec;
						));
ddAverage;
/////////////////////////////

ML.Types.NumericField ToIndep(TempRec3 L) := TRANSFORM
	//Takes relevant data from ML.Docs.Trans.Wordbag
	//and converts to numericfield
		SELF.id := L.id;
		SELF.number :=1;
		//Depending on NB Model value is either words_in_doc (term frequency) or 1 (term presence)
		SELF.value := L.vec1;
	END;

  nfIndep := PROJECT(ddAverage2,ToIndep(LEFT));
	dfIndep := ML.Discretize.ByRounding(nfIndep);

nfIndep;




*/


/////////////////////////
//sentMod;


ML_Core.Types.NumericField makeNF(SentInfo sent, UNSIGNED ctr) := TRANSFORM

                SELF.wi := 1;

                SELF.id := ctr;

                SELF.number := ctr;

                SELF.value := sent.vec[ctr];

END;

 

 
 //NORMALIZE(sentMod, makeNF(LEFT, COUNTER));
//nfInddd:= NORMALIZE(sentMod,numOfDimensions, makeNF(LEFT, COUNTER);



 
 
 

 
dfIndepp_train := normalize(myTrainData,numOfDimensions,
                          
					 transform(ML_Core.Types.NumericField,
					           self.wi := 1,
							 self.id           := left.id,
               SELF.number := COUNTER,
							 self.value       := left.vec[counter]));
 
 dfIndepp_train;

//be sure of this 
dfdepp_train := normalize(myTrainData_dep,1,
                          
					 transform(ML_Core.Types.NumericField,
					           self.wi := 1,
							 self.id           := left.id,
               SELF.number := 1,
							 self.value       := left.jname));

dfdepp_train;




dfIndepp_test := normalize(myTestData,numOfDimensions,
                          
					 transform(ML_Core.Types.NumericField,
					           self.wi := 1,
							 self.id           := left.id,
               SELF.number := COUNTER,
							 self.value       := left.vec[counter]));
 
 dfIndepp_test;

//be sure of this 
dfdepp_test := normalize(myTestData_dep,1,
                          
					 transform(ML_Core.Types.NumericField,
					           self.wi := 1,
							 self.id           := left.id,
               SELF.number := 1,
							 self.value       := left.jname));

dfdepp_test;

 
myDepTrainDataDF := ML_Core.Discretize.ByRounding(dfdepp_train);

myDepTestDataDF := ML_Core.Discretize.ByRounding(dfdepp_test);

myLearnerC := LT.ClassificationForest();
myModelC := myLearnerC.GetModel(dfIndepp_train, myDepTrainDataDF); // Notice second param uses the DiscreteField dataset
predictedClasses := myLearnerC.Classify(myModelC, dfIndepp_test);
//assessment:= ML_Core.Analysis.Classification.Accuracy( predictedClasses, myDepTestDatDF); // Both params are DF dataset
//assessmentCC;
output(predictedClasses,named('predictedClassess'));


assessmentC := ML_Core.Analysis.Classification.Accuracy(predictedClasses, myDepTestDataDF); // Both params are DF dataset
assessmentC;

classAcc := ML_Core.Analysis.Classification.AccuracyByClass(predictedClasses, myDepTestDataDF); // Both params are DF dataset
output(classAcc,named('Class_Accuracy')); 

ConfusionMatrixx:=ML_Core.Analysis.Classification.ConfusionMatrix(predictedClasses, myDepTestDataDF);
output(ConfusionMatrixx,named('ConfusionMatrix'));

////////////////////////////////////////
//Regression
 myLearnerR := LT.RegressionForest(); // We use the default configuration parameters.  That usually works fine.
myModelR := myLearnerR.GetModel(dfIndepp_train, dfdepp_train);
predictedDeps := myLearnerR.Predict(myModelR, dfIndepp_test);
assessmentR := ML_Core.Analysis.Regression.Accuracy(predictedDeps, dfdepp_test);
output(assessmentR,named('accuracyofregression'));
 








 /*
dfIndepp_test := normalize(sentMod_test,numOfDimensions,
                          
					 transform(ML_Core.Types.NumericField,
					           self.wi := 1,
							 self.id           := left.id,
               SELF.number := COUNTER,
							 self.value       := left.vec[counter]));
 
 

dfdepp_test := normalize(sentMod_test,numOfDimensions,
                          
					 transform(ML_Core.Types.NumericField,
					           self.wi := 2,
							 self.id           := left.id,
               SELF.number := COUNTER,
							 self.value       := 1));


*/


//dfdep_dis := ML_Core.Discretize.ByRounding(dfdepp);
//dfdep_dis_test := ML_Core.Discretize.ByRounding(dfdepp_test);

//myLearnerC := LT.ClassificationForest();
 
//myModelC := myLearnerC.GetModel(dfIndepp,dfdep_dis);//); // Notice second param uses the DiscreteField dataset

//predictedClasses := myLearnerC.Classify(myModelC, dfIndepp);
//assessmentC := ML_Core.Analysis.Classification.Accuracy(predictedClasses, dfdep_dis); // Both params are DF dataset
//assessmentC;
//predictedClasses;

/*
ML_Core.Types.NumericField makeNF(SentInfo sent, UNSIGNED ctr) := TRANSFORM

                SELF.wi := 1;

                SELF.id := ctr;

                SELF.number := ctr;

                SELF.value := sent.vec[ctr];

END;

 */

 

 //nfInd := NORMALIZE(sentMod, makeNF(LEFT, COUNTER);


//nfInd;












/////////////////////////////////////







/*
ml.ToField(ddAverage,o);
o; //ready now we have vectors dim and the other one 


//ml.ToField(ddAverage,o);
//o;

myIndTrainDataNF := o(number = 1); // Number is the field number  to get the vec values
myIndTrainDataNF;
myDepTrainDataNF := PROJECT(o(number = 2), TRANSFORM(RECORDOF(LEFT), SELF.number := 1, SELF := LEFT));//to get journal name
myDepTrainDataNF;

 
myIndTestData := o(number = 1); //we should split the data to train and test but now for the testing we keep both 

myDepTestData := PROJECT(o(number = 2), TRANSFORM(RECORDOF(LEFT), SELF.number := 1, SELF := LEFT));
//myDepTestData;

 
//myDepTrainDataDFf := ML_Core.Discretize.ByRounding(myDepTrainDataNF);
IMPORT LearningTrees AS LT;


D2 := ML.Discretize.ByRounding(o);
D2(Number=1);
BayesModule := ML.Classify.NaiveBayes;
//TestModule := BayesModule.TestD(myIndTrainDataNF,myDepTrainDataNF);
TestModule := BayesModule.TestD(D2(Number=1),D2(Number=2));//1= vec value 2 is journal number  //naive bayse should be discret the values that it has
//still I have problem with the value of the vectors why I got it as discret it's always 0 why?? is this right
TestModule.Raw;
TestModule.CrossAssignments;
TestModule.PrecisionByClass;
TestModule.Headline;


//ml.ToField(myIndTrainDataNF,myIndTrainDataNFnumeric);
//ml.ToField(myDepTrainDataNF,myDepTrainDataNFnumeric);
myIndTrainDataNF;
myDepTrainDataNF;



ddAverage2;
*/





//output(myIndTrainDataNFnumeric(Number=2));
/*
ML.Types.NumericField ToIndep(ML.Docs.Types.OWordElement L) := TRANSFORM
	//Takes relevant data from ML.Docs.Trans.Wordbag
	//and converts to numericfield
		SELF.id := L.id;
		SELF.number := L.text;
		//Depending on NB Model value is either words_in_doc (term frequency) or 1 (term presence)
		SELF.value := L.vec1;
	END;

  nfIndep := PROJECT(TweetBag,ToIndep(LEFT));
	dfIndep := ML.Discretize.ByRounding(nfIndep);
	

nfIndep;
*/
//myLearnerC := LT.ClassificationForest();
 
//myModelC := myLearnerC.GetModel(myIndTrainDataNF, myDepTrainDataNF);//); // Notice second param uses the DiscreteField dataset
//myIndTrainDataNF;

//o;


/*

IMPORT Sentilyze;

tweetsPositive := DATASET('~SENTILYZE::POSITIVE',Sentilyze.Types.TweetType,CSV);

tweetsNegative := DATASET('~SENTILYZE::NEGATIVE',Sentilyze.Types.TweetType,CSV);

rawPositive := Sentilyze.PreProcess.ConvertToRaw(tweetsPositive);

rawNegative := Sentilyze.PreProcess.ConvertToRaw(tweetsNegative);

processPositive := Sentilyze.PreProcess.RemoveTraining(rawPositive);

processNegative := Sentilyze.PreProcess.RemoveTraining(rawNegative);

positiveWordsTfidf := Sentilyze.KeywordCount.Generate(processPositive,200).Keywords_tfidf;

negativeWordsTfidf := Sentilyze.KeywordCount.Generate(processNegative,200).Keywords_tfidf;

sentimentWordsMI := Sentilyze.KeywordCount.Generate(processPositive,200).Keywords_MI(processNegative);

OUTPUT(positiveWordsTfidf,all,NAMED('PositiveTfidf_Words'));

OUTPUT(negativeWordsTfidf,all,NAMED('NegativeTfidf_Words'));

OUTPUT(sentimentWordsMI,all,NAMED('SentimentMI_Words'));

//////////////////////////
Tweets := DATASET('~SENTILYZE::TWEETS',Sentilyze.Types.TweetType.CSV);

rawTweets := Sentilyze.PreProcess.ConvertToRaw(Tweets);

processTweets := Sentilyze.PreProcess.RemoveAnalysis(rawTweets)

kcSentiment := Sentilyze.KeywordCount.Classify(processTweets);

nbSentiment := Sentilyze.NaiveBayes.Classify(processTweets);

OUTPUT(kcSentiment,NAMED('TwitterSentiment_KeywordCount'));

OUTPUT(nbSentiment,NAMED('TwitterSentiment_NaiveBayes'));


IMPORT Examples.Sentilyze AS Sentilyze;
IMPORT ML;

ML.Docs.Types.LexiconElement AddOne(ML.Docs.Types.LexiconElement L) := TRANSFORM
//Increases word_id value by 1
//This is so the number "1" can be used for the dependent sentiment variable
	SELF.word_id := L.word_id + 1;
	SELF := L;
END;

ML.Types.NumericField ToIndep(ML.Docs.Types.OWordElement L) := TRANSFORM
//Takes relevant data from ML.Docs.Trans.Wordsbag
//and converts to numericfield
	SELF.id := L.id;
	SELF.number := L.word;
	//Depending on NB Model value is either words_in_doc (term frequency) or 1 (term presence)
	SELF.value := L.words_in_doc;
END;

ML.Types.NumericField ToDep(Sentilyze.Types.SentimentType L) := TRANSFORM
// to extract document ids and sentiment values to a numericfield
	SELF.id := L.id;
	SELF.number := 1;
	SELF.value := L.sentiment;
END;

//Pre-Process Training Data
dPosiTrainer := DATASET(Sentilyze.Strings.PositiveTweets,Sentilyze.Types.TweetType,CSV);
dPosiRaw := Sentilyze.PreProcess.ConvertToRaw(dPosiTrainer);
dPosiProcess := Sentilyze.PreProcess.RemoveTraining(dPosiRaw);
//dPosiProcess := Sentilyze.PreProcess.ReplaceTraining(dPosiRaw);
dPosiTagged := PROJECT(dPosiProcess,TRANSFORM(Sentilyze.Types.SentimentType,SELF.id := LEFT.id;SELF.tweet := LEFT.txt;SELF.sentiment := 1));

dNegaTrainer := DATASET(Sentilyze.Strings.NegativeTweets,Sentilyze.Types.TweetType,CSV);
dNegaRaw := Sentilyze.PreProcess.ConvertToRaw(dNegaTrainer);
dNegaProcess := Sentilyze.PreProcess.RemoveTraining(dNegaRaw);
//dNegaProcess := Sentilyze.PreProcess.ReplaceTraining(dNegaRaw);
dNegaTagged := PROJECT(dNegaProcess,TRANSFORM(Sentilyze.Types.SentimentType,SELF.id := LEFT.id;SELF.tweet := LEFT.txt;SELF.sentiment := -1));

SentiMerge := PROJECT((dPosiTagged + dNegaTagged),TRANSFORM(Sentilyze.Types.SentimentType, SELF.id := COUNTER;SELF := LEFT));
SentiRaw := PROJECT(SentiMerge,TRANSFORM(ML.Docs.Types.Raw,SELF.id := LEFT.id;SELF.txt := LEFT.tweet));
SentiWords := ML.Docs.Tokenize.Split(ML.Docs.Tokenize.Clean(SentiRaw));

//Create Vocabulary
Senticon := PROJECT(ML.Docs.Tokenize.Lexicon(SentiWords),AddOne(LEFT));

//Create Wordbags
SentiO1 := ML.Docs.Tokenize.ToO(SentiWords,Senticon);
SentiBag := SORT(ML.Docs.Trans(SentiO1).WordBag,id,word);

//Train Classifier
nfIndep := PROJECT(SentiBag,ToIndep(LEFT));
dfIndep := ML.Discretize.ByRounding(nfIndep);
nfDep := PROJECT(SentiMerge,ToDep(LEFT));
dfDep := ML.Discretize.ByRounding(nfDep);
SentiModel := ML.Classify.NaiveBayes.LearnD(dfIndep,dfDep);

EXPORT Model := MODULE
	EXPORT Vocab := Senticon:PERSIST(Sentilyze.Strings.BayesVocab);
	EXPORT Model := SentiModel:PERSIST(Sentilyze.Strings.BayesModel);
END;
*/