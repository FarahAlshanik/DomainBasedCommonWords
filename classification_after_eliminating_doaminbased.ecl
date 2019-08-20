//EXPORT final_removeand := 'todo';

//EXPORT kevin_remove := 'todo';
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
 



 CSVRecord3 := RECORD
    
  string term; 
END;


//
 


words := DATASET('~thor::farah::bio_phy_domain_based',
                 CSVRecord3,
                 CSV);


 
 domain_based:=words[1..20000];
 output(domain_based,named('domainbasedwords'));
 
 
 
 CSVRecord1 := RECORD
    
  string text; 
END;

Sentence := Types.Sentence;


corpuss := DATASET('~thor::farah::Biochemical_physiology2',
                 CSVRecord1,
                 CSV);
 
domain_based;

 

//wSID := PROJECT(corpus, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));
 
 
 
 wSIDD := PROJECT(corpuss, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));
 
 
 Sentence filter2(Sentence doc) := TRANSFORM

expr3:='[^A-Za-z0-9]';	
SELF.text:=TRIM(REGEXREPLACE(expr3, doc.text, ' '),LEFT,RIGHT);  
 

SELF := doc;
END;
result2:= PROJECT(wSIDD, filter2(LEFT));
output(result2,named('aftercleaning'));


 
 
 
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
 
 
 /////////////////////////////////////////////////////////////////////////////////////////////////
 
 
 //////////////////////////////////////////////////////////////////////////////////////
 
 
  CSVRecord := RECORD
    
  string text; 
  integer jname;
END;

CSVRecord2 := RECORD
   integer id; 
  string text; 
  integer jname;
  
END;

  

corpus := DATASET('~thor::farah::Biochemical_physiology2',
                 CSVRecord,
                 CSV);
 output(corpus,named('readdataallbefore'));

//wSID := PROJECT(corpus, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));
 
 

corpus_jou_id := PROJECT(corpus, TRANSFORM(CSVRecord2, SELF.id := COUNTER, SELF:=LEFT));
output(corpus_jou_id,named('datawithjournalname'));
 
output(removeDomainTerms);


 
 
 wSID := PROJECT(removeDomainTerms, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));////////////////////////////////////////////////removeDomainTerms
 //wSID_test := PROJECT(testSentences, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));


//wSID;

sentences := DISTRIBUTE(wSID, sentId);
//sentences_test := DISTRIBUTE(wSID_test, sentId);
//sentences;
                       
sv := tv.SentenceVectors(minOccurs := 1,wordNGrams := 1);

//Sv := tv.SentenceVectors(minOccurs := 1, wordNGrams := 1, noProgressEpochs := 5, batchSize := 30);

// Train and return the model, given your set of sentences
model := sv.GetModel(sentences);
//model_test := sv.GetModel(sentences_test);

//model;
//output(model);


sentMod := model(typ = Types.t_ModRecType.sentence);
output(sentMod,named('w2vec'));

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
output(abst_jou,named('corpuswithoutstops'));


TempRec2 := record
        integer typ;
        integer id;
        string text;
        real vec1;
        integer dimension;
end;

//myDataES := SORT(abst_jou, id);

myTrainData := abst_jou[5886..29407];  // Treat first 5000 as training data.  Transform back to the original format.

myTestData := abst_jou[1..5886]; // Treat next 2000 as test data

output(myTrainData,named('traindata'));


output(myTestData,named('testdata'));

output(corpus_jou_id,named('workedcorpus'));



myTrainData_dep := corpus_jou_id[5886..29407];  // Treat first 5000 as training data.  Transform back to the original format.

myTestData_dep := corpus_jou_id[1..5886]; // Treat next 2000 as test data

 

 
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


assessmentCn := ML_Core.Analysis.Classification.Accuracy(predictedClasses, myDepTestDataDF); // Both params are DF dataset
output(assessmentCn,named('Accuracy')); 



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
 
 