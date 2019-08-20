 

IMPORT TextVectors AS tv;
IMPORT * from tv.Types;
IMPORT STD;
IMPORT tv.internal AS int;
IMPORT int.svUtils AS Utils;
#option('outputLimit',1000);

 CSVRecord := RECORD
  string text; 
END;

Sentence := Types.Sentence;

corpus := DATASET('~thor::farah::cell_dentist',
                 CSVRecord,
                 CSV);
 

wSID := PROJECT(corpus, TRANSFORM(Sentence, SELF.sentId := COUNTER, SELF := LEFT));


sentences := DISTRIBUTE(wSID, sentId);
                       
sv := tv.SentenceVectors(minOccurs := 1,wordNGrams := 1);
//sv := tv.SentenceVectors(minOccurs := 1,wordNGrams := 1,noProgressEpochs := 5, batchSize := 30);

// Train and return the model, given your set of sentences
model := sv.GetModel(sentences);
 
wordvecs := model(typ = Types.t_ModRecType.word);

wordvecs;



TempRec := record
  integer typ;
  integer id;
  string text;
  set of real vec;
end;

TempRec2 := record
	integer typ;
  integer id;
  string text;
  real vec1;
  integer dimension;
end;

TempRec3 := record
	integer typ;
	integer id;
	string text;
	real vec1;
	real aver_vec;
	real distance;
	integer dimension;
end;
 
numOfDimensions := 100;

normalizeRec := normalize(wordvecs,
                          numOfDimensions,
					 transform(TempRec2,
					           self.dimension := counter,
							 self           := left,
							 self.vec1       := left.vec[counter]));
                          
					
sortedDim := group(sort(distribute(normalizeRec, hash32(dimension)), dimension,id,local), dimension, local);				          				         

sumDimensions := rollup(sortedDim,true, 
                        transform(TempRec2,
				              self.vec1 := left.vec1 + right.vec1,
						    self     := right));

aveDimensions := project(sort(ungroup(sumDimensions), dimension),
                         transform(TempRec2,
					          self.vec1 := left.vec1 / left.id,
							self     := left));

addAverage    := join(normalizeRec, aveDimensions,
                      left.dimension = right.dimension,
				  transform(TempRec3,
				            self := left,
						  self.distance := ((left.vec1 - right.vec1) * (left.vec1 - right.vec1));
						  self.aver_vec := right.vec1));

sortAddAverage := sort(addAverage, id);
calcDistSum    := rollup(sortAddAverage,
                         left.id = right.id,
					transform(TempRec3,
					          self.distance := left.distance + right.distance,
							self          := left));
					
getSqrt       := project(calcDistSum,
                         transform(TempRec3,
					          self.distance := sqrt(left.distance),
							self          := left));
              

result:=sort(getSqrt, distance);
output(result);						

