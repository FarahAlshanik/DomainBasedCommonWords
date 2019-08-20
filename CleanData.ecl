import std;
CSVRecord := RECORD
  string abstract;
  string journalName;
  
END;

 corpus := DATASET('~thor::farah::cell_dentist_plant',
                 CSVrecord,
                 CSV);
corpus;

CSVRecord filter(corpus doc) := TRANSFORM

SELF.abstract:=STD.STr.ToLowerCase(doc.abstract); //return abstract
SELF.journalName:=STD.STr.ToLowerCase(doc.journalName);//return journalName

SELF := doc;
END;
result:= PROJECT(corpus, filter(LEFT));
output(result);


CSVRecord filter2(corpus doc) := TRANSFORM

expr3:='[^A-Za-z0-9-]';	
SELF.abstract:=TRIM(REGEXREPLACE(expr3, doc.abstract, ' '));  
SELF.journalName:=STD.STr.ToLowerCase(doc.journalName); 

SELF := doc;
END;
result2:= PROJECT(result, filter2(LEFT));
output(result2);


