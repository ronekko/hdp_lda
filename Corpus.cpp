#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/timer.hpp>
#include "Corpus.h"

Corpus::Corpus(const string &corpusFileName){
	using namespace std;

	ifstream ifs(corpusFileName.c_str());

	if(!ifs){
		cout << "file not found" << endl;
		exit(-1);
	}

	string buf;
	getline(ifs, buf);
	D = atoi(buf.c_str());
	getline(ifs, buf);
	W = atoi(buf.c_str());
	getline(ifs, buf);
	N = atoi(buf.c_str()); // BoWファイルの行数 (document-wordペアの個数)

//	int docId = 0; // ドキュメントIDが1始まりの場合はこっち
	int docId = -1;  // ドキュメントIDが0始まりの場合はこっち
	while(getline(ifs, buf)){
		int d;
		int w;
		int frequency;
		istringstream iss(buf);
		iss >> d >> w >> frequency;
//		w -= 1; // 単語IDが0始まりの場合はこの行をコメントアウトする

		if(d > docId){
			documents.push_back(Document());
			docId++;
//				cout << docId << endl;
			cout << "*";
		}

//		documents[d-1].words.push_back(Word(w, frequency)); // ドキュメントIDが1始まりの場合はこっち
		documents[d].tokens.push_back(Token(w, frequency)); // ドキュメントIDが0始まりの場合はこっち
	}

	int count = 0;
	for(int j=0; j<documents.size(); ++j){
		vector<Token> &tokens = documents[j].tokens;
		for(int i=0; i<tokens.size(); ++i){
			count += tokens[i].frequency;
		}
	}
	N = count;
	cout << endl;
	cout << "N: " << N << ", D: " << D << ", W: " << W << endl;
	cout << count << " tokens in the corpus." << endl;
}


Vocabulary::Vocabulary(const string &vocabFileName, const int &topicNum)
{
	std::ifstream ifs(vocabFileName.c_str());

	if(!ifs){
		cout << "file not found" << endl;
		exit(-1);
	}

	string buf;
	while(ifs >> buf){
		types.push_back(WordType(buf));
	}
}
