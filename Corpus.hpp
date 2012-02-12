#pragma once
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <boost/timer.hpp>

using namespace std;

class Token
{
public:
	int x;
	int z;

	Token(int _x) : x(_x){}
	Token(int _x, int _z) : x(_x), z(_z){}
	Token(const Token &t){
		x = t.x;
		z = t.z;
	}
};

class Document
{
public:
	vector<Token> tokens;
	vector<int> topicCount;

	Document(){}
	Document(const Document &d){
		tokens = d.tokens;
		topicCount = d.topicCount;
	}
};

class Corpus
{
public:
	int D; // ドキュメント数
	int V; // 語彙数
	int N; // トークン数
	vector<Document> documents;

	Corpus(){}

	Corpus(const string &corpusFileName){
		ifstream ifs(corpusFileName.c_str());

		if(!ifs){
			cout << "file not found" << endl;
			exit(-1);
		}

		string buf;
		getline(ifs, buf);
		D = atoi(buf.c_str());
		getline(ifs, buf);
		V = atoi(buf.c_str());
		getline(ifs, buf);
		N = atoi(buf.c_str()); // 不要（ここの値はBoWファイルの行数であってトークン数ではないから）

		int docId = 0;
		while(getline(ifs, buf)){
			int d;
			int v;
			int count;
			istringstream iss(buf);
			iss >> d >> v >> count;
			v -= 1;

			if(d > docId){
				documents.push_back(Document());
				docId++;
//				cout << docId << endl;
				cout << "*";
			}

			for(int i=0; i<count; ++i){
				documents[d-1].tokens.push_back(Token(v));
			}
		}

		int count = 0;
		for(int i=0; i<documents.size(); ++i){
			count += documents[i].tokens.size();
		}
		cout << endl;
		N = count;
		cout << "N: " << N << ", D: " << D << ", V: " << V << endl;
	}
		
	Corpus(const Corpus &corpus){
		D = corpus.D;
		V = corpus.V;
		N = corpus.N;
		documents = corpus.documents;
	}
};


class Word
{
public:
	string str;
	vector<int> topicCount;
	Word(const string &_str):str(_str){}
	Word(const Word &word){
		str = word.str;
		topicCount = word.topicCount;
	}
};


class Vocabulary
{
public:
	vector<Word> words;
	
	Vocabulary(){}
	Vocabulary(const string &vocabFileName, const int &topicNum)
	{
		ifstream ifs(vocabFileName.c_str());

		if(!ifs){
			cout << "file not found" << endl;
			exit(-1);
		}

		string buf;
		while(ifs >> buf){
			words.push_back(Word(buf));
		}
	}

	Vocabulary(const Vocabulary &vocabulary){
		words = vocabulary.words;
	}
};