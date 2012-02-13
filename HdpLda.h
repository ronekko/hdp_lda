#pragma once

#include <vector>
#include <list>
#include <memory>
#include <boost/random.hpp>
#include "Corpus.hpp"


class Topic
{
public:
	Topic(): V(0), beta(0.0), m(0), n(0), n_v(V), phi_v(V){};
	Topic(const int &V, const double &beta):V(V), beta(beta), m(0), n(0), n_v(V), phi_v(V){};
	Topic(const Topic &t): V(t.V), beta(t.beta), m(t.m), n(t.n), n_v(V), phi_v(V){};
	Topic & operator =(const Topic &t){
		V = t.V;
		beta = t.beta;
		m = t.m;
		n = t.n;
		n_v = t.n_v;
		phi_v = t.phi_v;
		return *this;
	}
	int V;
	double beta;
	int m;					// この料理が乗っているテーブル数のカウント
	int n;					// この料理を食べている人数のカウント Σ_w{n_w}
	std::vector<int> n_v;	// この料理を食べている人のvごとのカウント
	std::vector<double> phi_v;

};

class Table
{
public:
	Table(void): topic(), n(0), n_v(0){};
	Table(shared_ptr<Topic> &topic): topic(topic), n(0), n_v(topic->V, 0){};
	Table(const Table &t): topic(t.topic), n(t.n), n_v(t.n_v){};
	Table & operator =(const Table &t){
		topic = t.topic;
		n = t.n;
		int size = t.n_v.size();
		this->n_v.resize(size);
		for(int i=0; i<t.n_v.size(); ++i){
			n_v[i] = t.n_v[i];
		}
		return *this;
	}

	shared_ptr<Topic> topic;
	int n;					// このテーブルに着いている人数のカウント
	//std::unordered_map<int, int> n_v;
	std::vector<int> n_v;	// このテーブルに着いている人のvごとのカウント


};



class Customer
{
public:
	Customer(): word(0), table(){};
	Customer(shared_ptr<Table> &table): word(0), table(table){};
	Customer(const Customer &c): word(c.word), table(c.table){};
	Customer & operator =(const Customer &c){
		word = c.word;
		table = c.table;
		return *this;
	}
	int word;
	shared_ptr<Table> table;
};

class Restaurant
{
public:
	Restaurant(void): n(0), tables(), m(0){};
	std::vector<Customer> customers;
	int n;
	std::list<std::shared_ptr<Table>> tables;
	int m;


};


//class Franchise
//{
//	std::vector<Restaurant> restaurants;
//	std::list<Topic> topics;
//};


class HdpLda
{
public:
	HdpLda(void){};
	~HdpLda(void){};
	HdpLda(const Corpus &corpus, const Vocabulary &vocabulary, const unsigned long seed
		  , const double &gamma, const double &alpha0, const double &beta, const int &K = 3);
	void sampleTables();
	void sampleTopics();
	vector<vector<double>> calcPhi(void);
	vector<vector<double>> calcTheta(void);
	double calcPerplexity(void);

	Corpus corpus;
	Vocabulary vocabulary;
	boost::random::mt19937 engine;
	double gamma;	// G_0 ～ DP(γ, H) のγ
	double alpha0;	// G_j ～ DP(α_0, G_0) のα_0
	double beta;	// H = Dirichlet(β) のβ: W次元対称ディリクレ分布のconcentration parameter
	int D;
	int V;
	int N;
	int K;
	std::list<int> m_k; // コーパス全体で料理kを食べている人数のカウント
	std::vector<std::list<int>> n_jt; // 文書jのテーブルtに座っている人数のカウント
	std::vector<std::list<std::vector<int>>> n_jtk; // 文書jのテーブルtに座っている人が語彙vであるカウント
	std::list<std::vector<int>> phi_kw; // コーパス全体で料理kを食べている人が語彙wであるカウント
	std::vector<std::list<int>> psi_jt; // 文書jのテーブルtに乗っている料理k

	
	std::vector<Restaurant> restaurants;
	std::list<std::shared_ptr<Topic>> topics;
	int m;
};



