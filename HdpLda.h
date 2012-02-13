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
	int m;					// ���̗���������Ă���e�[�u�����̃J�E���g
	int n;					// ���̗�����H�ׂĂ���l���̃J�E���g ��_w{n_w}
	std::vector<int> n_v;	// ���̗�����H�ׂĂ���l��v���Ƃ̃J�E���g
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
	int n;					// ���̃e�[�u���ɒ����Ă���l���̃J�E���g
	//std::unordered_map<int, int> n_v;
	std::vector<int> n_v;	// ���̃e�[�u���ɒ����Ă���l��v���Ƃ̃J�E���g


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
	double gamma;	// G_0 �` DP(��, H) �̃�
	double alpha0;	// G_j �` DP(��_0, G_0) �̃�_0
	double beta;	// H = Dirichlet(��) �̃�: W�����Ώ̃f�B���N�����z��concentration parameter
	int D;
	int V;
	int N;
	int K;
	std::list<int> m_k; // �R�[�p�X�S�̂ŗ���k��H�ׂĂ���l���̃J�E���g
	std::vector<std::list<int>> n_jt; // ����j�̃e�[�u��t�ɍ����Ă���l���̃J�E���g
	std::vector<std::list<std::vector<int>>> n_jtk; // ����j�̃e�[�u��t�ɍ����Ă���l����bv�ł���J�E���g
	std::list<std::vector<int>> phi_kw; // �R�[�p�X�S�̂ŗ���k��H�ׂĂ���l����bw�ł���J�E���g
	std::vector<std::list<int>> psi_jt; // ����j�̃e�[�u��t�ɏ���Ă��闿��k

	
	std::vector<Restaurant> restaurants;
	std::list<std::shared_ptr<Topic>> topics;
	int m;
};



