#pragma once

#include <vector>
#include <list>
#include <memory>
#include <boost/random.hpp>
#include <unordered_map>
#include "Corpus.hpp"

namespace hdplda{

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

	class HdpLda
	{
	public:
		HdpLda(void){};
		~HdpLda(void){};
		HdpLda(const Corpus &corpus, const Vocabulary &vocabulary, const unsigned long seed
			  , const double &gamma, const double &alpha0, const double &beta
			  , const double &gamma_a = 0.01, const double &gamma_b = 0.1
			  , const double &alpha0_a = 10.0, const double &alpha0_b = 1.0, const int &K = 3);
		void sampling();
		void sampleTables();
		void sampleTopics();
		void sampleGamma(void);
		void sampleAlpha0(const int &iter = 20);
		double betaRandom(const double &alpha, const double &beta);
		vector<vector<double>> calcPhi(void);	// Φ[k][v], トピックkの単語比率V次元多項分布
		vector<vector<double>> calcTheta(void);	// Θ[j][k], 文書jのトピック比率K次元多項分布
		double calcPerplexity(const vector<vector<double>> &phi, const vector<vector<double>> &theta);
		void savePhi(const vector<vector<double>> &phi, const string &fileName);
		void saveTheta(const vector<vector<double>> &theta, const string &fileName);
		void savePhiTheta(const vector<vector<double>> &phi, const string &phiFileName,
						const vector<vector<double>> &theta, const string &thetaFileName);
		inline double calcLogNPlusBeta(int n);
		inline double calcLogNPlusVBeta(int n);
		void showAllCounts(void);
		void showAllParameters(void);


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
		double gamma_a; // γの事前分布であるガンマ分布のshapeパラメータ(k)
		double gamma_b; // γの事前分布であるガンマ分布のrateパラメータ(1/θ)
		double alpha0_a; // α_0の事前分布であるガンマ分布のshapeパラメータ(k)
		double alpha0_b; // α_0の事前分布であるガンマ分布のrateパラメータ(1/θ)
		std::vector<Restaurant> restaurants;
		std::list<std::shared_ptr<Topic>> topics;
		std::unordered_map<int, double> cacheLogNPlusBeta; // log({n_v + i} + β)のキャッシュ。{n_v + i}がキー
		std::unordered_map<int, double> cacheLogNPlusVBeta; // log({n_. + i} + Vβ)のキャッシュ。{n_. + i}がキー
		int m; // 全店のテーブル数の合計
	};
}