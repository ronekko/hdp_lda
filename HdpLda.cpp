#include <boost/random.hpp>
#include "HdpLda.h"




HdpLda::HdpLda(const Corpus &corpus, const Vocabulary &vocabulary, const unsigned long seed
			 , const double &gamma, const double &alpha0, const double &beta, const int &K)
			 : corpus(corpus), vocabulary(vocabulary), gamma(gamma), alpha0(alpha0), beta(beta)
			 , D(corpus.D), V(corpus.V), N(corpus.N), K(K)
{
	m = 0;
	engine.seed(seed);
	
	restaurants.resize(D);

	topics.push_back(shared_ptr<Topic>(new Topic(V, beta)));
	shared_ptr<Topic> &firstTopic = *(topics.begin());

	for(int j=0; j<D; ++j){
		Restaurant &restaurant = restaurants[j];
		const Document &document = corpus.documents[j];

		restaurant.tables.push_back(shared_ptr<Table>(new Table(firstTopic)));
		shared_ptr<Table> &firstTable = *(restaurant.tables.begin());
		firstTopic->m++;
		m++;

		restaurant.n = document.tokens.size();
		restaurant.customers.resize(restaurant.n);
		for(int i=0; i<restaurant.n; ++i){
			int v = document.tokens[i].x;
			restaurant.customers[i].word = v;
			firstTable->n++;
			firstTable->n_v[v]++;
			firstTable->topic->n++;
			firstTable->topic->n_v[v]++;
			restaurant.customers[i].table = firstTable;
		}
	}
}



void HdpLda::sampleTables()
{
	using namespace boost;
	using std::shared_ptr;

	uniform_real<> uniformDistribution(0, 1);
	variate_generator<mt19937&, uniform_real<>> uniform(engine, uniformDistribution);

	for(int j=0; j<D; ++j){
		Restaurant &restaurant = restaurants[j];
		vector<Customer> &customers = restaurant.customers;
		list<shared_ptr<Table>> &tables = restaurant.tables;

		for(int i=0; i<customers.size(); ++i){
			int v = customers[i].word;
			customers[i].table->n--;
			customers[i].table->n_v[v]--;
			customers[i].table->topic->n--;
			customers[i].table->topic->n_v[v]--;
			// 客がいなくなった場合はそのテーブルを削除
			if(customers[i].table->n == 0){
				shared_ptr<Topic> &topicOfThisTable = customers[i].table->topic;
				tables.remove_if([](shared_ptr<Table> &table){
					return table->n == 0;
				});
				topicOfThisTable->m--;
				m--;
				// 削除されたテーブルに乗っていた料理が、全レストランのどのテーブルにも提供されていなくなった場合はその料理をフランチャイズのメニューから削除
				if(topicOfThisTable->m == 0){
					topics.remove_if([](shared_ptr<Topic> &topic){
						return topic->m == 0;
					});
				}
			}

			// テーブルごとの着席確率の離散累積分布を求める（ただし正規化されていない着席確率）
			int T = tables.size();
			vector<double> unnormalizedCDF(T + 1);
			vector<shared_ptr<Table>> ptrTables(T);
			double sum = 0.0;
			{
				int t = 0;
				for(auto table = tables.begin(); table!=tables.end(); ++table){
					sum += (*table)->n;
					unnormalizedCDF[t] = sum;
					ptrTables[t] = (*table);
					++t;
				}
				sum += alpha0;
				unnormalizedCDF[T] = sum;
			}
			
			// 離散累積分布からテーブル番号をサンプリング
			double rnd = uniform() * sum;
			int newT = T;
			for(int t=0; t<T+1; ++t){
				if(unnormalizedCDF[t] > rnd){
					newT = t;
					break;
				}
			}

			// 着席するテーブルを更新
			shared_ptr<Table> newTable;
			if(newT < T){ // 既存のテーブルの場合
				newTable = ptrTables[newT];
			}
			else{ // 新しいテーブルの場合
				shared_ptr<Topic> &topic = *(topics.begin()); // TODO: 新しいテーブル用にトピックをサンプリングするように
				newTable = shared_ptr<Table>(new Table(topic));
				tables.push_back(newTable);
				topic->m++;
				m++;
			}
			newTable->n++;
			newTable->n_v[v]++;
			newTable->topic->n++;
			newTable->topic->n_v[v]++;
			customers[i].table = newTable;
		}
	}


}



void HdpLda::sampleTopics(){}