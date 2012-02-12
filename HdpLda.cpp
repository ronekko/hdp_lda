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
			int K = topics.size();
			vector<double> unnormalizedCDF(T + 1);
			vector<shared_ptr<Table>> ptrTables(T);
			double sum = 0.0;
			
			// テーブルサンプリング式(式24)の第二項目のG_0(v)を求める
			// ついでにそのとき計算するψ_jt (== φ_k)をキャッシュしておく
			// また、新しいテーブルが選ばれたときのためにk ∝ G0_v(k)の(正規化されていない)離散累積分布も求めておく
			
			double G0_v = 0.0;
			vector<double> phi_kv(T);
			vector<double> G0_vk(K+1);
			vector<shared_ptr<Topic>> ptrTopics(K);
			{
				int k = 0;
				for(auto topic=topics.begin(); topic!=topics.end(); ++topic){
					(*topic)->phi_v[v] = ((*topic)->n_v[v] + beta) / ((*topic)->n + V * beta); // φ_k(v)
					G0_v += (*topic)->m * (*topic)->phi_v[v]; //m_.k * φ_k(v)
					G0_vk[k] = G0_v;
					ptrTopics[k] = (*topic);
				}
				G0_v += gamma * (1.0 / static_cast<double>(V)); // γH(v), because H(v) = β/Vβ = 1/V
				G0_vk[K] = G0_v;
				G0_v /= (m + gamma);
			}

			// テーブルサンプリング式(式24)の第一項目の分子を求める（分母は必要ない）
			{
				int t = 0;
				for(auto table = tables.begin(); table!=tables.end(); ++table){
					shared_ptr<Topic> &topic = (*table)->topic;
					sum += (*table)->n * topic->phi_v[v]; // n_jt. * ψ_jt(v)
					unnormalizedCDF[t] = sum;
					ptrTables[t] = (*table);
					++t;
				}
			}

			sum += alpha0 * G0_v; // 新しいテーブルに着く確率
			unnormalizedCDF[T] = sum;
			
			// 離散累積分布からテーブル番号をサンプリング
			double tRnd = uniform() * unnormalizedCDF[T];
			int tNew = T;
			for(int t=0; t<T+1; ++t){
				if(unnormalizedCDF[t] > tRnd){
					tNew = t;
					break;
				}
			}

			// 着席するテーブルを更新
			shared_ptr<Table> newTable;
			if(tNew < T){ // 既存のテーブルの場合
				newTable = ptrTables[tNew];
			}
			else{ // 新しいテーブルの場合
				// 新しいテーブルに乗せる料理（トピック）のサンプリング
				double kRnd = uniform() * G0_vk[K];
				int kNew = K;
				for(int k=0; k<K+1; ++k){
					if(G0_vk[k] > kRnd){
						kNew = k;
						//for(int b=0; b<=K; ++b){cout << G0_vk[b] << " ";}
						//cout <<"(" <<kRnd<<"), G0_v = "<<G0_v<< endl;
						break;
					}
				}
				shared_ptr<Topic> newTopic;
				if(kNew < K){ // 既存のトピックの場合
					newTopic = ptrTopics[kNew];
				}
				else{ // 新しいトピックの場合
					newTopic = shared_ptr<Topic>(new Topic(V, beta));
					topics.push_back(newTopic);
				}
				newTable = shared_ptr<Table>(new Table(newTopic));
				tables.push_back(newTable);
				newTopic->m++;
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