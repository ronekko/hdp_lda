#include <cmath>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <boost/random.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <boost/timer.hpp>
#include "HdpLda.h"
#include <omp.h>




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



void HdpLda::sampling(void)
{
	sampleTables();
	sampleTopics();
	K = topics.size();
}



void HdpLda::sampleTables(void)
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
			int &dbg = customers[i].table->topic->n;
			// �q�����Ȃ��Ȃ����ꍇ�͂��̃e�[�u�����폜
			if(customers[i].table->n == 0){
				shared_ptr<Topic> &topicOfThisTable = customers[i].table->topic;
				tables.remove_if([](shared_ptr<Table> &table){
					return table->n == 0;
				});
				topicOfThisTable->m--;
				m--;
				// �폜���ꂽ�e�[�u���ɏ���Ă����������A�S���X�g�����̂ǂ̃e�[�u���ɂ��񋟂���Ă��Ȃ��Ȃ����ꍇ�͂��̗������t�����`���C�Y�̃��j���[����폜
				if(topicOfThisTable->m == 0){
					topics.remove_if([](shared_ptr<Topic> &topic){
						return topic->m == 0;
					});
				}
			}

			// �e�[�u�����Ƃ̒��Ȋm���̗��U�ݐϕ��z�����߂�i���������K������Ă��Ȃ����Ȋm���j
			int T = tables.size();
			int K = topics.size();
			vector<double> unnormalizedCDF(T + 1);
			vector<shared_ptr<Table>> ptrTables(T);
			double sum = 0.0;
			
			// �e�[�u���T���v�����O��(��24)�̑�񍀖ڂ�G_0(v)�����߂�
			// ���łɂ��̂Ƃ��v�Z�����_jt (== ��_k)���L���b�V�����Ă���
			// �܂��A�V�����e�[�u�����I�΂ꂽ�Ƃ��̂��߂�k �� G0_v(k)��(���K������Ă��Ȃ�)���U�ݐϕ��z�����߂Ă���
			
			double G0_v = 0.0;
			vector<double> phi_kv(T);
			vector<double> G0_vk(K+1);
			vector<shared_ptr<Topic>> ptrTopics(K);
			{
				int k = 0;
				for(auto topic=topics.begin(); topic!=topics.end(); ++topic){
					(*topic)->phi_v[v] = ((*topic)->n_v[v] + beta) / ((*topic)->n + V * beta); // ��_k(v)
					G0_v += (*topic)->m * (*topic)->phi_v[v]; //m_.k * ��_k(v)
					G0_vk[k] = G0_v;
					ptrTopics[k] = (*topic);
				}
				G0_v += gamma * (1.0 / static_cast<double>(V)); // ��H(v), because H(v) = ��/V�� = 1/V
				G0_vk[K] = G0_v;
				G0_v /= (m + gamma);
			}

			// �e�[�u���T���v�����O��(��24)�̑�ꍀ�ڂ̕��q�����߂�i����͕K�v�Ȃ��j
			{
				int t = 0;
				for(auto table = tables.begin(); table!=tables.end(); ++table){
					shared_ptr<Topic> &topic = (*table)->topic;
					sum += (*table)->n * topic->phi_v[v]; // n_jt. * ��_jt(v)
					unnormalizedCDF[t] = sum;
					ptrTables[t] = (*table);
					++t;
				}
			}

			sum += alpha0 * G0_v; // �V�����e�[�u���ɒ����m��
			unnormalizedCDF[T] = sum;
			
			// ���U�ݐϕ��z����e�[�u���ԍ����T���v�����O
			double tRnd = uniform() * unnormalizedCDF[T];
			int tNew = T;
			for(int t=0; t<T+1; ++t){
				if(unnormalizedCDF[t] > tRnd){
					tNew = t;
					break;
				}
			}

			// ���Ȃ���e�[�u�����X�V
			shared_ptr<Table> newTable;
			if(tNew < T){ // �����̃e�[�u���̏ꍇ
				newTable = ptrTables[tNew];
			}
			else{ // �V�����e�[�u���̏ꍇ
				// �V�����e�[�u���ɏ悹�闿���i�g�s�b�N�j�̃T���v�����O
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
				if(kNew < K){ // �����̃g�s�b�N�̏ꍇ
					newTopic = ptrTopics[kNew];
				}
				else{ // �V�����g�s�b�N�̏ꍇ
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


inline double HdpLda::calcLogNPlusBeta(int n)
{
	return (cacheLogNPlusBeta.find(n) != cacheLogNPlusBeta.end()) ? cacheLogNPlusBeta[n]
																  : (cacheLogNPlusBeta[n] = log(n + beta));
}
inline double HdpLda::calcLogNPlusVBeta(int n)
{
	return (cacheLogNPlusVBeta.find(n) != cacheLogNPlusVBeta.end()) ? cacheLogNPlusVBeta[n]
																	: (cacheLogNPlusVBeta[n] = log(n + V * beta));
}

void HdpLda::sampleTopics(void)
{
	using namespace boost;
	using std::shared_ptr;
	timer timer;
	double tm = 0.0;
	uniform_real<> uniformDistribution(0, 1);
	variate_generator<mt19937&, uniform_real<>> uniform(engine, uniformDistribution);
	
	for(int j=0; j<D; ++j){
		Restaurant &restaurant = restaurants[j];
		vector<Customer> &customers = restaurant.customers;
		list<shared_ptr<Table>> &tables = restaurant.tables;
		
		timer.restart();
		for(auto it=tables.begin(); it!=tables.end(); ++it){
			shared_ptr<Table> &table = *it;
			shared_ptr<Topic> &oldTopic = table->topic;
			m--;
			oldTopic->m--;

			// ���̃e�[�u���ɒ����Ă���P��Ƃ��̃J�E���g�����߂Ă���
			vector<pair<int, int>> n_v; // n_v.first = v, n_v.second = table->n_v[v]
			for(int v=0; v<V; ++v){
				if(table->n_v[v] != 0){
					n_v.push_back(pair<int, int>(v, table->n_v[v]));
				}
			}
			// �������񋟂���Ă���e�[�u�����Ȃ��Ȃ����烁�j���[���痿�����폜
			if(oldTopic->m == 0){
				topics.remove_if([](shared_ptr<Topic> &topic){
					return topic->m == 0;
				});
			}
			else{	
				oldTopic->n -= table->n;
				for(int l=0; l<n_v.size(); ++l){
					int v = n_v[l].first;
					int count = n_v[l].second;
					oldTopic->n_v[v] -= count;
				}
			}
			
			// �������Ƃ̑I���m�������߂�
			// �������A�l�����ɏ������Ȃ�̂őΐ��Ōv�Z����
			int K = topics.size();
			vector<double> unnormalizedCDF(K + 1);
			vector<double> logPk(K + 1, 0.0);
			vector<shared_ptr<Topic>> ptrTopics(K);
			boost::copy(topics, ptrTopics.begin());

//#pragma omp for
			for(int k=0; k<K; ++k){
				shared_ptr<Topic> &topic = ptrTopics[k];

				logPk[k] = log(static_cast<double>(topic->m));
				for(int i=0; i<table->n; ++i){
					logPk[k] -= log(topic->n + i + V * beta);
					//���̂悤��log�𖈉�v�Z����̂͏d���̂ŃL���b�V���������̂��g��
					//logPk[k] -= calcLogNPlusVBeta(topic->n + i);
				}

				for(int l=0; l<n_v.size(); ++l){
					for(int i=0; i<n_v[l].second; ++i){
						logPk[k] += log(topic->n_v[n_v[l].first] + i + beta);
						//���̂悤��log�𖈉�v�Z����̂͏d���̂ŃL���b�V���������̂��g��
						//logPk[k] += calcLogNPlusBeta(topic->n_v[n_v[l].first] + i); 
					}
				}
			}
			// �V�����������T���v�����O����m���̑ΐ�
			logPk[K] = log(gamma);
			for(int i=0; i<table->n; ++i){
				logPk[K] -= log(i + V * beta);
				//logPk[K] -= calcLogNPlusVBeta(i);;
			}
			for(int l=0; l<n_v.size(); ++l){
				for(int i=0; i<n_v[l].second; ++i){
					logPk[K] += log(i + beta);
					//���̂悤��log�𖈉�v�Z����̂͏d���̂ŃL���b�V���������̂��g��
					//logPk[K] += calcLogNPlusBeta(i);
				}
			}
			
			double maxLogP = *(boost::min_element(logPk));
			unnormalizedCDF[0] = exp(logPk[0] - maxLogP);
			for(int k=1; k<K+1; ++k){
				unnormalizedCDF[k] = unnormalizedCDF[k-1] + exp(logPk[k] - maxLogP);
			}

			// ���U�ݐϕ��z����e�[�u���ԍ����T���v�����O
			double kRnd = uniform() * unnormalizedCDF[K];
			int kNew = K;
			for(int k=0; k<K+1; ++k){
				if(unnormalizedCDF[k] > kRnd){
					kNew = k;
					break;
				}
			}

			shared_ptr<Topic> newTopic;
			if(kNew < K){ // �����̃g�s�b�N�̏ꍇ
				newTopic = ptrTopics[kNew];
			}
			else{ // �V�����g�s�b�N�̏ꍇ
				newTopic = shared_ptr<Topic>(new Topic(V, beta));
				topics.push_back(newTopic);
			}
			newTopic->m++;
			newTopic->n += table->n;
			for(int l=0; l<n_v.size(); ++l){
				int v = n_v[l].first;
				int count = n_v[l].second;
				newTopic->n_v[v] += count;
			}
			table->topic = newTopic;
			m++;
		}
		tm += timer.elapsed();
	}
	cout << "\ttm="<<tm<<endl;
}



double HdpLda::calcPerplexity(const vector<vector<double>> &phi, const vector<vector<double>> &theta)
{
	double perplexity = 0.0;
	const int K = phi.size();

	for(int j=0; j<D; ++j){
		vector<Customer> &customers = restaurants[j].customers;
		for(int i=0; i<customers.size(); ++i){
			int v = customers[i].word;
			double p_v = 0.0;
			for(int k=0; k<K; ++k){
				p_v += theta[j][k] * phi[k][v];
			}
			perplexity -= log(p_v);
		}
	}

	perplexity = exp(perplexity / static_cast<double>(N));

	return perplexity;
}



vector<vector<double>> HdpLda::calcPhi(void)
{
	
	const int K = topics.size();
	vector<vector<double>> phi(K);

	boost::for_each(phi, [&](vector<double> &phi_k){phi_k.resize(V, 0.0);});

	vector<shared_ptr<Topic>> ptrTopics(K);
	boost::copy(topics, ptrTopics.begin());

	// �������߂�
	for(int k=0; k<K; ++k){
		shared_ptr<Topic> &topic = ptrTopics[k];
		for(int v=0; v<V; ++v){
			phi[k][v] = (topic->n_v[v] + beta) / (topic->n + V * beta);
		}
	}

	return phi;
}



vector<vector<double>> HdpLda::calcTheta(void)
{
	double perplexity = 0.0;
	const int K = topics.size();
	vector<vector<double>> theta(D);
	boost::for_each(theta, [&](vector<double> &theta_j){theta_j.resize(K, 0.0);});

	vector<shared_ptr<Topic>> ptrTopics(K);
	boost::copy(topics, ptrTopics.begin());

	for(int j=0; j<D; ++j){
		const int T = restaurants[j].tables.size();
		vector<shared_ptr<Table>> tables(T);		
		boost::copy(restaurants[j].tables, tables.begin());
		
		for(int t=0; t<T; ++t){
			int k = distance(ptrTopics.begin(), boost::find(ptrTopics, tables[t]->topic));
			theta[j][k] += tables[t]->n;
		}
		for(int k=0; k<K; ++k){
			theta[j][k] += alpha0 * (ptrTopics[k]->m + gamma / static_cast<double>(K)) / (m + gamma);
			theta[j][k] /= (restaurants[j].n + alpha0);
		}
	}

	return theta;
}



void HdpLda::savePhi(const vector<vector<double>> &phi, const string &fileName)
{	
	const int K = phi.size();
	ofstream ofs(fileName.c_str());
	for(int k=0; k<K; ++k){
		ofs << "Topic: " << k << endl;
		vector<pair<double, string>> phi_k;
		phi_k.resize(V);
		for(int v=0; v<V; ++v){
			phi_k[v].first = phi[k][v];
			phi_k[v].second = vocabulary.words[v].str;
		}

		boost::sort(phi_k, greater<pair<double, string>>());
			
		for(int v=0; v<20; ++v){
			ofs << "\t" << phi_k[v].second << ": " << phi_k[v].first << endl;
		}
		ofs << "\n" << endl;
	}
	ofs.close();
}



void HdpLda::saveTheta(const vector<vector<double>> &theta, const string &fileName)
{
	const int K = theta[0].size();
	ofstream ofs(fileName.c_str());
	for(int j=0; j<D; ++j){
		ofs << "Document: " << j << endl;
		vector<pair<double, int>> theta_j;
		theta_j.resize(K);
		for(int k=0; k<K; ++k){
			theta_j[k].first = theta[j][k];
			theta_j[k].second = k;
		}

		boost::sort(theta_j, greater<pair<double, int>>());
			
		for(int k=0; k<K; ++k){
			ofs << "\t" << theta_j[k].second << ": " << theta_j[k].first << endl;
		}
		ofs << "\n" << endl;
	}
	ofs.close();
}



void HdpLda::savePhiTheta(const vector<vector<double>> &phi, const string &phiFileName,
					const vector<vector<double>> &theta, const string &thetaFileName)
{
	savePhi(phi, phiFileName);
	saveTheta(theta, thetaFileName);
}