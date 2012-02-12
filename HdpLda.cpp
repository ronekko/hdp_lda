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



void HdpLda::sampleTopics(){}