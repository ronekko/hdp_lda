#define _CRTDBG_MAP_ALLOC // メモリリーク検出用コード
#include <stdlib.h> // メモリリーク検出用コード
#include <crtdbg.h> // メモリリーク検出用コード

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/range/numeric.hpp>
#include <boost/timer.hpp>
#include <boost/random.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "direct.h"

#include "HdpLda.h"
#include "Corpus.hpp"

using namespace std;


int main(int argc, char** argv)
{
_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // メモリリーク検出用コード
	boost::timer timer;
	string dirName;
	{
		using namespace boost::posix_time;
		dirName = to_iso_string(second_clock::local_time());
	}
	_mkdir(dirName.c_str());

	const int K = 16;
	const int ITERATION = 300;
	const int INTERVAL = 50;
	const double GAMMA = 1.0;
	const double ALPHA0 = 0.1;
	const double BETA = 0.5;

	
	Corpus corpus("docword.kos\\docword.kos.txt"); // K=15~18あたりが最善
//	Corpus corpus("docword.kos\\docword.smallkos.txt"); // K=15~18あたりが最善
//	Corpus corpus("docword.kos\\docword.minimalkos.txt"); // K=15~18あたりが最善
	Vocabulary vocabulary("docword.kos\\vocab.kos.txt", K);
//	Corpus corpus("docword.nips\\docword.nips.txt"); // K=50近辺が最善
//	Vocabulary vocabulary("docword.nips\\vocab.nips.txt", K);
//	Corpus corpus("C:\\Documents and Settings\\sakurai\\My Documents\\Dataset\\Clothing2\\corpus.txt"); // K=15~18あたりが最善
//	Vocabulary vocabulary("C:\\Documents and Settings\\sakurai\\My Documents\\Dataset\\Clothing2\\vocab.txt", K);

//	HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(time(0)), GAMMA, ALPHA0, BETA, K);
	HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(0), GAMMA, ALPHA0, BETA, K);

	for(int i=0; i<1000; ++i){
		cout << "*** " << i << " *******************************************************" << endl;
		hdp.sampleTables();
		hdp.sampleTopics();
		cout << "hdp.m: " << hdp.m << endl;
		cout << "hdp.topics.size(): " << hdp.topics.size() << endl;	
		cout << "hdp.topics[0].n: " << boost::accumulate(hdp.topics, 0, [](int n, shared_ptr<Topic> t){return n + t->n;}) << endl;
		cout << endl;
	}
	for(int j=0; j<hdp.restaurants.size(); ++j){ cout << "[" << j << "] " << hdp.restaurants[j].tables.size() << endl; }

/*//	cout << "\n############### K = " << K << " : ROUND " << round << " ###################" << endl;
	HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(time(0)), GAMMA, ALPHA0, BETA, K);

	stringstream ss;
	ss << dirName << "\\K=" << K << "_" << round << ", ITER=" << ITERATION << " alpha=" << alpha << " beta=" << beta << ".txt";
	ofstream ofs(ss.str().c_str());

	double bestPerplexity=DBL_MAX;
	vector<vector<double>> bestPhi;
	vector<vector<double>> bestTheta;

	for(int i=0; i<ITERATION; ++i){
		timer.restart();
//		lda.update();
		cout << "time: " << timer.elapsed() << endl;
//		lda.showAllCounts();

		vector<vector<double> > phi = lda.calcPhi();
		vector<vector<double> > theta = lda.calcTheta();
		double perplexity = lda.calcPerplexity(phi, theta);
		cout << "iteration " << i << "\t- perplexity: " << perplexity << endl;
		ofs << perplexity << endl;

		if(perplexity < bestPerplexity){
			bestPerplexity = perplexity;
			bestPhi = phi;
			bestTheta = theta;
		}
		//if(i%INTERVAL == 0){
		//	cout << "\niteration: " << i << endl;
		//	stringstream ss;
		//	ss << dirName << "\\" << i << ".txt";
		//	lda.saveCurrentModel(ss.str());
		//}
		//cout << "*";
	}
	ofs.close();

	stringstream ssp;
	ssp << dirName << "\\phi, K=" << K << "_" << round << ", ITER=" << ITERATION << " PERPLEXITY=" << bestPerplexity << " alpha=" << alpha << " beta=" << beta << ".txt";
	stringstream sst;
	sst << dirName << "\\theta, K=" << K << "_" << round << ", ITER=" << ITERATION << " PERPLEXITY=" << bestPerplexity << " alpha=" << alpha << " beta=" << beta << ".txt";
//	lda.savePhiTheta(bestPhi, ssp.str(), bestTheta, sst.str());
*/

	return 0;
}
