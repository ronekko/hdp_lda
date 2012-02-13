#define _CRTDBG_MAP_ALLOC // ���������[�N���o�p�R�[�h
#include <stdlib.h> // ���������[�N���o�p�R�[�h
#include <crtdbg.h> // ���������[�N���o�p�R�[�h

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
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // ���������[�N���o�p�R�[�h




	const int K = 16;
	const int ITERATION = 500;
	const int INTERVAL = 50;
	const double GAMMA = 0.5;
	const double ALPHA0 = 0.1;
	const double BETA = 0.5;

	const string corpusName = "kos";	// K=15~18�����肪�őP
//	const string corpusName = "smallkos";
//	const string corpusName = "minimalkos";
//	const string corpusName = "nips";	// K=50�ߕӂ��őP

	Corpus corpus("docword." + corpusName + "\\docword." + corpusName + ".txt"); 
	Vocabulary vocabulary("docword." + corpusName + "\\vocab." + corpusName + ".txt", K);
//	Corpus corpus("C:\\Documents and Settings\\sakurai\\My Documents\\Dataset\\Clothing2\\corpus.txt");
//	Vocabulary vocabulary("C:\\Documents and Settings\\sakurai\\My Documents\\Dataset\\Clothing2\\vocab.txt", K);

	HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(time(0)), GAMMA, ALPHA0, BETA, K);
//	HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(0), GAMMA, ALPHA0, BETA, K);


	// �����΂�ǂ��������ʂ̕ۑ��p
	double bestPerplexity=DBL_MAX;
	vector<vector<double>> bestPhi;
	vector<vector<double>> bestTheta;

	// �ۑ��t�@�C����
	string dirName;
	using namespace boost::posix_time;
	dirName = to_iso_string(second_clock::local_time()) + " " + corpusName;
	_mkdir(dirName.c_str());
	stringstream ss;
	ss << dirName << "\\Perplexity, ��=" << GAMMA << ", ��0=" << ALPHA0 << ", ��" << BETA << ", ITER=" << ITERATION << ".txt";
	ofstream ofs(ss.str().c_str());

	for(int i=0; i<ITERATION; ++i){
		cout << "*** " << i << " *******************************************************" << endl;
		
		boost::timer timer;
		hdp.sampling();
		cout << "time: " << timer.elapsed() << endl;

		vector<vector<double> > phi = hdp.calcPhi();
		vector<vector<double> > theta = hdp.calcTheta();
		double perplexity = hdp.calcPerplexity(phi, theta);
		ofs << perplexity << endl;

		if(perplexity < bestPerplexity){
			bestPerplexity = perplexity;
			bestPhi = phi;
			bestTheta = theta;
		}

		cout << "hdp.m: " << hdp.m << endl;
		cout << "hdp.topics.size(): " << hdp.topics.size() << endl;	
		cout << "hdp.topics[0].n: " << boost::accumulate(hdp.topics, 0, [](int n, shared_ptr<Topic> t){return n + t->n;}) << endl;	
		cout << "Perplexity: " << perplexity << endl;
		cout << endl;
	}
	ofs.close();

	stringstream ssp;
	ssp << dirName << "\\phi, ��=" << GAMMA << ", ��0=" << ALPHA0 << ", ��" << BETA << ", ITER=" << ITERATION << "PERPLEXITY=" << bestPerplexity << ".txt";
	stringstream sst;
	sst << dirName << "\\theta, ��=" << GAMMA << ", ��0=" << ALPHA0 << ", ��" << BETA << ", ITER=" << ITERATION << "PERPLEXITY=" << bestPerplexity << ".txt";
	hdp.savePhiTheta(bestPhi, ssp.str(), bestTheta, sst.str());

	return 0;
}
