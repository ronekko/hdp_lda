#define _CRTDBG_MAP_ALLOC // メモリリーク検出用コード
#include <stdlib.h> // メモリリーク検出用コード
#include <crtdbg.h> // メモリリーク検出用コード

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <boost/timer.hpp>
#include <boost/random.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <opencv2/opencv.hpp>
#include "direct.h"

#include "HdpLda.h"
#include "Corpus.hpp"

#ifdef _DEBUG
	#pragma comment(lib, "opencv_core246d.lib")
	#pragma comment(lib, "opencv_imgproc246d.lib")
	#pragma comment(lib, "opencv_highgui246d.lib")
	#pragma comment(lib, "opencv_features2d246d.lib")
#else
	#pragma comment(lib, "opencv_core246.lib")
	#pragma comment(lib, "opencv_imgproc246.lib")
	#pragma comment(lib, "opencv_highgui246.lib")
	#pragma comment(lib, "opencv_features2d246.lib")
#endif

using namespace std;


void showSticks2(const string &title, const vector<double> &stickLengths)
{
	using namespace cv;
	int h = 30;
	int w = 700;
	int K = stickLengths.size();

	vector<double> stickProportions(K);
	double totalStickLength = boost::accumulate(stickLengths, 0.0, [](double sum, double length){ return sum += length;});
	boost::transform(stickLengths, stickProportions.begin(), [totalStickLength](double length){ return length / totalStickLength;});

	Mat image = Mat::zeros(h, w, CV_8UC1);
	vector<int> breakingPoints(K);
	double sum = 0.0;
	for(int k=0; k<K; ++k){
		sum += stickProportions[k];
		breakingPoints[k] = sum * w;
	}
	int currentPoint = 0;
	for(int k=0; k<K; ++k){
		rectangle(image, Point(currentPoint, 0), Point(breakingPoints[k], h), Scalar((k%3)*128-1), CV_FILLED);
		currentPoint = breakingPoints[k];
	}
	imshow(title, image);
	waitKey(1);

}


int main(int argc, char** argv)
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // メモリリーク検出用コード




	const int K = 16;
	const int ITERATION = 2000;
	const int INTERVAL = 50;
	const double GAMMA = 0.1;
	const double ALPHA0 = 10;
	const double BETA = 0.5;

	const string corpusName = "kos";	// K=15~18あたりが最善
//	const string corpusName = "smallkos";
//	const string corpusName = "minimalkos";
//	const string corpusName = "nips";	// K=50近辺が最善

	Corpus corpus("docword." + corpusName + "\\docword." + corpusName + ".txt"); 
	Vocabulary vocabulary("docword." + corpusName + "\\vocab." + corpusName + ".txt", K);
//	Corpus corpus("C:\\Documents and Settings\\sakurai\\My Documents\\Dataset\\Clothing2\\corpus.txt");
//	Vocabulary vocabulary("C:\\Documents and Settings\\sakurai\\My Documents\\Dataset\\Clothing2\\vocab.txt", K);

//	hdplda::HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(time(0)), GAMMA, ALPHA0, BETA, K);
	hdplda::HdpLda hdp(corpus, vocabulary, static_cast<unsigned long>(0), GAMMA, ALPHA0, BETA, K);


	// いちばん良かった結果の保存用
	double bestPerplexity=DBL_MAX;
	vector<vector<double>> bestPhi;
	vector<vector<double>> bestTheta;

	// 保存ファイル名
	string dirName;
	using namespace boost::posix_time;
	dirName = to_iso_string(second_clock::local_time()) + " " + corpusName;
	_mkdir(dirName.c_str());
	stringstream ss;
	ss << dirName << "\\Perplexity, γ=" << GAMMA << ", α0=" << ALPHA0 << ", β" << BETA << ", ITER=" << ITERATION << ".txt";
	ofstream ofs(ss.str().c_str());

	hdp.showAllParameters();

	for(int i=0; i<ITERATION; ++i){
		cout << "*** " << i << " *******************************************************" << endl;
		
		boost::timer timer;
		hdp.sampling();
		if((i % 20) == 19){ hdp.sampleGamma(); }
		if((i % 30) == 29){ hdp.sampleAlpha0(20); }
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

		hdp.showAllCounts();
		hdp.showAllParameters();
		cout << "Perplexity: " << perplexity << endl;
		cout << endl;

		// スティックの表示
		vector<double> stickLengths = hdp.calcSticksOfG0();
		vector<double> entropy_k = hdp.calcEntropyOfTopics(phi);
		if((i % 30) == 0){
			for(int k=0; k<entropy_k.size(); ++k){
				cout << k << ": " << entropy_k[k] << endl;
			}
		}
		cout << boost::format("%s %s") % stickLengths.size() % entropy_k.size() << endl;;
		vector<double> weightedStickLengths(entropy_k.size());
		boost::transform(entropy_k, stickLengths, weightedStickLengths.begin(), [](double entropy, double length){
			return length / exp(entropy);
		});		
		showSticks2("Entropy", entropy_k);
		showSticks2("stick length", stickLengths);
		showSticks2("weightedStickLengths", weightedStickLengths);

	}
	ofs.close();

	stringstream ssp;
	ssp << dirName << "\\phi, γ=" << GAMMA << ", α0=" << ALPHA0 << ", β" << BETA << ", ITER=" << ITERATION << "PERPLEXITY=" << bestPerplexity << ".txt";
	stringstream sst;
	sst << dirName << "\\theta, γ=" << GAMMA << ", α0=" << ALPHA0 << ", β" << BETA << ", ITER=" << ITERATION << "PERPLEXITY=" << bestPerplexity << ".txt";
	hdp.savePhiTheta(bestPhi, ssp.str(), bestTheta, sst.str());

	cout << "getchar()";
	getchar();

	return 0;
}
