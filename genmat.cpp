//17/11/7 = Tue

//generate a random matrix, whose diagonal are zeros, and write the matrix into a file

#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <string>

#define INF 1000000

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main(int argc, char *argv[])
{
	string fname("mat");
	if (argc > 1)
		fname = argv[1];
	std::ofstream ofile(fname);
	int n = 10;
	double prob_inf = 0.5, prob_neg = 0.01, mean_pos = 100, mean_neg = 10;
	string sep(100, '-');
	cout << sep << endl;
	cout << "This program generates a matrix as input for Bellman-Ford algorithm." << endl;
	cout << "The matrix is to be stored in : " << fname << endl;
	cout << "The first line of the file indicates the number of nodes (n), which is followed by a n-by-n matrix." << endl;
	cout << "The diagonal of the matrix is guaranteed to be zero." << endl;
	cout << sep << endl;
	if (argc > 1) {
		cout << "Input number of nodes n (e.g. 10) : ";
		cin >> n;
		cout << "Input probability that no edge exists between two nodes (e.g. 0.5) : ";
		cin >> prob_inf;
		cout << "Input probability of negative edges (e.g. 0.01) : ";
		cin >> prob_neg;
		cout << "Input mean length of positive edges (e.g. 100) : ";
		cin >> mean_pos;
		cout << "Input mean length of negative edges (e.g. 10) : ";
		cin >> mean_neg;
	}
	cout << "Number of nodes n = " << n << endl;
	cout << "Probability that no edge exists between two nodes = " << prob_inf << endl;
	cout << "Probability of negative edges = " << prob_neg << endl;
	cout << "Mean length of positive edges = " << mean_pos << endl;
	cout << "Mean length of negative edges = " << mean_neg << endl;
	cout << sep << endl;
	std::default_random_engine e(time(0));
	std::bernoulli_distribution ber_inf(prob_inf);
	std::bernoulli_distribution ber_neg(prob_neg);
	std::exponential_distribution<double> exp_pos(1/mean_pos);
	std::exponential_distribution<double> exp_neg(1/mean_neg);

	ofile << n << endl;
	int k = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)
				ofile << 0;
			else {
				bool inf = ber_inf(e);
				if (inf)
					ofile << INF;
				else {
					int r;
					std::exponential_distribution<double> exp;
					bool neg = ber_neg(e);
					exp = neg ? exp_neg : exp_pos;
					do
						r = (int)exp(e);
					while (r >= INF);
					if (neg)
						r *= -1;
					ofile << r;
				}
			}
			if (j != n - 1)
				ofile << '\t';
		}
		ofile << endl;
	}
	ofile.close();
	return 0;
}