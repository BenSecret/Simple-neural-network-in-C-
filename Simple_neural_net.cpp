#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

inline MatrixXf sigmoid(const MatrixXf& v) {
	return (1.0 + (-v).array().exp()).inverse().matrix(); }

inline MatrixXf sigmoidDerivative(const MatrixXf& v) {
	return v.array() * (1 - v.array()); }


int main() 
{
	MatrixXf X(4,3);
	X << 0, 0, 1,
	     0, 1, 1, 
	     1, 0, 1,
	     1, 1, 1;

	MatrixXf y(4,1);
	y << 0, 
	     1, 
	     1, 
	     0;

	MatrixXf synapse_0, synapse_1;
	synapse_0 = MatrixXf::Random(3, 4);
	synapse_1 = MatrixXf::Random(4, 1);

	MatrixXf layer_0, layer_1, layer_1_error, layer_1_delta,
		 layer_2, layer_2_error, layer_2_delta;

	float alpha = 1;

	for (int i = 0; i < 60000; i++)
	{
		layer_0 = X;
		layer_1 = sigmoid(layer_0 * synapse_0);
		layer_2 = sigmoid(layer_1 * synapse_1);
		layer_2_error = layer_2 - y;

		if (i % 10000 == 0) {
			cout << "\nError at " << i << " " << layer_2_error.mean() << endl;
		}

		layer_2_delta = layer_2_error.array() * sigmoidDerivative(layer_2).array();
		layer_1_error = layer_2_delta * synapse_1.transpose();
		layer_1_delta = layer_1_error.array() * sigmoidDerivative(layer_1).array();

		synapse_1 -= alpha * (layer_1.transpose() * layer_2_delta);
		synapse_0 -= alpha * (layer_0.transpose() * layer_1_delta);
	}
}
