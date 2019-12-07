#include <iostream>
#include <chrono>
#include "Neuron.cuh"
#include "Matrix.cuh"
#include "NeuralNetwork.cuh"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv)
{
	//test neurons
	/*
	Neuron* n = new Neuron(0.9);
	cout << "Val: " << n->getVal() << endl;
	cout << "ActivatedVal: " << n->getActivatedVal() << endl;
	cout << "DerivedVal: " << n->getDerivedVal() << endl;
	*/

	//test random matrix weights creation and transposing
	/*
	Matrix* m = new Matrix(3, 2, true);
	m->printToConsole();

	cout << "--------------------------------------------------" << endl;

	Matrix* mT = m->transpose();
	mT->printToConsole();
	*/

	//test network creation with input
	/*
	vector<int> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(3);

	vector<double> input;
	input.push_back(1.0);
	input.push_back(0.0);
	input.push_back(1.0);

	NeuralNetwork* nn = new NeuralNetwork(topology);
	nn->setCurrentInput(input);

	nn->printToConsole();
	*/

	//check feedforward is calculating correctly through each layer
	/*
	vector<double> input;
	input.push_back(1);
	input.push_back(0);
	input.push_back(1);

	vector<int> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	NeuralNetwork* nn = new NeuralNetwork(topology);
	nn->setCurrentInput(input);
	nn->feedForward();
	nn->printToConsole();
	*/

	//check error calculations
	/*
	vector<double> input;
	input.push_back(1);
	input.push_back(0);
	input.push_back(1);

	vector<int> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(3);

	NeuralNetwork* nn = new NeuralNetwork(topology);
	nn->setCurrentInput(input);
	nn->setCurrentTarget(input);
	nn->feedForward();
	nn->setErrors();

	nn->printToConsole();

	cout << "Total Error: " << nn->getTotalError() << endl;
	*/

	//test backprop
	/*
	vector<double> input;
	input.push_back(1);
	input.push_back(0);
	input.push_back(1);

	vector<int> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(3);

	NeuralNetwork* nn = new NeuralNetwork(topology);
	nn->setCurrentInput(input);
	nn->setCurrentTarget(input);

	//training process
	for (int i = 0; i < 100000; ++i)
	{
		cout << "Epoch: " << i + 1 << endl;
		nn->feedForward();
		nn->setErrors();
		cout << "Total Error: " << nn->getTotalError() << endl;
		nn->backPropagation();
	}
	*/
	auto start = high_resolution_clock::now();

	//Start Copy Area
	//---------------
	vector<double> input;
	input.push_back(0.204829);
	input.push_back(0.905264);
	input.push_back(0.908988);
	input.push_back(0.111249);
	input.push_back(0.204858);
	input.push_back(0.272447);
	input.push_back(0.413717);
	input.push_back(0.38753);
	input.push_back(0.36959);
	input.push_back(0.922271);
	input.push_back(0.789755);
	input.push_back(0.325534);
	input.push_back(0.634112);
	input.push_back(0.498457);
	input.push_back(0.21524);
	input.push_back(0.933576);
	input.push_back(0.921143);
	input.push_back(0.755182);
	input.push_back(0.116525);
	input.push_back(0.559233);
	input.push_back(0.57177);
	input.push_back(0.0726854);
	input.push_back(0.670817);
	input.push_back(0.119705);
	input.push_back(0.488141);
	input.push_back(0.361441);
	input.push_back(0.215676);
	input.push_back(0.721447);
	input.push_back(0.369415);
	input.push_back(0.0293743);
	input.push_back(0.147754);
	input.push_back(0.970896);
	input.push_back(0.250955);
	input.push_back(0.539908);
	input.push_back(0.605861);
	input.push_back(0.960779);
	input.push_back(0.635517);
	input.push_back(0.316769);
	input.push_back(0.960725);
	input.push_back(0.621318);
	input.push_back(0.965797);
	input.push_back(0.807346);
	input.push_back(0.570731);
	input.push_back(0.605002);
	input.push_back(0.554977);
	input.push_back(0.329614);
	input.push_back(0.684465);
	input.push_back(0.753524);
	input.push_back(0.863657);
	input.push_back(0.340875);
	input.push_back(0.431487);
	input.push_back(0.7814);
	input.push_back(0.98531);
	input.push_back(0.128381);
	input.push_back(0.0876298);
	input.push_back(0.139685);
	input.push_back(0.872101);
	input.push_back(0.998877);
	input.push_back(0.255069);
	input.push_back(0.724658);
	input.push_back(0.144958);
	input.push_back(0.31156);
	input.push_back(0.686715);
	input.push_back(0.0477682);
	input.push_back(0.765126);
	input.push_back(0.0244933);
	input.push_back(0.827679);
	input.push_back(0.072816);
	input.push_back(0.343121);
	input.push_back(0.424249);
	input.push_back(0.81941);
	input.push_back(0.412208);
	input.push_back(0.508354);
	input.push_back(0.138124);
	input.push_back(0.128154);
	input.push_back(0.673945);
	input.push_back(0.882555);
	input.push_back(0.508395);
	input.push_back(0.590559);
	input.push_back(0.0423979);
	input.push_back(0.320903);
	input.push_back(0.325571);
	input.push_back(0.151148);
	input.push_back(0.742018);
	input.push_back(0.90565);
	input.push_back(0.761725);
	input.push_back(0.621686);
	input.push_back(0.911977);
	input.push_back(0.898315);
	input.push_back(0.713556);
	input.push_back(0.0673958);
	input.push_back(0.44961);
	input.push_back(0.58746);
	input.push_back(0.824068);
	input.push_back(0.913278);
	input.push_back(0.842669);
	input.push_back(0.779844);
	input.push_back(0.0169593);
	input.push_back(0.453853);
	input.push_back(0.980303);

	vector<double> target;
	target.push_back(0.112477);
	target.push_back(0.707666);
	target.push_back(0.34463);
	target.push_back(0.350625);
	target.push_back(0.860009);
	target.push_back(0.612047);
	target.push_back(0.883199);
	target.push_back(0.222597);
	target.push_back(0.13941);
	target.push_back(0.175897);
	target.push_back(0.000857692);
	target.push_back(0.343907);
	target.push_back(0.627541);
	target.push_back(0.0594374);
	target.push_back(0.104905);
	target.push_back(0.0359212);
	target.push_back(0.905715);
	target.push_back(0.141927);
	target.push_back(0.996518);
	target.push_back(0.215232);
	target.push_back(0.0824562);
	target.push_back(0.745878);
	target.push_back(0.0143425);
	target.push_back(0.215253);
	target.push_back(0.320253);
	target.push_back(0.509083);
	target.push_back(0.0856378);
	target.push_back(0.201664);
	target.push_back(0.359278);
	target.push_back(0.628177);
	target.push_back(0.196674);
	target.push_back(0.123806);
	target.push_back(0.0389247);
	target.push_back(0.372505);
	target.push_back(0.727115);
	target.push_back(0.476662);
	target.push_back(0.511225);
	target.push_back(0.91862);
	target.push_back(0.652862);
	target.push_back(0.905527);
	target.push_back(0.365569);
	target.push_back(0.633777);
	target.push_back(0.840801);
	target.push_back(0.415222);
	target.push_back(0.520421);
	target.push_back(0.657444);
	target.push_back(0.421453);
	target.push_back(0.768645);
	target.push_back(0.186551);
	target.push_back(0.191828);
	target.push_back(0.503818);
	target.push_back(0.313275);
	target.push_back(0.294159);
	target.push_back(0.816052);
	target.push_back(0.804983);
	target.push_back(0.947408);
	target.push_back(0.229653);
	target.push_back(0.834781);
	target.push_back(0.958809);
	target.push_back(0.247835);
	target.push_back(0.700575);
	target.push_back(0.525473);
	target.push_back(0.132892);
	target.push_back(0.340281);
	target.push_back(0.550285);
	target.push_back(0.0206471);
	target.push_back(0.535477);
	target.push_back(0.601023);
	target.push_back(0.680661);
	target.push_back(0.735823);
	target.push_back(0.539517);
	target.push_back(0.0106388);
	target.push_back(0.890069);
	target.push_back(0.545499);
	target.push_back(0.393838);
	target.push_back(0.648125);
	target.push_back(0.036597);
	target.push_back(0.113843);
	target.push_back(0.388771);
	target.push_back(0.459056);
	target.push_back(0.875386);
	target.push_back(0.61228);
	target.push_back(0.734639);
	target.push_back(0.284857);
	target.push_back(0.440033);
	target.push_back(0.721332);
	target.push_back(0.90889);
	target.push_back(0.150754);
	target.push_back(0.0327202);
	target.push_back(0.677347);
	target.push_back(0.169406);
	target.push_back(0.065766);
	target.push_back(0.849176);
	target.push_back(0.557173);
	target.push_back(0.944951);
	target.push_back(0.617289);
	target.push_back(0.801286);
	target.push_back(0.895104);
	target.push_back(0.621062);
	target.push_back(0.0791005);

	vector<int> topology;
	topology.push_back(100);
	topology.push_back(94);
	topology.push_back(52);
	topology.push_back(53);
	topology.push_back(100);
	//---------------
	//aerA ypoC tratS

	NeuralNetwork* nn = new NeuralNetwork(topology);
	nn->setCurrentInput(input);
	nn->setCurrentTarget(target);

	//training process
	for (int i = 0; i < 10000; ++i)
	{
		cout << "Epoch: " << i + 1 << endl;
		nn->feedForward();
		nn->setErrors();
		cout << "Total Error: " << nn->getTotalError() << endl;
		nn->backPropagation();

		cout << "----------------------------------------" << endl;
		cout << "OUTPUT: ";
		nn->printOutputToConsole();

		cout << "TARGET: ";
		nn->printTargetToConsole();
		cout << "----------------------------------------" << endl;
		cout << endl;

	}

	//nn->printHistoricalErrors();

	delete nn;

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	cout << "CUDA" << endl;
	cout << "Time taken by function: "
		<< duration.count() / 1000000 << " seconds" << endl;

	return 0;
}