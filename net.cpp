#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>


using namespace std;
//////////////////////////////////////////////////////////////////////
class TrainingData
{
private:
	ifstream m_trainingDataFile;
public:
	TrainingData(const string fileName);
	bool isEof(void){return m_trainingDataFile.eof();}
	void getTopology(vector<unsigned> &topology);

	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);
};
TrainingData::TrainingData(const string fileName)
{
	m_trainingDataFile.open(fileName.c_str());
}
void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile,line);
	stringstream ss(line);
	ss >> label;

	if(this->isEof() || label.compare("topology:") !=0)
	{
		abort();
	}
	while(!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);		
	}
	return;
}
unsigned TrainingData::getNextInputs(vector<double>&inputVals)
{
	inputVals.clear();
	string line;
	getline(m_trainingDataFile,line);
	stringstream ss(line);

	string label;
	ss >> label;
	if(label.compare("in:")==0)
	{
		double oneValue;
		while(ss>>oneValue)
		{
			inputVals.push_back(oneValue);
		}
	}
	return inputVals.size();
}
unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();
	string line;
	getline(m_trainingDataFile,line);
	stringstream ss(line);

	string label;
	ss >> label;
	if(label.compare("out:")==0)
	{
		double oneValue;
		while(ss>>oneValue)
		{
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}
//////////////////////////////////////////////////////////////////////
struct Connection
{
	double weight;
	double deltaWeight;
};
//////////////////////////////////////////////////////////////////////
class Neuron;

typedef vector<Neuron> Layer;
//////////////////////////////////////////////////////////////////////
class Neuron
{
private:
	static double eta;
	static double alpha;
	unsigned m_myIndex;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	static double randomWeight(){return rand()/double(RAND_MAX);}
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer &nextLayer) const;
	double m_gradient;
public:
	Neuron(unsigned numOutputs,unsigned myIndex);
	void setOutputVal(double val){m_outputVal=val;}
	double getOutputVal(void)const{return m_outputVal;}
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
Neuron::Neuron(unsigned numOutputs,unsigned myIndex)
{
	for(unsigned c = 0;c<numOutputs;c++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight=randomWeight();
	}
	m_myIndex = myIndex;
}
void Neuron::feedForward(const Layer &prevLayer)
{
	double sum=0.0;
	for(unsigned n=0;n<prevLayer.size();n++)
	{
		sum+=prevLayer[n].getOutputVal() *
				prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal=Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x)
{
	return tanh(x);//-1 --- +1
};

double Neuron::transferFunctionDerivative(double x)
{
	return 1.0-x*x;//aproximate
};
void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow=sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}
double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;
	for(unsigned n=0;n<nextLayer.size()-1;n++)
	{
		sum +=m_outputWeights[n].weight *
				nextLayer[n].m_gradient;
	}
	return sum;
}
void Neuron::updateInputWeights(Layer &prevLayer)
{
	for(unsigned n=0;n<prevLayer.size();n++)
	{
		Neuron &neuron=prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		double newDeltaWeight = 
			eta
			*neuron.getOutputVal()
			*m_gradient
				+alpha
					*oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
};
//////////////////////////////////////////////////////////////////////
class Net
{
private:
	vector<Layer>m_layers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double>&inputVals);
	void backProp(const vector<double>&targetVals);
	void getResults(vector<double>&resultVals) const;
	double getRecientAverageError(void)const {return m_recentAverageError;};
};

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum<numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		printf("Layer %d\n",layerNum);
		unsigned numOutputs = (layerNum==topology.size()-1)?0:topology[layerNum+1];
		for(unsigned neuronNum=0 ; neuronNum <= topology[layerNum]; neuronNum++)	//<= for bias
		{
			printf("\tAdd Neuron numOutputs=%d  neuronNum=%d\n",numOutputs,neuronNum);
			m_layers.back().push_back(Neuron(numOutputs , neuronNum));
		}
		m_layers.back().back().setOutputVal(1.0);
		printf("Layer %d neuron %d bias set to 1.0\n",layerNum,m_layers.back().size()-1);
	}
}
void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() -1);
	//load input values
	for(unsigned i=0;i<inputVals.size();i++)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	//Forward propagation
	for(unsigned layerNum = 1;layerNum<m_layers.size();layerNum++)
	{
		Layer &prevLayer=m_layers[layerNum-1];
		for(unsigned n=0;n<m_layers[layerNum].size()-1;n++)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}
void Net::backProp(const vector<double>&targetVals)
{
	//Calculate overall net errors
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	for(unsigned n = 0; n<outputLayer.size()-1;n++)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta*delta;
	}
	m_error/=outputLayer.size()-1;//dont count the bias
	m_error = sqrt(m_error);

	//Recent average measurment
	m_recentAverageError=
		(m_recentAverageError*m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1);

	//Calculate output gradients
	for(unsigned n = 0; n<outputLayer.size()-1;n++)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate gradients on hidden layers
	for(unsigned layerNum=m_layers.size()-2;layerNum>0;layerNum--)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum+1];
		for(unsigned n=0;n<hiddenLayer.size();n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//update weights
	for(unsigned layerNum = m_layers.size()-1;layerNum>0;layerNum--)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum-1];
		for(unsigned n=0;n<layer.size()-1;n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
};
void Net::getResults(vector<double>&resultVals) const
{
	resultVals.clear();
	for(unsigned n=0;n<m_layers.back().size()-1;n++)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void showVectorVals(string label,vector<double> &v)
{
	cout <<label<<" ";
	for(unsigned i=0;i<v.size();i++)
	{
		cout << v[i] << " ";
	}
}

/////////////////////////////////////////////////////////////////////////////
int main()
{

	TrainingData trainData("trainingData.txt");
	vector<unsigned>topology;
	trainData.getTopology(topology);
	Net myNet(topology);
	
	vector<double> inputVals, targetVals, resultsVals;

	int trainingPass = 0;
	for(int p=1;p<2;p++)
	{
		while(!trainData.isEof())
		{
			
			cout<<"Pass "<<trainingPass;
			if(trainData.getNextInputs(inputVals)!=topology[0])
			{
				break;
			}
			showVectorVals(": In:",inputVals);
			myNet.feedForward(inputVals);

			myNet.getResults(resultsVals);
			showVectorVals("Out:",resultsVals);

			trainData.getTargetOutputs(targetVals);
			showVectorVals("T:",targetVals);

			assert(targetVals.size() == topology.back());

			myNet.backProp(targetVals);
			
			cout << "E: "<< myNet.getRecientAverageError() << endl;
		}
		trainingPass++;
	}
	cout <<endl<<"Done"<<endl;
}
