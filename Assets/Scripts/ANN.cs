using System.Collections.Generic;
using UnityEngine;

public class ANN // all work is here
{
    public int numInputs; // how many inputs has NN
    public int numOutputs; // how many outputs has NN
    public int numHidden; // how many hidden layers has NN
    public int numNPerHidden; // how many neurons do you want to have in each hidden layer
    public double alpha; // how fast your NN is going to learn, it's look like batch in %% (1.0 = 100%)
    List<Layer> layers = new List<Layer>();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="nI">number of inputs NN</param>
    /// <param name="nO">number of outputs NN</param>
    /// <param name="nH">number of hidden layers NN</param>
    /// <param name="nPH">neurons per each hidden layer NN</param>
    /// <param name="a">alpha - full training set 1.0, 0.8 - 80%</param>
    public ANN(int nI, int nO, int nH, int nPH, double a)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;

        if (numHidden > 0) // if we have hidden layer(s)
        {
            layers.Add(new Layer(numNPerHidden, numInputs)); // numInputs - how many inputs on each neuron in this layer
            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }
            layers.Add(new Layer(numOutputs, numNPerHidden));
        }
        else
        {
            layers.Add(new Layer(numOutputs, numInputs)); // if we do not have any hidden layers
        }
    }
    /// <summary>
    /// here we train our NN and recieve results from List of double
    /// </summary>
    /// <param name="inputValues">values from dataset</param>
    /// <param name="desiredOutputs">correct values from dataset</param>
    /// <returns></returns>
    public List<double> Go(List<double> inputValues, List<double> desiredOutput)  // here we train our NN with inputs data and data desired results
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);
        for (int i = 0; i < numHidden + 1; i++)
        {
            if (i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                double N = 0;
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }

                N -= layers[i].neurons[j].bias;

                if (i == numHidden)
                    layers[i].neurons[j].output = ActivationFunctionOutputLayer(N); //if this is an output layer we use this activation function
                else
                    layers[i].neurons[j].output = ActivationFunction(N);

                outputs.Add(layers[i].neurons[j].output);
            }
        }

        UpdateWeights(outputs, desiredOutput);

        return outputs;

    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        for (int i = numHidden; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                if (i == numHidden)
                {
                    error = desiredOutput[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                    //errorGradient calculated with Delta Rule
                }
                else
                {
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    double errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].numNeurons; p++)
                    {
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    if (i == numHidden)
                    {
                        error = desiredOutput[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }


    double ActivationFunction(double value) // choose an activation function
    {
        //return Sigmoid(value);
        //return Step(value);
        //return TanH(value);
        //return ReLU(value);
        return LeakyReLU(value);
    }

    double ActivationFunctionOutputLayer(double value) // choose an activation function for output layer
    {
        return Sigmoid(value);
        //return Step(value);
        //return TanH(value);
        //return ReLU(value);
        //return LeakyReLU(value);
    }

    double Sigmoid(double value) //logistic soft step
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
    }

    double TanH(double value)
    {
        double k = (double)System.Math.Exp(-2 * value);
        return (2 / (1.0f + k)) - 1;

        //return (2 * (Sigmoid(2 * value)) - 1);
    }

    double Step(double value) //binary step
    {
        if (value < 0) return 0;
        else return 1;
    }

    double ReLU(double value)
    {
        if (value < 0) return 0;
        else return value;
    }

    double LeakyReLU(double value)
    {
        if (value < 0) return value * 0.01;
        else return value;
    }
}
