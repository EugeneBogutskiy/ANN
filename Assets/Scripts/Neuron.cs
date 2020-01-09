using System.Collections.Generic;
using UnityEngine;

public class Neuron // it's basically perceptron
{
    public int numInputs; // number of inputs in neuron
    public double bias; 
    public double output; // number outputs of neuron
    public double errorGradient; // 
    public List<double> weights = new List<double>(); // store weights
    public List<double> inputs = new List<double>(); 

    public Neuron(int nInputs)
    {
        bias = Random.Range(-1.0f, 1.0f);
        numInputs = nInputs;
        for (int i = 0; i < nInputs; i++)
        {
            weights.Add(Random.Range(-1.0f, 1.0f)); // we init each weight randomly on each input
        }
    }
}
