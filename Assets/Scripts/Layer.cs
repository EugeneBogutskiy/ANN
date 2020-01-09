using System.Collections.Generic;

public class Layer 
{
    public int numNeurons; // how many neurons in layer 
    public List<Neuron> neurons = new List<Neuron>(); // we store each neuron in this list

    public Layer(int nNeurons, int numNeuronInputs) // numNeuronInputs - number of inputs in a previos layer 
    {
        numNeurons = nNeurons;
        for (int i = 0; i < nNeurons; i++)
        {
            neurons.Add(new Neuron(numNeuronInputs));
        }
    }
}
