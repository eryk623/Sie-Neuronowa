
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuronWebWPF.Logic
{
    class Neuron
    {
        public (int Layer, int Row) NeuronNum { get; private set; }
        public double Output { get; set; }

        public Neuron((int layer, int row) neuronNum)
        {
            NeuronNum = neuronNum;
        }
    }

    class Weight
    {
        public double Value { get; set; }
        public Neuron From { get; private set; }
        public Neuron To { get; private set; }

        public Weight(Neuron from, Neuron to, double value)
        {
            From = from;
            To = to;
            Value = value;
        }
    }

    class NeuralNetwork
    {
        private List<Neuron> neurons;
        private List<Weight> weights;
        private int[] layers;
        private double learningRate;
        private static Random rnd = new Random();

        public NeuralNetwork(int[] layers, double learningRate = 0.1, double weightMin = -5.0, double weightMax = 5.0)
        {
            this.layers = layers;
            this.learningRate = learningRate;
            neurons = new List<Neuron>();
            weights = new List<Weight>();
            CreateNetwork(weightMin, weightMax);
        }

        private void CreateNetwork(double weightMin, double weightMax)
        {
            for (int layer = 0; layer < layers.Length; layer++)
                for (int i = 0; i < layers[layer]; i++)
                    neurons.Add(new Neuron((layer, i)));

            var bias = new Neuron((-1, -1)) { Output = 1.0 };
            neurons.Add(bias);

            for (int layer = 0; layer < layers.Length - 1; layer++)
            {
                var fromLayer = neurons.Where(n => n.NeuronNum.Layer == layer).ToList();
                var toLayer = neurons.Where(n => n.NeuronNum.Layer == layer + 1).ToList();

                foreach (var to in toLayer)
                {
                    weights.Add(new Weight(bias, to, DrawWeight(weightMin, weightMax)));
                    foreach (var from in fromLayer)
                        weights.Add(new Weight(from, to, DrawWeight(weightMin, weightMax)));
                }
            }
        }

        private static double DrawWeight(double min, double max) => rnd.NextDouble() * (max - min) + min;
        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private static double SigmoidDerivative(double output) => output * (1 - output);

        private void PropagationForward(double[] inputs)
        {
            foreach (var n in neurons.Where(n => n.NeuronNum.Layer == 0))
                n.Output = inputs[n.NeuronNum.Row];

            foreach (var n in neurons.Where(n => n.NeuronNum.Layer > 0))
            {
                double sum = weights.Where(w => w.To == n).Sum(w => w.From.Output * w.Value);
                n.Output = Sigmoid(sum);
            }
        }

        private void PropagationBackward(double[] expected)
        {
            int maxLayer = layers.Length - 1;
            var deltas = new Dictionary<Neuron, double>();
            var outputNeurons = neurons.Where(n => n.NeuronNum.Layer == maxLayer).OrderBy(n => n.NeuronNum.Row).ToList();

            for (int i = 0; i < outputNeurons.Count; i++)
            {
                var n = outputNeurons[i];
                double error = expected[i] - n.Output;
                deltas[n] = error * SigmoidDerivative(n.Output);
            }

            for (int layer = maxLayer - 1; layer > 0; layer--)
            {
                var hidden = neurons.Where(n => n.NeuronNum.Layer == layer);
                foreach (var n in hidden)
                {
                    double sum = weights.Where(w => w.From == n && deltas.ContainsKey(w.To)).Sum(w => w.Value * deltas[w.To]);
                    deltas[n] = sum * SigmoidDerivative(n.Output);
                }
            }

            foreach (var w in weights)
                if (deltas.ContainsKey(w.To))
                    w.Value += learningRate * w.From.Output * deltas[w.To];
        }

        public void Train(List<(double[] inputs, double[] targets)> data, int epochs)
        {
            for (int e = 0; e < epochs; e++)
                foreach (var (inputs, targets) in data)
                {
                    PropagationForward(inputs);
                    PropagationBackward(targets);
                }
        }

        public double[] Result(double[] inputs)
        {
            PropagationForward(inputs);
            return neurons
                .Where(n => n.NeuronNum.Layer == layers.Length - 1)
                .OrderBy(n => n.NeuronNum.Row)
                .Select(n => n.Output)
                .ToArray();
        }
    }
}
