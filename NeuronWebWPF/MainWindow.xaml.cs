
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using NeuronWebWPF.Logic;

namespace NeuronWebWPF
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void TrainNetwork_Click(object sender, RoutedEventArgs e)
        {
            var selectedTask = TaskComboBox.SelectedIndex;

            List<(double[], double[])> data;
            int inputCount, outputCount;

            switch (selectedTask)
            {
                case 0:
                    data = new List<(double[], double[])>
                    {
                        (new double[]{0,0}, new double[]{0}),
                        (new double[]{0,1}, new double[]{1}),
                        (new double[]{1,0}, new double[]{1}),
                        (new double[]{1,1}, new double[]{0})
                    };
                    inputCount = 2;
                    outputCount = 1;
                    break;
                case 1:
                    data = new List<(double[], double[])>
                    {
                        (new double[]{0,0}, new double[]{0,1}),
                        (new double[]{0,1}, new double[]{1,0}),
                        (new double[]{1,0}, new double[]{1,0}),
                        (new double[]{1,1}, new double[]{0,0})
                    };
                    inputCount = 2;
                    outputCount = 2;
                    break;
                case 2:
                    data = new List<(double[], double[])>
                    {
                        (new double[]{0,0,0}, new double[]{0,0}),
                        (new double[]{0,0,1}, new double[]{1,0}),
                        (new double[]{0,1,0}, new double[]{1,0}),
                        (new double[]{0,1,1}, new double[]{0,1}),
                        (new double[]{1,0,0}, new double[]{1,0}),
                        (new double[]{1,0,1}, new double[]{0,1}),
                        (new double[]{1,1,0}, new double[]{0,1}),
                        (new double[]{1,1,1}, new double[]{1,1})
                    };
                    inputCount = 3;
                    outputCount = 2;
                    break;
                default:
                    MessageBox.Show("Wybierz poprawne zadanie.");
                    return;
            }

            if (!int.TryParse(HiddenLayersBox.Text, out int hiddenLayerCount) || hiddenLayerCount < 0)
            {
                MessageBox.Show("Nieprawidłowa liczba warstw ukrytych.");
                return;
            }

            int[] neuronsPerLayer;
            try
            {
                neuronsPerLayer = NeuronsPerLayerBox.Text.Split(',')
                    .Select(s => int.Parse(s.Trim()))
                    .ToArray();

                if (neuronsPerLayer.Length != hiddenLayerCount)
                {
                    MessageBox.Show("Liczba neuronów nie pasuje do liczby warstw.");
                    return;
                }
            }
            catch
            {
                MessageBox.Show("Nieprawidłowe dane neuronów.");
                return;
            }

            var layers = new[] { inputCount }
                .Concat(neuronsPerLayer)
                .Concat(new[] { outputCount })
                .ToArray();

            var network = new NeuralNetwork(layers, learningRate: 0.1);
            network.Train(data, 50000);

            StringBuilder result = new StringBuilder("Wyniki po treningu:\n");
            foreach (var (input, expected) in data)
            {
                var output = network.Result(input);
                result.AppendLine($"Wejście: [{string.Join(",", input)}] " +
                                  $"Oczekiwane: [{string.Join(",", expected)}] " +
                                  $"Wynik: [{string.Join(",", output.Select(x => x.ToString("F3")))}]");
            }

            OutputBox.Text = result.ToString();
        }
    }
}
