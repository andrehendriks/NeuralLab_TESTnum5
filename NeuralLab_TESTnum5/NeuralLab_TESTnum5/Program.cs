using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLab_TESTnum5
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nbegin ANN.\n");
            int numInput2 = 4; // number features
            int numHidden3 = 5;
            int numHidden4 = 5;
            int numOutput2 = 3; // number of classes for Y
            int numRows2 = 1000;
            int seed2 = 1; // gives nice demo

            Console.WriteLine("\nGenerating " + numRows2 +
              " artificial data items with " + numInput2 + " features");
            double[][] allData = MakeAllData(numInput2, numHidden3, numHidden4, numOutput2,
              numRows2, seed2);
            Console.WriteLine("Done");

            //ShowMatrix(allData, allData.Length, 2, true);

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            double[][] trainData2;
            double[][] testData2;
            SplitTrainTest(allData, 0.80, seed2, out trainData2, out testData2);
            Console.WriteLine("Done\n");

            Console.WriteLine("Training data:");
            ShowMatrix(trainData2, 4, 2, true);
            Console.WriteLine("Test data:");
            ShowMatrix(testData2, 4, 2, true);

            Console.WriteLine("Creating a " + numInput2 + "-" + numHidden3 +
              "-" + numHidden4 + "-" + numOutput2 + " neural network");
            NeuralNetwork1 nn = new NeuralNetwork1(numInput2, numHidden3, numHidden4, numOutput2, numRows2, seed2);

            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F2"));

            Console.WriteLine("\nStarting training");
            double[] weights = nn.Train(trainData2, maxEpochs, learnRate, momentum);
            Console.WriteLine("Done");
            Console.WriteLine("\nFinal neural network model weights and biases:\n");
            //            ShowVector(weights, 2, 10, true);

            //double[] y = nn.ComputeOutputs(new double[] { 1.0, 2.0, 3.0, 4.0 });
            //ShowVector(y, 3, 3, true);

            double trainAcc = nn.Accuracy(trainData2);
            Console.WriteLine("\nFinal accuracy on training data = " +
              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData2);
            Console.WriteLine("Final accuracy on test data     = " +
              testAcc.ToString("F4"));



            Console.WriteLine("\nBegin van Vector.\n");
            double[][] trainData = new double[4][];
            trainData[0] = new double[] { 2.0, 2.0 };  // data cannot be perfectly classified
            trainData[1] = new double[] { 3.5, -4.0 };
            trainData[2] = new double[] { 4.0, -5.5 };
            trainData[3] = new double[] { 4.5, -2.0 };

            int[] Y = new int[4] { 0, 1, 1, 1 };

            Console.WriteLine("\nTraining data: \n");
            ShowTrainData(trainData, Y);

            double[] weights2 = null;
            double bias = 0.0;
            double alpha = 0.001;
            int maxEpochs2 = 500;

            Console.Write("\nSetting learning rate to " + alpha.ToString("F3"));
            Console.WriteLine(" and maxEpochs to " + maxEpochs2);

            Console.WriteLine("\nBeginning training the perceptron");
            Train(trainData, alpha, maxEpochs2, Y, out weights2, out bias);
            Console.WriteLine("Training complete");

            Console.WriteLine("\nBest percetron weights found: ");
            ShowVector(weights2, 4);
            Console.WriteLine("\nBest perceptron bias found = " + bias.ToString("F4"));

            double acc = Accuracy(trainData, weights2, bias, Y);
            Console.Write("\nAccuracy of the perceptron on the training data = ");
            Console.WriteLine(acc.ToString("F2"));
            Console.WriteLine("\nDone\n");
            Console.ReadLine();

        } //Main

        public static void ShowMatrix(double[][] matrix, int numRows,
  int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices == true)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }

        public static void ShowVector(double[] vector, int decimals,
          int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        static double[][] MakeAllData(int numInput2, int numHidden3, int numHidden4, int numOutput2, int numRows2, int seed2)
        {
            Random rnd = new Random(seed2);
            int numWeights = (numInput2 * numHidden3) + numHidden4 +
              (numHidden4 * numOutput2) + numOutput2;
            double[] weights = new double[numWeights]; // actually weights & biases
            for (int i = 0; i < numWeights; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to 10.0]

            Console.WriteLine("Generating weights and biases:");
            ShowVector(weights, 2, 10, true);

            double[][] result = new double[numRows2][]; // allocate return-result
            for (int i = 0; i < numRows2; ++i)
                result[i] = new double[numInput2 + numOutput2]; // 1-of-N in last column

            NeuralNetwork1 gnn =
              new NeuralNetwork1(numInput2, numHidden3, numHidden4, numOutput2, numRows2, seed2); // generating NN
            gnn.SetWeights(weights);

            for (int r = 0; r < numRows2; ++r) // for each row
            {
                // generate random inputs
                double[] inputs = new double[numInput2];
                for (int i = 0; i < numInput2; ++i)
                    inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to -10.0]

                // compute outputs
                double[] outputs = gnn.ComputeOutputs(inputs);

                // translate outputs to 1-of-N
                double[] oneOfN = new double[numOutput2]; // all 0.0

                int maxIndex = 0;
                double maxValue = outputs[0];
                for (int i = 0; i < numOutput2; ++i)
                {
                    if (outputs[i] > maxValue)
                    {
                        maxIndex = i;
                        maxValue = outputs[i];
                    }
                }
                oneOfN[maxIndex] = 1.0;

                // place inputs and 1-of-N output values into curr row
                int c = 0; // column into result[][]
                for (int i = 0; i < numInput2; ++i) // inputs
                    result[r][c++] = inputs[i];
                for (int i = 0; i < numOutput2; ++i) // outputs
                    result[r][c++] = oneOfN[i];
            } // each row
            return result;
        } // MakeAllData

        static void SplitTrainTest(double[][] allData, double trainPct, int seed2, out double[][] trainData2, out double[][] testData2)
        {
            Random rnd = new Random(seed2);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData2 = new double[numTrainRows][];
            testData2 = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData2[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData2[i] = copy[i + numTrainRows];
        } // SplitTrainTest


        static int ComputeOutput(double[] data, double[] weights2, double bias) // -1 ot +1
        {
            double result = 0.0;
            for (int j = 0; j < data.Length; ++j)
                result += data[j] * weights2[j];
            result += bias;
            return Activation(result);
        }

        static int Activation(double x)
        {
            if (x >= 0.0) return +1;
            else return -1;
        }

        static double Accuracy(double[][] trainData, double[] weights2, double bias, int[] Y)
        {
            int numCorrect = 0;
            int numWrong = 0;
            for (int i = 0; i < trainData.Length; ++i)
            {
                int output = ComputeOutput(trainData[i], weights2, bias);
                if (output == Y[i]) ++numCorrect;
                else ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        static double TotalError(double[][] trainData, double[] weights2, double bias, int[] Y)
        {
            // sum of squared deviations, before activation, of all training data
            double totErr = 0.0;
            for (int i = 0; i < trainData.Length; ++i)
                totErr += Error(trainData[i], weights2, bias, Y[i]);
            return totErr;
        }

        static double Error(double[] data, double[] weights2, double bias, int Y)
        {
            // error for a single training data
            double sum = 0.0;
            for (int j = 0; j < data.Length; ++j)
                sum += data[j] * weights2[j];
            sum += bias;
            return 0.5 * (sum - Y) * (sum - Y);
        }

        static void Train(double[][] trainData, double alpha, int maxEpochs2, int[] Y, out double[] weights2, out double bias)
        {
            int numWeights = trainData[0].Length;

            double[] bestWeights = new double[numWeights];  // best weights found, return value 
            weights2 = new double[numWeights]; // working values (initially 0.0)
            double bestBias = 0.0;
            bias = 0.01;  // working value (initial small arbitrary value)
            double bestError = double.MaxValue;
            int epoch = 0;

            while (epoch < maxEpochs2)
            {
                for (int i = 0; i < trainData.Length; ++i)  // each input
                {
                    int output = ComputeOutput(trainData[i], weights2, bias);
                    int desired = Y[i];  // -1 or +1

                    if (output != desired)  // misclassification so adjust weights and bias
                    {
                        double delta = desired - output;  // how far off are we?
                        for (int j = 0; j < numWeights; ++j)
                            weights2[j] = weights2[j] + (alpha * delta * trainData[i][j]);

                        bias = bias + (alpha * delta);

                        // new best?
                        double totalError = TotalError(trainData, weights2, bias, Y);
                        if (totalError < bestError)
                        {
                            bestError = totalError;
                            Array.Copy(weights2, bestWeights, weights2.Length);
                            bestBias = bias;
                        }
                    }
                }
                ++epoch;
            } // while

            Array.Copy(bestWeights, weights2, bestWeights.Length);
            bias = bestBias;
            return;
        }

        static void ShowVector(double[] vector, int decimals)
        {
            for (int i = 0; i < vector.Length; ++i)
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            Console.WriteLine("");
        }

        static void ShowTrainData(double[][] trainData, int[] Y)
        {
            for (int i = 0; i < trainData.Length; ++i)
            {
                Console.Write("[" + i.ToString().PadLeft(2, ' ') + "]  ");
                for (int j = 0; j < trainData[i].Length; ++j)
                {
                    Console.Write(trainData[i][j].ToString("F1").PadLeft(6, ' '));
                }
                Console.WriteLine("  ->  " + Y[i].ToString("+0;-0"));
            }
        }

    } // class
} // namespace
