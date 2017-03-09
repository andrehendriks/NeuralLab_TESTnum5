using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLab_TESTnum5
{
    internal class NeuralNetwork1
    {
        private int numHidden3;
        private int numHidden4;
        private int numInput2;
        private int numOutput2;
        private int numRows2;
        private int seed2;
        private Random rnd;
        public NeuralNetwork1(int numInput2, int numHidden3, int numHidden4, int numOutput2, int numRows2, int seed2)
        {
            this.numInput2 = numInput2;
            this.numHidden3 = numHidden3;
            this.numHidden4 = numHidden4;
            this.numOutput2 = numOutput2;
            this.numRows2 = numRows2;
            this.seed2 = seed2;
            this.rnd = new Random(0);
            this.InitializeWeights(); // all weights and biases
        }

        private static double[][] MakeMatrix(int rows, int cols, double v) // helper for ctor, Train
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }

        //private static double[][] MakeMatrixRandom(int rows,
        //  int cols, int seed) // helper for ctor, Train
        //{
        //  Random rnd = new Random(seed);
        //  double hi = 0.01;
        //  double lo = -0.01;
        //  double[][] result = new double[rows][];
        //  for (int r = 0; r < result.Length; ++r)
        //    result[r] = new double[cols];
        //  for (int i = 0; i < rows; ++i)
        //    for (int j = 0; j < cols; ++j)
        //      result[i][j] = (hi - lo) * rnd.NextDouble() + lo;
        //  return result;
        //}

        private void InitializeWeights() // helper for ctor
        {
            // initialize weights and biases to small random values
            int numWeights = (numInput2 * numHidden3) +
              (numHidden3 * numOutput2) + numHidden4 + numOutput2;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }

        public void SetWeights(double[] weights)
        {
            // copy serialized weights and biases in weights[] array
            // to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (numInput2 * numHidden3) +
              (numHidden3 * numOutput2) + numHidden4 + numOutput2;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput2 * numHidden3) +
              (numHidden3 * numOutput2) + numHidden4 + numOutput2;
            double[] result = new double[numWeights];
            int k = 0;
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden4]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput2]; // output nodes sums

            double[] retResult = new double[numOutput2]; // could define a GetOutputs 
//            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // does all output nodes at once so scale
            // doesn't have to be re-computed each time

            double sum = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                sum += Math.Exp(oSums[i]);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / sum;

            return result; // now scaled so that xi sum to 1.0
        }

        public double[] Train(double[][] trainData2, int maxEpochs, double learnRate, double momentum)
        {
            // train using back-prop
            // back-prop specific arrays
            double[][] hoGrads = MakeMatrix(numHidden4, numOutput2, 0.0); // hidden-to-output weight gradients
            double[] obGrads = new double[numOutput2];                   // output bias gradients

            double[][] ihGrads = MakeMatrix(numInput2, numHidden3, 0.0);  // input-to-hidden weight gradients
            double[] hbGrads = new double[numHidden3];                   // hidden bias gradients

            double[] oSignals = new double[numOutput2];                  // local gradient output signals - gradients w/o associated input terms
            double[] hSignals = new double[numHidden4];                  // local gradient hidden node signals

            // back-prop momentum specific arrays 
            double[][] ihPrevWeightsDelta = MakeMatrix(numInput2, numHidden3, 0.0);
            double[] hPrevBiasesDelta = new double[numHidden3];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden4, numOutput2, 0.0);
            double[] oPrevBiasesDelta = new double[numOutput2];

            int epoch = 0;
            double[] xValues = new double[numInput2]; // inputs
            double[] tValues = new double[numOutput2]; // target values
            double derivative = 0.0;
            double errorSignal = 0.0;

            int[] sequence = new int[trainData2.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxEpochs / 10; // interval to check error
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 && epoch < maxEpochs)
                {
                    double trainErr = Error(trainData2);
                    Console.WriteLine("epoch = " + epoch + "  error = " +
                      trainErr.ToString("F4"));
                    //Console.ReadLine();
                }

                Shuffle(sequence); // visit each training data in random order
                for (int ii = 0; ii < trainData2.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData2[idx], xValues, numInput2);
                    Array.Copy(trainData2[idx], numInput2, tValues, 0, numOutput2);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs 

                    // indices: i = inputs, j = hiddens, k = outputs

                    // 1. compute output node signals (assumes softmax)
                    for (int k = 0; k < numOutput2; ++k)
                    {
                      //  errorSignal = tValues[k] - outputs[k];  // Wikipedia uses (o-t)
                      //  derivative = (1 - outputs[k]) * outputs[k]; // for softmax
                        oSignals[k] = errorSignal * derivative;
                    }

                    // 2. compute hidden-to-output weight gradients using output signals
                    for (int j = 0; j < numHidden4; ++j)
                        for (int k = 0; k < numOutput2; ++k) ;
                         //   hoGrads[j][k] = oSignals[k] * hOutputs[j];

                    // 2b. compute output bias gradients using output signals
                    for (int k = 0; k < numOutput2; ++k)
                        obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

                    // 3. compute hidden node signals
                    for (int j = 0; j < numHidden4; ++j)
                    {
                        //derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // for tanh
                        double sum = 0.0; // need sums of output signals times hidden-to-output weights
                        for (int k = 0; k < numOutput2; ++k)
                        {
                            //sum += oSignals[k] * hoWeights[j][k]; // represents error signal
                        }
                        hSignals[j] = derivative * sum;
                    }

                    // 4. compute input-hidden weight gradients
                    for (int i = 0; i < numInput2; ++i)
                        for (int j = 0; j < numHidden3; ++j) ;
                    // ihGrads[i][j] = hSignals[j] * inputs[i];

                    // 4b. compute hidden node bias gradients
                   // for (int j = 0; j < numHidden3++j) ;
                    //    hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

                    // == update weights and biases

                    // update input-to-hidden weights
                    for (int i = 0; i < numInput2; ++i)
                    {
                        for (int j = 0; j < numHidden3; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
//                            ihWeights[i][j] += delta; // would be -= if (o-t)
//                            ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta; // save for next time
                        }
                    }

                    // update hidden biases
                    for (int j = 0; j < numHidden3; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
//                        hBiases[j] += delta;
//                        hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // update hidden-to-output weights
                    for (int j = 0; j < numHidden4; ++j)
                    {
                        for (int k = 0; k < numOutput2; ++k)
                        {
                            double delta = hoGrads[j][k] * learnRate;
//                            hoWeights[j][k] += delta;
//                            hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // update output node biases
                    for (int k = 0; k < numOutput2; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
//                        oBiases[k] += delta;
//                        oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // each training item

            } // while
            double[] bestWts = GetWeights();
            return bestWts;
        } // Train

        private void Shuffle(int[] sequence) // instance method
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } // Shuffle

        private double Error(double[][] trainData2)
        {
            // average squared error per training item
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput2]; // first numInput values in trainData
            double[] tValues = new double[numOutput2]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData2.Length; ++i)
            {
                Array.Copy(trainData2[i], xValues, numInput2);
                Array.Copy(trainData2[i], numInput2, tValues, 0, numOutput2); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
                for (int j = 0; j < numOutput2; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData2.Length;
        } // MeanSquaredError

        public double Accuracy(double[][] testData2)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput2]; // inputs
            double[] tValues = new double[numOutput2]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData2.Length; ++i)
            {
                Array.Copy(testData2[i], xValues, numInput2); // get x-values
                Array.Copy(testData2[i], numInput2, tValues, 0, numOutput2); // get t-values
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
                int tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }


    } // NeuralNetwork

}

