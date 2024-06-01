namespace Neurax
{
    public class Perceptron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public double LearningRate { get; private set; }

        public Perceptron(int inputCount, double learningRate = 0.1)
        {
            Weights = new double[inputCount];
            Bias = 0;
            LearningRate = learningRate;
        }

        public int Predict(double[] inputs)
        {
            double sum = Bias;
            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * inputs[i];
            }
            return sum >= 0 ? 1 : -1;
        }

        public void Train(double[] inputs, int target)
        {
            int prediction = Predict(inputs);
            double error = target - prediction;
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] += LearningRate * error * inputs[i];
            }
            Bias += LearningRate * error;
        }
    }


}
