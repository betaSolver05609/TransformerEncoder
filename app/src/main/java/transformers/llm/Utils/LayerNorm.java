package transformers.llm.Utils;

public class LayerNorm {
    private int dim;
    private double[] gamma;
    private double[] beta;
    private double epsilon = 1e-6;

    public LayerNorm(int dim) {
        this.dim = dim;
        this.gamma = new double[dim];
        this.beta = new double[dim];

        for (int i = 0; i < dim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public double[][] forward(double[][] input) {
        int seqLen = input.length;
        double[][] output = new double[seqLen][dim];

        for (int i = 0; i < seqLen; i++) {
            double[] row = input[i];
            double mean = mean(row);
            double variance = variance(row, mean);

            for (int j = 0; j < dim; j++) {
                output[i][j] = gamma[j] * ((row[j] - mean) / Math.sqrt(variance + epsilon)) + beta[j];
            }
        }

        return output;
    }

    private double mean(double[] row) {
        double sum = 0;
        for (double v : row) sum += v;
        return sum / row.length;
    }

    private double variance(double[] row, double mean) {
        double sum = 0;
        for (double v : row) sum += (v - mean) * (v - mean);
        return sum / row.length;
    }
}
