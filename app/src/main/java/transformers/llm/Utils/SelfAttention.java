package transformers.llm.Utils;

import java.util.Random;

public class SelfAttention {
    private double[][] Wq, Wk, Wv, Wo;
    private int inputDim, outputDim;

    public SelfAttention(int inputDim, int outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.Wq = randomMatrix(inputDim, outputDim);
        this.Wk = randomMatrix(inputDim, outputDim);
        this.Wv = randomMatrix(inputDim, outputDim);
        this.Wo = randomMatrix(outputDim, inputDim);
    }

    public double[][] forward(double[][] input) {
        double[][] Q = MatrixUtils.dot(input, Wq);
        double[][] K = MatrixUtils.dot(input, Wk);
        double[][] V = MatrixUtils.dot(input, Wv);

        double[][] attended = ScaledDotProductAttention.compute(Q, K, V);
        return MatrixUtils.dot(attended, Wo); // Output projection
    }

    private double[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = rand.nextGaussian() * 0.02; // small values
        return m;
    }
}
