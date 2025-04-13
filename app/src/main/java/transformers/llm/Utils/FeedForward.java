package transformers.llm.Utils;
import java.util.Random;

public class FeedForward {
    private int inputDim, hiddenDim;
    private double[][] W1, b1, W2, b2;

    public FeedForward(int inputDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.W1 = randomMatrix(inputDim, hiddenDim);
        this.b1 = new double[1][hiddenDim];
        this.W2 = randomMatrix(hiddenDim, inputDim);
        this.b2 = new double[1][inputDim];
    }

    public double[][] forward(double[][] input) {
        double[][] hidden = MatrixUtils.add(MatrixUtils.dot(input, W1), b1);
        relu(hidden);
        double[][] output = MatrixUtils.add(MatrixUtils.dot(hidden, W2), b2);
        return output;
    }

    private void relu(double[][] mat) {
        for (int i = 0; i < mat.length; i++)
            for (int j = 0; j < mat[0].length; j++)
                mat[i][j] = Math.max(0, mat[i][j]);
    }

    private double[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = rand.nextGaussian() * 0.02;
        return m;
    }
}

