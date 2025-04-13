package transformers.llm.Utils;

public class ScaledDotProductAttention {

    public static double[][] compute(double[][] q, double[][] k, double[][] v) {
        int dk = k[0].length;
        //QK^T
        double[][] kt = MatrixUtils.transpose(k);
        double[][] scores=MatrixUtils.dot(q, kt);

        double scale = Math.sqrt(dk);
        for (int i = 0; i < scores.length; i++) {
            for (int j = 0; j < scores[0].length; j++) {
                scores[i][j] /= scale;
            }
        }

        double[][] weights = MatrixUtils.softmax(scores);

        return MatrixUtils.dot(weights, v);
    }

}
