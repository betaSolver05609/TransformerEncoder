package transformers.llm.Utils;

public class MatrixUtils {
    //Dot product of 2 matrices
    public static double[][] dot(double[][] a, double[][] b) {
        int aRows = a.length;
        int aCols = a[0].length;
        int bCols=b[0].length;

        double[][] result=new double[aRows][bCols];

        for(int i=0;i<aRows;i++) {
            for(int j=0;j<bCols; j++) {
                for(int k=0;k<aCols;k++) {
                    result[i][j]+=a[i][k]*b[k][j];
                }
            }
        }

        return result;
    }

    public static double[][] transpose(double[][] m) {
        int rows = m.length, cols = m[0].length;
        double[][] result = new double[cols][rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j][i] = m[i][j];
        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {
    int rows = a.length;
    int cols = a[0].length;

    double[][] result = new double[rows][cols];

    if (b.length == 1 && b[0].length == cols) {
        // Broadcast row vector
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i][j] + b[0][j];
    } else if (b.length == rows && b[0].length == cols) {
        // Element-wise add
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i][j] + b[i][j];
    } else {
        throw new IllegalArgumentException("Incompatible matrix sizes for add.");
    }

    return result;
}

    public static double[][] softmax(double[][] m) {
        int rows = m.length, cols = m[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, m[i][j]);
            }
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.exp(m[i][j] - max);
                sum += result[i][j];
            }
            for (int j = 0; j < cols; j++) {
                result[i][j] /= sum;
            }
        }
        return result;
    }

}
