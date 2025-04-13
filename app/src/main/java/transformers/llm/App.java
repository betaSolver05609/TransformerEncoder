package transformers.llm;

import transformers.llm.Utils.EmbeddingLayer;
import transformers.llm.Utils.TransformerEncoderBlock;

public class App {
    public static void main(String[] args) {
        // 1. Tokenized input: "the cat sat"
        int[] tokenIds = {0, 1, 2};  // pretend vocab

        // 2. Embedding
        EmbeddingLayer embedding = new EmbeddingLayer(10, 4); // vocab size=10, embedding dim=4
        double[][] embeddedInput = embedding.forward(tokenIds);

        System.out.println("🔹 Embedded Input:");
        printMatrix(embeddedInput);

        // 3. Transformer block
        TransformerEncoderBlock encoder = new TransformerEncoderBlock(4, 8); // modelDim=4, ffDim=8
        double[][] output = encoder.forward(embeddedInput);

        System.out.println("\n🔸 Output after TransformerEncoder:");
        printMatrix(output);
    }

    private static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double val : row) {
                System.out.printf("%.4f ", val);
            }
            System.out.println();
        }
    }
}
