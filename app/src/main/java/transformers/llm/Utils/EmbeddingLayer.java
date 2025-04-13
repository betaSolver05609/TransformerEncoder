package transformers.llm.Utils;

public class EmbeddingLayer {
    private double[][] embeddings;

    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this.embeddings = new double[vocabSize][embeddingDim];
        // Initialize randomly
        for (int i = 0; i < vocabSize; i++)
            for (int j = 0; j < embeddingDim; j++)
                embeddings[i][j] = Math.random() - 0.5;
    }

    public double[][] forward(int[] tokenIds) {
        double[][] output = new double[tokenIds.length][embeddings[0].length];
        for (int i = 0; i < tokenIds.length; i++) {
            output[i] = embeddings[tokenIds[i]];
        }
        return output;
    }
}
