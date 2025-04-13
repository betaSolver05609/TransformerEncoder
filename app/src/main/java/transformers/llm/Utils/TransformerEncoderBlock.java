package transformers.llm.Utils;

public class TransformerEncoderBlock {
    private SelfAttention selfAttention;
    private FeedForward feedForward;
    private LayerNorm norm1;
    private LayerNorm norm2;

    public TransformerEncoderBlock(int inputDim, int hiddenDim) {
        this.selfAttention = new SelfAttention(inputDim, inputDim);
        this.feedForward = new FeedForward(inputDim, hiddenDim);
        this.norm1 = new LayerNorm(inputDim);
        this.norm2 = new LayerNorm(inputDim);
    }

    public double[][] forward(double[][] input) {
        // Step 1: Self Attention
        double[][] attnOutput = selfAttention.forward(input);
        double[][] attnAdded = MatrixUtils.add(input, attnOutput); // Residual
        double[][] normed1 = norm1.forward(attnAdded); // Layer Norm

        // Step 2: Feedforward
        double[][] ffOutput = feedForward.forward(normed1);
        double[][] ffAdded = MatrixUtils.add(normed1, ffOutput); // Residual
        double[][] normed2 = norm2.forward(ffAdded); // Layer Norm

        return normed2;
    }
}
