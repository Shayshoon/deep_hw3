r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=128,
        seq_len=64,
        h_dim=256,
        n_layers=3,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=2,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "DUCHESS OF THE CS FACULTY:"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""

Question 1

**Your answer:**
this necessary for several key reasons:
1. Memory Constraints: During training, the Backpropagation Through Time algorithm requires storing all intermediate activations for every time-step to compute gradients. If we were to input the entire corpus as a single sequence, the memory requirements would exceed the capacity of any modern GPU.

2. Vanishing/Exploding Gradients: RNNs suffer from stability issues when sequences are too long. As the gradient is propagated back through many time-steps, it tends to either vanish or explode due to repeated matrix multiplications. Shorter sequences help maintain stable gradient flow and more effective learning.

3. Computational Efficiency: Training on smaller sequences allows us to utilize Mini-batch Gradient Descent. By organizing the data into batches of sequences, we can perform parallel computations on the GPU and update the model's weights more frequently, leading to faster and more robust convergence.
"""

part1_q2 = r"""

Question 2

**Your answer:**
During training and text generation, the hidden state of the RNN from the end of one sequence is passed as the initial hidden state for the next sequence. This allows the model to propagate information across sequence boundaries. Consequently, the hidden state acts as a continuous summary of the entire history of the text processed so far, enabling the model to maintain context.
"""

part1_q3 = r"""

Question 3

**Your answer:**
We do not shuffle the order of batches during training because RNNs are designed to learn dependencies within a sequence. In this specific implementation, we rely on the hidden state being passed from one batch to the next to maintain a continuous "memory" of the text. 

If we were to shuffle the batches, the temporal continuity of the corpus would be broken. The final hidden state of Batch N would no longer correspond to the beginning of Batch N+1 in the original text, making it impossible for the model to learn long-term dependencies that span across different batches. Maintaining the original order ensures that the model can learn from the global structure of the corpus rather than just isolated fragments.
"""

part1_q4 = r"""

Question 4

**Your answer:**
1. We lower the temperature to make the model's predictions more confident and less "random." By reducing $T$ below 1.0, we sharpen the probability distribution, giving much higher weight to the most likely next characters and suppressing the unlikely ones. This results in generated text that is more coherent, grammatically correct, and structurally stable.

2. When the temperature is very high ($T \to \infty$), the probability distribution becomes increasingly uniform. This means all possible characters (even the highly unlikely ones) get almost equal probability of being sampled. As a result, the generated text becomes chaotic, filled with spelling mistakes, and lacks any meaningful structure or semantic sense.

3. When the temperature is very low ($T \to 0$), the distribution becomes extremely "sharp" or "greedy." The character with the highest logit will receive a probability near 1, while all others drop to near 0. This makes the sampling process deterministic. While the output will be very "safe" and follow rules, it often becomes repetitive, stuck in infinite loops of the same words or phrases, and lacks creativity.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=8, h_dim=256, z_dim=32, x_sigma2=0.001, learn_rate=0.0001, betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The hyperparameter $\sigma^2$ represents the variance of the Gaussian distribution that we assume models the output data given the hidden representation $z$.

1. Low values of $\sigma^2$: A small variance means we are assuming a very "sharp" distribution around the reconstructed mean. Mathematically, this increases the weight of the reconstruction term relative to the KL divergence term in the loss function. This forces the model to prioritize high-fidelity reconstructions, making it very sensitive to small pixel-wise differences, but it may lead to a less regularized latent space.

2. High values of $\sigma^2$: A large variance means we are assuming a "flatter" distribution, which effectively reduces the importance of the reconstruction error. This allows the KL divergence term to dominate. As a result, the model will focus more on making the latent space follow the prior, potentially leading to smoother interpolations but at the cost of blurry or less accurate reconstructions.
"""

part2_q2 = r"""
**Your answer:**
1. Purpose of the VAE loss terms:
   - Reconstruction Loss: This term ensures that the decoder can accurately reconstruct the input data from the hidden representation $z$. Its primary purpose is information preservation, forcing the model to retain the essential features of the data.
   - KL Divergence Loss: This term acts as a regularizer by forcing the learned hidden distribution $q(z|x)$ to be as close as possible to a prior distribution . Its purpose is to organize the hidden space and prevent it from becoming overfitted or disjointed.

2. Effect on the hidden-space distribution:
   The KL loss term acts as a "contractive" force that pushes the hidden representations of different inputs toward the center of the space and encourages them to have a unit variance. Without the KL term, the encoder might map each input to a far-off, isolated point in space. The KL term ensures that these distributions are compact, centered, and have significant overlap.

3. Benefit of this effect:
   The main benefit is the creation of a continuous and smooth hidden space. This allows the VAE to be a generative model: because there are no "gaps" in the distribution, we can sample any random point from the hidden space and the decoder will be able to generate a realistic, novel output. It also enables smooth interpolation, where moving between two points in the hidden space results in a gradual semantic transformation of the generated data.
"""

part2_q3 = r"""
**Your answer:**
In generative modeling, our ultimate goal is to learn a model that captures the true distribution of the data $X$. Maximizing the evidence distribution, $p(X)$, is the standard approach known as MLE. By maximizing $p(X)$, we ensure that the model assigns high probability to the observed data, meaning it has learned to represent and generate realistic samples.

In a VAE, $p(X)$ is computationally intractable because it requires integrating over all possible hidden variables $z$ ($p(X) = \int p(X|z)p(z)dz$). Therefore, we start with the objective of maximizing $\log p(X)$ and derive a lower bound for it, known as the ELBO. Maximizing this lower bound is a proxy for maximizing the evidence itself, which leads to the dual objective of minimizing reconstruction error while regularizing the hidden space.
"""

part2_q4 = r"""
**Your answer:**
We model the log of the variance, $v = \log(\sigma^2)$, instead of the variance itself for two primary reasons related to optimization and numerical stability:

1. Unconstrained Optimization: Variance is strictly non-negative ($\sigma^2 > 0$). If the network were to output $\sigma^2$ directly, we would need to apply a specific activation function to enforce positivity. However, the log of the variance can take any real value ($v \in \mathbb{R}$). This allows the final layer of the encoder to be a simple linear layer without constraints, which is easier for the optimizer to train.

2. Numerical Stability: In the VAE objective function, specifically the KL divergence term, we are required to calculate $\log(\sigma^2)$. If the network directly predicts the log-variance, we avoid potential numerical issues that occur when calculating the logarithm of a value very close to zero.
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim=16,       # Must be > 0 and divisible by num_heads
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        window_size=4,      # Must be even
        droupout=0.1,
        lr=0.001,
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
In the first layer, each token attends only to its immediate neighbors within a window of size $w$. However, in the second layer, those neighbors have already integrated information from their own respective windows. Therefore, when a token in the third layer attends to its neighbors from the second layer, it is indirectly receiving information from a wider range of the original input. 

For example a window with size of $w$ and $L$ layers, the effective context size at the top layer expands to approximately $L \times w$. This allows the model to capture long-range dependencies across the entire sequence even though each individual attention operation is strictly local.
"""

part3_q2 = r"""
**Your answer:**
Global + Sliding Window Attention
To maintain linear complexity while capturing global context, I propose a hybrid attention pattern where a small, fixed number of tokens are designated as "Global Tokens."

1. The Pattern: Most tokens attend only to their local window of size $w$. However, specific tokens are granted "Global Attention." These tokens attend to all other tokens in the sequence, and conversely, every token in the sequence attends to them.

2. Time Complexity Analysis: If we have $n$ tokens and $k$ global tokens (where $k$ is a small constant), the complexity is $O(n \cdot w + n \cdot k)$. Since $w$ and $k$ are constants independent of $n$, the overall complexity remains $O(n)$, which is significantly more efficient than the original $O(n^2)$.

3. Information Sharing: Global information is shared very efficiently. Instead of information "traveling" layer-by-layer through local windows, any token can receive information from any other part of the sequence in just two steps: Token A $\to$ Global Token $\to$ Token B. This allows the model to capture long-range dependencies in the very first layers.

4. Limitations: The primary limitation is the "bottleneck" effect. Since only a few tokens handle the global context, they must learn to compress and represent the most relevant information from the entire sequence. If the sequence is extremely long and complex, these few tokens might not be able to capture all necessary global nuances.
"""

# ==============
