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

Why do we split the corpus into sequences instead of training on the whole text?


**Your answer:**
"""

part1_q2 = r"""

Question 2

How is it possible that the generated text clearly shows memory longer than the sequence length?


**Your answer:**
"""

part1_q3 = r"""

Question 3

Why are we not shuffling the order of batches when training?


**Your answer:**
"""

part1_q4 = r"""

Question 4

1. Why do we lower the temperature for sampling (compared to the default of 1.0)?
2. What happens when the temperature is very high and why?
3. What happens when the temperature is very low and why?

**Your answer:**
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
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
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
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
