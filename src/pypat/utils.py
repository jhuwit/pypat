## Model is dependent on these functions,
## Make sure to run this block before you create your model

import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input

# Modified Transformer Block to output attention weights with explicit layer names (otherwise the same as the )
def TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.1, name_prefix="encoder"):
    input_layer = layers.Input(shape=(None, embed_dim), name=f"{name_prefix}_input")
    attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, name=f"{name_prefix}_attention")
    attention_output, attention_weights = attention_layer(input_layer, input_layer, return_attention_scores=True)
    attention_output = layers.Dropout(rate, name=f"{name_prefix}_dropout")(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm1")(input_layer + attention_output)
    ff_output = layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ff1")(out1)
    ff_output = layers.Dense(embed_dim, name=f"{name_prefix}_ff2")(ff_output)
    ff_output = layers.Dropout(rate, name=f"{name_prefix}_dropout2")(ff_output)
    final_output = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm2")(out1 + ff_output)
    return models.Model(inputs=input_layer, outputs=[final_output, attention_weights], name=f"{name_prefix}_transformer")

# Sine/Cosine positional embeddings
def get_positional_embeddings(num_patches, embed_dim):
    position = tf.range(num_patches, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / embed_dim))
    pos_embeddings = tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1)
    return pos_embeddings


def load_encoder_model(input_size, patch_size, embed_dim, encoder_num_layers, saved_weights_path):
    """
    Reconstructs the encoder model and loads the weights.

    Args:
        input_size (int): The input size for the encoder.
        patch_size (int): The patch size for reshaping.
        embed_dim (int): Embedding dimension for the Dense layer.
        encoder_num_layers (int): Number of Transformer layers in the encoder.
        saved_weights_path (str): Path to the saved encoder weights.
        transformer_block_fn (function): Function to create a TransformerBlock.
        positional_embeddings_fn (function): Function to generate positional embeddings.

    Returns:
        tf.keras.Model: Reconstructed encoder model with loaded weights.
    """
    # Number of patches
    num_patches = input_size // patch_size

    # Define the encoder input
    encoder_input = Input(shape=(input_size,), name="inputs")

    # Reshape the input into patches
    x = layers.Reshape((num_patches, patch_size), name="reshape")(encoder_input)

    # Dense layer for patch embeddings
    x = layers.Dense(embed_dim, name="dense")(x)

    # Add positional embeddings
    positional_embeddings = get_positional_embeddings(num_patches, embed_dim)
    x = x + positional_embeddings

    # Add Transformer layers
    attention_weights = []
    for i in range(encoder_num_layers):
        transformer_block = TransformerBlock(embed_dim, encoder_num_heads, encoder_ff_dim, encoder_rate, name_prefix=f"encoder_layer_{i+1}")
        x, weights = transformer_block(x)
        attention_weights.append(weights)

    # Define the final encoder model
    encoder_model = models.Model(inputs=encoder_input, outputs=[x] + attention_weights, name="encoder_model")

    # Load the saved weights
    encoder_model.load_weights(saved_weights_path)

    return encoder_model
