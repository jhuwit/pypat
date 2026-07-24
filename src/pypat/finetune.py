from src.pypat.utils import *

# Tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import random

#@title Connect to TPU, GPU, or CPU
print("TensorFlow version:", tf.__version__)

# Connect to TPU, GPU, or CPU
try:
    # Try to connect to a TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)  # TPU strategy
    print("TPU available.")
except Exception as tpu_error:
    print(f"TPU initialization failed: {tpu_error}")
    try:
        # If TPU fails, try connecting to a GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # Allow memory growth for each GPU
            strategy = tf.distribute.MirroredStrategy()  # Multi-GPU or single GPU strategy
        else:
            print("No GPUs found; falling back to CPU.")
            strategy = tf.distribute.get_strategy()
    except RuntimeError as gpu_error:
        print(f"Error initializing GPUs: {gpu_error}")
        print("Using CPU as fallback.")
        strategy = tf.distribute.get_strategy()


# Function to load the encoder and build the fine-tuning model with consistent patching and positional embedding
def create_finetuning_model(encoder_path=None, input_size=None, return_attention=False):


    ### THIS IS ALL YOU NEED TO LOAD THE PRETRAINED PORTION ###

    # Load the saved encoder model
    encoder_model = load_encoder_model(input_size, patch_size, embed_dim, encoder_num_layers, encoder_path)
    # Define new inputs for the fine-tuning model
    inputs = layers.Input(shape=(input_size,), name="finetuning_inputs")
    # Get encoder outputs
    encoder_outputs = encoder_model(inputs)
    encoder_outputs, attention_weights = encoder_outputs[0], encoder_outputs[1:]


    ### YOU CAN CUSTOMIZE THE BELOW LAYERS IN WHATEVER WAY IS MOST USEFUL FOR YOUR PROJECT ###

    # Pass through a GlobalAveragePooling layer
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(encoder_outputs)
    x = layers.Dropout(0.1, name="dropout")(x)
    x = layers.Dense(128, activation='relu', name="dense_128")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    # Include attention weights in the final model outputs if requested
    if return_attention:
        outputs = [outputs] + attention_weights

    # Create and return the fine-tuning model
    finetuning_model = models.Model(inputs=inputs, outputs=outputs, name="finetuning_model")
    return finetuning_model
