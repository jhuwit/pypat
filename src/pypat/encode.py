## DOWNLOAD FILES from dropbox ##
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from src.pypat.utils import *
from src.pypat.finetune import create_finetuning_model

# Keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import AUC

# Tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import random

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


# URL of the encoder file
encoder_url = "https://www.dropbox.com/scl/fi/ha9b0cj4b3gvcfq4etc6h/weight_only_encoder_large_90_unsmoothed_mse_all.h5?rlkey=sbu5fd9p56qawnquz4w6stjzr&st=aewhfwq5&dl=1"

# Download the file
encoder_path = 'WEIGHTS_encoder_large_90_unsmoothed_mse_all.h5'
response = requests.get(encoder_url)
with open(encoder_path, 'wb') as f:
    f.write(response.content)

print("Encoder model saved to:", encoder_path)


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


# Call the function
size = "large"

"""
Model Size
"""
## Model Size
if size == "small":

  patch_size = 18
  embed_dim = 96
  # encoder
  encoder_num_heads = 6
  encoder_ff_dim = 256
  encoder_num_layers = 1
  encoder_rate = 0.1

if size == "medium":

  patch_size = 18
  embed_dim = 96
  # encoder
  encoder_num_heads = 12
  encoder_ff_dim = 256
  encoder_num_layers = 2
  encoder_rate = 0.1

if size == "large":

  patch_size = 9
  embed_dim = 96
  # encoder
  encoder_num_heads = 12
  encoder_ff_dim = 256
  encoder_num_layers = 4
  encoder_rate = 0.1

# Load seven complete 1,440-minute days for each participant and combine them
# into one 10,080-minute observation.  Days 1 and 9 are excluded because the
# data documentation notes that they can be incomplete.
activity_url = (
    "https://physionet.org/files/minute-level-step-count-nhanes/"
    "1.0.1/csv/nhanes_1440_AC.csv.xz"
)
activity_path = Path("nhanes_1440_AC.csv.xz")
minute_columns = [f"min_{minute:04d}" for minute in range(1, 1441)]

if not activity_path.exists():
    print(f"Downloading NHANES activity data to: {activity_path}")
    temporary_path = activity_path.with_suffix(activity_path.suffix + ".part")
    with requests.get(activity_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)
    temporary_path.replace(activity_path)
else:
    print(f"Using existing NHANES activity data: {activity_path}")

activity_data = pd.read_csv(activity_path, compression="xz")
missing_columns = set(["SEQN", "PAXDAYM", *minute_columns]) - set(activity_data.columns)
if missing_columns:
    raise ValueError(
        "The NHANES activity file is missing expected columns: "
        f"{sorted(missing_columns)}"
    )

# A 10,080-minute week requires one and only one record for each day 2--8.
week_data = activity_data.loc[
    activity_data["PAXDAYM"].between(2, 8),
    ["SEQN", "PAXDAYM", *minute_columns],
].sort_values(["SEQN", "PAXDAYM"])
complete_participants = (
    week_data.groupby("SEQN")["PAXDAYM"]
    .agg(["count", "nunique"])
    .query("count == 7 and nunique == 7")
    .index
)
week_data = week_data.loc[week_data["SEQN"].isin(complete_participants)]

X = week_data.loc[:, minute_columns].to_numpy(dtype=np.float32)
X = np.nan_to_num(X, nan=0.0).reshape(-1, 7 * 1440)
if X.shape[1] != 10080:
    raise RuntimeError(f"Expected 10,080 minutes per participant, got {X.shape[1]}")

X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42, shuffle=True
)
print(f"NHANES weekly data: {X.shape}; train: {X_train.shape}; test: {X_test.shape}")

original_length = X_train.shape[1]

padding_needed = (-original_length) % patch_size

print(padding_needed, "minutes of padding needed")

# Add zeros to the end of each sequence when required by the patch size.
X_train = np.pad(X_train, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)
X_test = np.pad(X_test, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)

print(f"New shape after adding {padding_needed} zeros to the end of each sequence")
# Print the shapes of the padded datasets to verify
padded_shapes = {
    "X_train": X_train.shape,
    "X_test": X_test.shape,
}
padded_shapes




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

# Scale X_train
train_scalar = StandardScaler()
train_scalar.fit(X_train)
X_train = train_scalar.transform(X_train)

# Scale X_test
X_test = train_scalar.transform(X_test)    

# Training Config

early_stopper = EarlyStopping(
    monitor='val_auc',  # monitor validation AUC
    mode='max',  # maximize AUC
    patience=250,  # number of epochs with no improvement after which training will be stopped
    verbose=1,  # display messages when early stopping is triggered
    restore_best_weights=True  # restore model weights from the epoch with the best value of the monitored quantity
)

# Set Class Weights = Balance (due to high class imbalance)
class1 = sum(y_train)
total = len(y_train)
class0 = total-class1

class_weights = {0: (class1/total),
              1: ((class0/total))}


# URL of the encoder file
encoder_url = "https://www.dropbox.com/scl/fi/ha9b0cj4b3gvcfq4etc6h/weight_only_encoder_large_90_unsmoothed_mse_all.h5?rlkey=sbu5fd9p56qawnquz4w6stjzr&st=aewhfwq5&dl=1"

# Download the file
encoder_path = 'WEIGHTS_encoder_large_90_unsmoothed_mse_all.h5'
response = requests.get(encoder_url)
with open(encoder_path, 'wb') as f:
    f.write(response.content)
    
   

# Compile the model -----
with strategy.scope():
  train_model = create_finetuning_model(return_attention=False, input_size=X_train.shape[1], encoder_path=encoder_path)
  train_model.compile(
    # Metrics
    loss= tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics= [tf.keras.metrics.AUC(name='auc')],
    # Optimizer
    optimizer= tf.keras.optimizers.Adam(
      learning_rate=0.000001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False
))



# Save the original model weights
train_model.save_weights('original_model_weights.weights.h5')

# reset the model
train_model.load_weights('original_model_weights.weights.h5')

# End to end finetune the model
history = train_model.fit(
    X_train, y_train,
    epochs= 150, # Edit
    batch_size= 64,
    validation_data = (X_val, y_val),
    shuffle=False,
    class_weight=class_weights,
    callbacks = [early_stopper],
    verbose = 2)              
