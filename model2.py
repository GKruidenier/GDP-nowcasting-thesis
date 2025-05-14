from tensorflow.keras import layers
import tensorflow as tf

# Inputs
monthly_input = layers.Input(shape=(timesteps, monthly_features))
quarterly_input = layers.Input(shape=(timesteps // 3, quarterly_features))  # Quarterly is 3x slower

# Quarterly LSTM
quarterly_lstm = layers.LSTM(32, return_sequences=True)(quarterly_input)
quarterly_upsampled = layers.UpSampling1D(size=3)(quarterly_lstm)  # Repeat each output 3 times to match monthly

# Concatenate repeated quarterly output with monthly input
monthly_augmented_input = layers.Concatenate()([monthly_input, quarterly_upsampled])

# Monthly LSTM
monthly_lstm = layers.LSTM(64, return_sequences=True)(monthly_augmented_input)

# Output layer
output = layers.TimeDistributed(layers.Dense(1))(monthly_lstm)

model = tf.keras.Model(inputs=[monthly_input, quarterly_input], outputs=output)
model.summary()
