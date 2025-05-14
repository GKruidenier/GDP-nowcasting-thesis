from tensorflow.keras import layers 
import tensorflow as tf

timesteps = 12 * 2
monthly_features = 10
quarterly_features = 15

# Inputs
monthly_input = tf.keras.Input(shape=(timesteps, monthly_features))
quarterly_input = tf.keras.Input(shape=(timesteps // 4, quarterly_features))  # Quarterly every 4th step

# Separate LSTM layers
quarterly_lstm = tf.keras.layers.LSTM(64, return_state=True)
monthly_lstm = tf.keras.layers.LSTM(64, return_state=True)

# Initialize outputs list
outputs = []
state_h, state_c = None, None

# Alternating logic: quarter, month, month, month, quarter, ...
for i in range(timesteps):
    if i % 4 == 0:
        # Quarterly step
        q_index = i // 4
        q_input = tf.keras.layers.Lambda(lambda x: x[:, q_index:q_index+1, :])(quarterly_input)
        out, state_h, state_c = quarterly_lstm(q_input, initial_state=[state_h, state_c] if state_h is not None else None)
    else:
        # Monthly step
        m_input = tf.keras.layers.Lambda(lambda x: x[:, i:i+1, :])(monthly_input)
        out, state_h, state_c = monthly_lstm(m_input, initial_state=[state_h, state_c])

    outputs.append(tf.keras.layers.Reshape((1, 64))(out))

# Concatenate all outputs
combined_output = tf.keras.layers.Concatenate(axis=1)(outputs)

# Output layer
final_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(combined_output)

# Define model
model = tf.keras.Model(inputs=[monthly_input, quarterly_input], outputs=final_output)
model.summary()