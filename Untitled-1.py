# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%
import pandas as pd
df = pd.read_csv("tictactoemoves.csv")
len(df)

# %%
X = df.drop('move', axis=1)
y = df['move']


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# %%
X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
x_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int32)

# %%
X_train_tf

# %%
def build_dqn(input_shape, num_actions):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_actions)  # Output layer: Q-values for each action
    ])
    return model

# %%
state_size = 9
input_shape = (state_size,)
num_actions = 9

# %%
dqn_model = build_dqn(input_shape, num_actions)

# %%
dqn_model.summary()

# %%
dqn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
history = dqn_model.fit(X_train_tf, y_train_tf, epochs=10, batch_size=32, validation_data=(x_test_tf, y_test_tf))

# %%



