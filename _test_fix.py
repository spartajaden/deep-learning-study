import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 데이터 준비 (예시 데이터) — 실제 문제의 코드에서는 이 단계가 빠져 있었음
X = np.random.rand(1000, 10).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# train/test 분할 (sklearn 없이 간단히)
idx = np.arange(len(X))
np.random.shuffle(idx)
X_train, X_test = X[idx[:800]], X[idx[800:]]
y_train, y_test = y[idx[:800]], y[idx[800:]]
print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)

model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)

loss = model.evaluate(X_test, y_test, verbose=0)
print("test loss:", loss)
print("OK - X_train defined and training works")