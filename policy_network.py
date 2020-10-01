import numpy as np
import keras
import keras.optimizers
import keras.losses
from keras import layers


class PolicyNetwork:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        input_state = keras.Input(shape=(4, 4, 1,))
        # x = layers.Conv2D(256, kernel_size=(2, 2))(input_state)
        # x = layers.Conv2D(256, kernel_size=(2, 2))(x)
        x = layers.Flatten()(input_state)
        x = layers.Dense(2048, activation="relu")(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        output = layers.Dense(4)(x)

        self.model = keras.Model(inputs=input_state, outputs=output)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.MeanSquaredError(),
        )

    def predict(self, s: np.array) -> np.array:
        return self.model.predict(x=s)

    def update(self, s: np.array, a: np.array, r: np.array) -> None:
        r = r * a
        self.model.fit(
            x=s, y=r, batch_size=self.batch_size, epochs=1, 
            verbose=0
        )

    def save(self) -> None:
        self.model.save("model.h5")
