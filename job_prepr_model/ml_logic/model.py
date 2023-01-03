from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam



def initialize_model(X,y_cat_len,
                     Xshape,
                     maxpooling2d=2,
                     activation_for_hidden='relu',
                     kernel_size=(3,3),
                     kernel_size_detail=(2,2),
                     last_dense_layer_neurons_1=100,
                     last_dense_layer_neurons_2=100,
                     ):

    model = models.Sequential()
    model.add(Rescaling(1./255, input_shape=Xshape))
    model.add(layers.Conv2D(50, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Conv2D(50, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Dropout(.4))
    model.add(layers.MaxPooling2D(maxpooling2d))
    model.add(layers.Conv2D(30, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Conv2D(30, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Dropout(.4))
    model.add(layers.MaxPooling2D(maxpooling2d))
    model.add(layers.Conv2D(20, kernel_size=kernel_size_detail, activation=activation_for_hidden))
    model.add(layers.Conv2D(20, kernel_size=kernel_size_detail, activation=activation_for_hidden))
    model.add(layers.Dropout(.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(last_dense_layer_neurons_1, activation='relu'))
    model.add(layers.Dense(last_dense_layer_neurons_2, activation='relu'))
    model.add(layers.Dense(y_cat_len, activation='softmax'))
    return model



def compile_model(model):

    lr_schedule = ExponentialDecay(initial_learning_rate=1e-2,
                                    decay_steps=10000,
                                    decay_rate=0.9
                                    )

    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule),
              metrics=['accuracy'])

    return model

def train_model(model,
                X,
                y=None,
                batch_size=32,
                patience=2,
                validation_split=0.2,
                epochs=500,
                ):

    es = EarlyStopping(patience = patience,
                       restore_best_weights= True,
                       monitor = "val_accuracy",
                       mode = "max")

    history = model.fit(X, y,
                batch_size=batch_size,
                epochs = epochs,
                callbacks=[es],
                validation_split = validation_split,
                shuffle = True,
                verbose = 1)

    return model, history

def evaluate_model(model, X, y):
    metrics = model.evaluate(X, y)
    return metrics
