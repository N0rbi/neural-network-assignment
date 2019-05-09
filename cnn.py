from more_itertools import peekable
import itertools
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.vgg19 import preprocess_input
import os
from keras import applications
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import h5py
from keras.layers import Dense,Flatten,Dropout
import numpy as np

def expand_dict(data):
    grouped = [a for a, b in data.items() if isinstance(b, list)]
    p = [[a, list(b)] for a, b in itertools.groupby(itertools.product(*[data[i] for i in grouped]), key=lambda x:x[0])]
    return itertools.chain(*[[{**data, **dict(zip(grouped, i))} for i in c] for _, c in p])

datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest',
        #validation_split=0.3)
	)

train_generator = datagen.flow_from_directory(
        'project/train/',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=100000,
        class_mode='categorical',
        #subset='training'
	)

class_to_int = train_generator.class_indices

steps_per_iter = len(train_generator.filenames) // 100

train_generator = peekable(train_generator)

<<<<<<< HEAD
=======
val_generator = datagen.flow_from_directory(
  'project/train/',  # this is the target directory
  target_size=(64, 64),  # all images will be resized to 150x150
  class_mode='categorical',
  subset='validation',
  batch_size=200
)

val_steps_per_iter = len(val_generator.filenames) // 100

>>>>>>> 6966f7eefacfb66c0dc40adce9d3fcc224805492

int_to_class = {v: k for k, v in class_to_int.items()}



METADATA = {
    "input_shape": train_generator.peek()[0].shape[1:],
    "output_shape": train_generator.peek()[1].shape[-1]
}

print("METADATA:" + str(METADATA))

def build_model(optimizer, convolution_units, dense_units):
    model = Sequential()
    for conv_unit in convolution_units:
        model.add(Convolution2D(conv_unit, 3, 3, input_shape=(64, 64, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    for dense_unit in dense_units:
        model.add(Dense(dense_unit, activation= "relu"))
    model.add(Dense(METADATA["output_shape"], activation="softmax"))

    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model



"""from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn= build_model)
"""
params = {"optimizer": ["adam"],
         "convolution_units": [[32], [64], [64], [32]],
         "dense_units": [[128], [64, 64], [128, 64], [512, 512, 64]]}

#model = GridSearchCV(estimator = classifier, param_grid = params, scoring = "val_accuracy")

# os.mkdir("saved_models")
from keras.callbacks import ModelCheckpoint, EarlyStopping
best_model = {"val_loss": np.inf, "model": None, "index": -1}

train_X, train_y = next(train_generator)

for i, param_set in enumerate(expand_dict(params)):
    print(param_set)
    model_checkpoint = ModelCheckpoint("/saved_models/%d_weights.{epoch:02d}.hdf5" % i, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    model = build_model(param_set["optimizer"], param_set["convolution_units"], param_set["dense_units"])

    history = model.fit(
	train_X,
	train_y,
	validation_split=.2,        
	epochs=2000,
        callbacks= [model_checkpoint, early_stopping])

    val_loss = np.min(history.history['val_loss'])

    if best_model["val_loss"] > val_loss:
        best_model = {"val_loss": val_loss, "model": model, "index": i}

print(best_model)
model = best_model["model"]

test = 'project/test_images/'
test_images = os.listdir(test)
test_images = test_images[0:10]
result = dict()
for image in test_images:
    im = load_img(test+"/"+image, target_size=(224, 224))
    im = np.asarray(im, dtype=np.uint8)
    if len(im.shape) != 3:
        im = np.stack((im,)*3, axis=-1)
    transformed_img = datagen.apply_transform(im, transform_parameters=dict())[np.newaxis, :] / 255.0
    result[image]=int_to_class[np.where(model.predict(transformed_img))[1][0]]

with open("saved_models/result.txt", "w") as result_file:
    print(str(result), file=result_file)
    print(str(result))
