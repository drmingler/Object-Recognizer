from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# importing of service libraries
import numpy as np
import matplotlib.pyplot as plt

print('Libraries imported.')

# training constants
BATCH_SIZE = 128
N_EPOCH = 20  # use 20 for best initial results
N_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

print('Main variables initialised.')

IMG_ROWS = 32
IMG_COLS = 32
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)

print('Image variables initialisation')


def read_image(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (IMG_ROWS, IMG_COLS))


def getFilesPath(path_, train):
    images_paths = []
    for folder in os.listdir(path_):
        for file in os.listdir(os.path.join(path_, folder)):
            full_path = os.path.join(path_, os.path.join(folder, file))
            images_paths.append((full_path, folder))

    if train is True:
        np.random.shuffle(images_paths)

    return images_paths


def get_files(path_, train):
    files_path = getFilesPath(path_, train)
    images = []
    for image_path, class_ in files_path:
        images.append((read_image(image_path), class_))

    return images


def data_to_repr(data):
    labels_to_id = {
        "city": 0,
        "face": 1,
        "green": 2,
        "house_building": 3,
        "house_indoor": 4,
        "office": 5,
        "sea": 6,
    }
    labels = []
    images = []

    for image, label in data:
        images.append(image)
        labels.append(labels_to_id.get(label))

    return np.array(images, dtype=float), np.array(labels)


class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()

        # CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


print('LeNet class defined.')


if __name__ == "__main__":
    # load dataset
    train_data = get_files("dataset/train", train=True)
    test_data = get_files("dataset/test", train=False)

    (input_X_train, output_y_train) = data_to_repr(train_data)
    (input_X_test, output_y_test) = data_to_repr(test_data)

    # convert to categorical
    output_Y_train = utils.to_categorical(output_y_train, N_CLASSES)
    output_Y_test = utils.to_categorical(output_y_test, N_CLASSES)

    # float and normalization
    input_X_train = input_X_train.astype('float32')
    input_X_test = input_X_test.astype('float32')
    input_X_train /= 255
    input_X_test /= 255

    input_X_train = input_X_train[:, :, :, np.newaxis]
    input_X_test = input_X_test[:, :, :, np.newaxis]

    # initialize the optimizer and compile the model
    model = LeNet.build(input_shape=INPUT_SHAPE, classes=N_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    model.summary()

    # training/fitting of the DNN model
    history = model.fit(input_X_train, output_Y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH,
                        validation_split=VALIDATION_SPLIT, verbose=VERBOSE)

    # Testing
    score = model.evaluate(input_X_test, output_Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print("\nTest score/loss:", score[0])
    print('Test accuracy:', score[1])

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

