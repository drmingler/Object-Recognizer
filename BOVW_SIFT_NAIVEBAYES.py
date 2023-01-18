import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def read_image(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))


def get_files(path, train=True):
    images = {}
    for folder in os.listdir(path):
        category = []
        for file in os.listdir(os.path.join(path, folder)):
            image_path = os.path.join(path, os.path.join(folder, file))
            category.append(read_image(image_path))

        # category_ = [np.random.shuffle(category)] if train else category
        images[folder] = category

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

    for label, image in data:
        images.append(image)
        labels.append(labels_to_id.get(label))

    return images, labels


def extract_features(images, labels, extractor):
    feature_vectors = []
    descriptor_list = []
    for images_, labels_ in zip(images, labels):
        for img in images_:
            kp, des = extractor.detectAndCompute(img, None)
            if des is not None:
                descriptor_list.extend(des)
                feature_vectors.append((des, labels_))

    return descriptor_list, feature_vectors


def cluster_features(descriptors, no_clusters):
    kmeans = KMeans(n_clusters=no_clusters).fit(np.array(descriptors))
    return kmeans


def create_image_histograms(features, kmeans, no_clusters):
    histograms = []
    for descriptors, label in features:
        histogram_ = np.zeros(no_clusters)
        for descriptor in descriptors:
            feature = descriptor.reshape(1, 128)
            ind = kmeans.predict(feature)
            histogram_[ind] += 1
        histograms.append((histogram_, label))

    return histograms


def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {"C": Cs, "gamma": gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def find_SVM(im_features, train_labels, kernel):
    features = im_features
    if kernel == "precomputed":
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param, class_weight=None)
    svm.fit(features, train_labels)
    return svm


def get_histogram_with_labels(histograms_labels):
    labels = []
    histograms_ = []
    for histogram, label in histograms_labels:
        labels.append(label)
        histograms_.append(histogram)

    return np.array(labels, dtype=float), np.array(histograms_)


def find_accuracy(true, predictions):
    print("accuracy score: %0.3f" % accuracy_score(true, predictions))


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def plot_confusions(true, predictions):
    np.set_printoptions(precision=2)

    class_names = [
        "city",
        "face",
        "green",
        "house building",
        "house indoor",
        "office",
        "sea",
    ]
    plot_confusion_matrix(
        true,
        predictions,
        classes=class_names,
        title="Confusion matrix, without normalization",
    )

    plot_confusion_matrix(
        true,
        predictions,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix",
    )

    plt.show()


def train_model(no_clusters, kernel, extractor, images, labels):
    descriptor_list, feature_vectors = extract_features(images, labels, extractor)
    visual_words = cluster_features(descriptor_list, no_clusters=no_clusters)
    histograms_labels = create_image_histograms(
        features=feature_vectors, kmeans=visual_words, no_clusters=no_clusters
    )

    labels, histograms_ = get_histogram_with_labels(histograms_labels)

    print("histograms_ ", histograms_)
    print("labels ", labels)

    scale = StandardScaler().fit(histograms_)
    histograms_ = scale.transform(histograms_)

    # svm = find_SVM(histograms_, labels, kernel)
    params_NB = {"var_smoothing": np.logspace(0, -9, num=100)}
    nb_classifier = GaussianNB()
    gs_NB = GridSearchCV(
        estimator=nb_classifier,
        param_grid=params_NB,
        cv=5,  # use any cross validation technique
        verbose=1,
        scoring="accuracy",
    )
    gs_NB.fit(histograms_, labels)

    print("SVM fitted.")
    print("Training completed.")

    return visual_words, scale, gs_NB, histograms_


def test_model(
    no_clusters,
    kernel,
    extractor,
    visual_words,
    svm,
    training_histograms,
    images,
    labels,
    scale,
):
    _, feature_vectors = extract_features(images, labels, extractor)
    histograms_labels = create_image_histograms(
        features=feature_vectors, kmeans=visual_words, no_clusters=no_clusters
    )

    labels, histograms_ = get_histogram_with_labels(histograms_labels)
    histograms_ = scale.transform(histograms_)

    # kernel_test = histograms_
    # if kernel == "precomputed":
    #     kernel_test = np.dot(histograms_, training_histograms.T)

    predictions = [i for i in svm.predict(histograms_)]
    print("Test images classified.")

    find_accuracy(labels, predictions)
    print("Accuracy calculated.")

    plot_confusions(labels, predictions)
    print("Confusion matrixes plotted.")


if __name__ == "__main__":
    train_data = get_files("dataset/train", train=True).items()
    test_data = get_files("dataset/test", train=False).items()

    (input_X_train, output_Y_train) = data_to_repr(train_data)
    (input_X_test, output_Y_test) = data_to_repr(test_data)
    sift = cv2.SIFT_create()

    visual_words, scale, svm, histograms_ = train_model(
        no_clusters=100,
        kernel="precomputed",
        extractor=sift,
        images=input_X_train,
        labels=output_Y_train,
    )

    test_model(
        no_clusters=100,
        kernel="precomputed",
        extractor=sift,
        visual_words=visual_words,
        scale=scale,
        svm=svm,
        training_histograms=histograms_,
        images=input_X_test,
        labels=output_Y_test,
    )
