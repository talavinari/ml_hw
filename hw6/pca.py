#################################
# Your name: Tal Avinari
#################################

# matplotlib.use('TkAgg')
import os
import ssl

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_lfw_people


def plot_vector_as_image(image, h, w):
    title = "bla"
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    plt.show()
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """


def get_pictures_with_limit_number(threshold=70):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    labels = []
    n_samples, h, w = lfw_people.images.shape
    unique, counts = np.unique(lfw_people.target, return_counts=True)
    pic_dict = dict(zip(unique, counts))
    # print(pic_dict)
    all_relevant_targets = [k for k, v in pic_dict.items() if v >= threshold]
    # print(all_relevant_targets)

    for image, target in zip(lfw_people.images, lfw_people.target):
        if target in all_relevant_targets:
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
            labels.append(target)
    return selected_images, labels, h, w


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if (target == target_label):
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k, d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
              of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    u, S, vh = np.linalg.svd(X, full_matrices=False)
    U = vh[:k]
    S = S[:k]
    return U, S


def exc_d():
    selected_images, labels, h, w = get_pictures_with_limit_number()
    mean_zero_pics = normalize_to_zero_mean(selected_images)
    ks = [1, 5, 10, 30, 50, 100, 150, 300, len(mean_zero_pics[0])]
    results = np.ndarray(shape=(len(ks), 2), dtype=float)

    for index, k in enumerate(ks):
        V = find_V_for_reduce_dim(mean_zero_pics, k)
        reduced_images = [transform_to_V(V, img) for img in mean_zero_pics]
        x_train, y_train, x_val, y_val = split_train_validation(reduced_images, labels)
        validation_size = len(x_val)

        clf = svm.SVC(C=1000.0, decision_function_shape='ovr', kernel='rbf', gamma=pow(10, -7))
        clf.fit(x_train, y_train)
        predict = clf.predict(x_val)
        correct_classifications = np.count_nonzero(predict == y_val)
        accuracy = correct_classifications / validation_size
        results[index] = [k, accuracy]
        # print("accuracy for dim " + str(k) + " is : " + str(accuracy))

    # plt.title('Accuracy on the test set a function of K')
    # plt.ylabel('Accuracy ')
    # plt.xlabel('K')
    # plt.plot(results[:, 0], results[:, 1], color='blue')
    # plt.axis([0, len(mean_zero_pics[0]), 0, 1])
    # plt.grid(axis='x', linestyle='-')
    # plt.grid(axis='y', linestyle='-')
    # plt.show()


def split_train_validation(images, labels):
    train_size = int(len(images) * 0.75)
    x_train, y_train, x_val, y_val = images[:train_size], labels[:train_size], images[
                                                                               train_size:], labels[
                                                                                             train_size:]

    return x_train, y_train, x_val, y_val


def find_V_for_reduce_dim(origin_data, target_dim):
    eigen_vectors, s = PCA(origin_data, target_dim)
    V = np.transpose(eigen_vectors)
    Vt = eigen_vectors
    # return V, Vt
    return V


def transform_to_V(V, x):
    Vt = np.transpose(V)
    return np.dot(np.dot(V, Vt), x)


def normalize_to_zero_mean(selected_images):
    selected_images = np.array(selected_images).reshape(len(selected_images), len(selected_images[0]))
    mean_vector = np.mean(selected_images, axis=0)
    return selected_images - mean_vector


def exc_c(chosen_name):
    selected_images, h, w = get_pictures_by_name(chosen_name)
    # print("selected images number of " + chosen_name + " is: " + str(len(selected_images)))
    new_pics = normalize_to_zero_mean(selected_images)

    ks = [1, 5, 10, 30, 50, 100]
    l2_sums = np.ndarray(shape=(6, 2), dtype=float)

    for index, k in enumerate(ks):
        eigen_vectors, s = PCA(new_pics, k)
        V = np.transpose(eigen_vectors)
        Vt = eigen_vectors
        selected_5 = new_pics[:5]
        transformed_pics = []
        for i in range(len(selected_5)):
            transformed_pics.append(np.dot(np.dot(V, Vt), selected_5[i]))

        reshped_transformed = [u.reshape(h, w) for u in transformed_pics]
        reshped_original = [u.reshape(h, w) for u in selected_5]

        # figure, axarr = plt.subplots(5, 2)
        # figure.suptitle("5 Random images for reduce dim to k= " + str(k), fontsize=16)
        dist_sum = 0
        for i in range(5):
            # axarr[i, 0].imshow(reshped_original[i].reshape((h, w)), cmap=plt.cm.gray)
            # axarr[0, 0].set_title("original")
            # axarr[i, 1].imshow(reshped_transformed[i].reshape((h, w)), cmap=plt.cm.gray)
            # axarr[0, 1].set_title("reduced ")

            dist_sum += np.linalg.norm(reshped_original[i] - reshped_transformed[i])
        l2_sums[index] = [k, dist_sum]

    # plt.show()

    # print(l2_sums)

    # plt.title('Sum of L2 distances between 5 image pairs as function of K')
    # plt.ylabel('Sum of L2')
    # plt.xlabel('K tested = 1, 5, 10, 30, 50, 100')
    # plt.plot(l2_sums[:, 0], l2_sums[:, 1], color='blue')
    # plt.axis([0, 100, 0, 6500])
    # plt.grid(axis='x', linestyle='-')
    # plt.grid(axis='y', linestyle='-')
    # plt.show()


def exc_b(chosen_name):
    selected_images, h, w = get_pictures_by_name(chosen_name)
    # print("selected images number of " + chosen_name + " is: " + str(len(selected_images)))
    new_pics = normalize_to_zero_mean(selected_images)
    eigen_vectors, s = PCA(new_pics, 10)

    images = [u.reshape(h, w) for u in eigen_vectors]
    figure, axarr = plt.subplots(2, 5)
    figure.suptitle("First 10 eigen-vectors as images", fontsize=16)
    for i in range(5):
        axarr[0, i].imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        axarr[0, i].set_title("u" + str(i))
    for i in range(5, 10):
        axarr[1, i % 5].imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        axarr[1, i % 5].set_title("u" + str(i))

    # plt.show()


def main():
    names = ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
             'Gerhard Schroeder', 'Hugo Chavez', 'Tony Blair']

    chosen_name = names[0]
    exc_b(chosen_name)
    exc_c(chosen_name)
    exc_d()


if __name__ == '__main__':

    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context
    np.set_printoptions(suppress=True)
    main()
