import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import os


cwd = os.getcwd()
path = os.path.join(cwd, 'data')
print(path)

def load_dataset():
    file_name = os.path.join(path, 'train_catvnoncat.h5')
    train_dataset = h5py.File(file_name, "r")
    X_train = np.array(train_dataset["train_set_x"][:])
    Y_train = np.array(train_dataset["train_set_y"][:])

    file_name = os.path.join(path, 'test_catvnoncat.h5')
    train_dataset = h5py.File(file_name, "r")
    X_test = np.array(train_dataset["test_set_x"][:])
    Y_test = np.array(train_dataset["test_set_y"][:])

    classes = ['non-cat', 'cat']

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test, classes

X_train, Y_train, X_test, Y_test, classes = load_dataset()

print('X_train.shape = ', X_train.shape)
print('X_test.shape = ', X_test.shape)
print('Y_train.shape = ', Y_train.shape)
print('Y_test.shape = ', Y_test.shape)

def m_tst_f ():
    m_train = X_train.shape[0]
    num_px = X_train.shape[1]
    m_test = X_test.shape[0]

    return m_train, num_px, m_test

m_train, num_px, m_test = m_tst_f()

print("\nNumber of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px =" + str(num_px))
print("Each image is of size: (" + str(num_px) + "," + str(num_px) + ", 3)")


#cat tsk index 1
def ind_1 ():
    Y_prediction_test = np.random.randint(0, 2, size=Y_test.shape)
    index = 1
    plt.imshow(X_test[index].reshape(num_px, num_px, 3))
    y_true = Y_test[index, 0]
    y_predicted = Y_prediction_test[index, 0]
    return y_predicted, y_true

y_true, y_predicted = ind_1()

print(f'\ny_predicted_index_1 = {y_predicted} (true label = {y_true}), you predicted that it is a {classes[y_predicted]} picture.')

#cat tsk index 6
def ind_6 ():
    Y_prediction_test = np.random.randint(0, 2, size=(X_test.shape[0], 1))
    index = 6
    plt.imshow(X_test[index, :].reshape(num_px, num_px, 3))
    y_true_6 = Y_test[index, 0]
    y_predicted_6 = Y_prediction_test[index, 0]
    return y_predicted_6 , y_true_6
y_true_6, y_predicted_6 = ind_6()

print(f'\ny_predicted_index_6 = {y_predicted_6} (true label = {y_true_6}), you predicted that it is a {classes[y_predicted_6]} picture.')