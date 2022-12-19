# -*- coding: utf-8 -*-
import Util
import Models

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd


training_dataset_path = "UTKFace_downsampled/training_set/"
validation_dataset_path = "UTKFace_downsampled/validation_set/"
test_dataset_path = "UTKFace_downsampled/test_set/"

#Get training and validation data.
X_train, y_train, t_file_names = Util.get_data(training_dataset_path, -1)  # Getting the training set.
X_validation, y_validation, val_file_names = Util.get_data(validation_dataset_path, -1)  # Getting the validation set.
print("Train_X: " + str(X_train.shape) + " Train_Y: ", str(y_train.shape) + " File names: " + str(t_file_names.shape))
print("Val_X: " + str(X_validation.shape) + " Val_Y: ", str(y_validation.shape) + " File names: " + str(val_file_names.shape))

#Get model.
model = Models.get_model((91, 91, 3)) #Image size is 91x91 with RGB

#Train model.
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=256, epochs=75)

#Display training plots.
df = pd.DataFrame(history.history)
df.head()
loss_plot = df.plot(y=['loss', 'val_loss'], title='Loss vs Epochs', legend=True)
loss_plot.set(xlabel='Epochs')

mae_plot = df.plot(y=['mean_absolute_error', 'val_mean_absolute_error'], title='MAE vs Epochs', legend=True)
mae_plot.set(xlabel='Epochs')
plt.show()

# Testing on the validation data.
y_pred = []
y_true = []

for i in range(len(y_validation)):
    img = tf.expand_dims(X_validation[i], 0)
    label = y_validation[i]
    y_true = np.append(y_true, label)
    predicted_val = tf.round(model.predict(img, verbose=0)).numpy()[0][0]
    y_pred = np.append(y_pred, predicted_val)

m = tf.keras.metrics.MeanAbsoluteError()
m.update_state(y_true=y_true, y_pred=y_pred)
print("Validation MAE: " + str(m.result().numpy()))

X_test, y_test, names = Util.get_data(test_dataset_path, -1)  # Getting the test set.
print("Test_X: " + str(X_test.shape) + " Test_Y: ", str(y_test.shape) + " File names: " + str(names.shape))


# Testing the test data.
y_pred = []
y_true = []

y_pred_dict = {}
y_true_dict = {}

for i in range(len(y_test)):
    img = tf.expand_dims(X_test[i], 0)
    label = y_test[i]
    y_true = np.append(y_true, label)
    file_name = names[i]

    predicted_val = tf.round(model.predict(img, verbose=0)).numpy()[0][0]

    y_pred_dict[file_name] = predicted_val
    y_true_dict[file_name] = label

    y_pred = np.append(y_pred, predicted_val)

m = tf.keras.metrics.MeanAbsoluteError()
m.update_state(y_true=y_true, y_pred=y_pred)
print("Test MAE: " + str(m.result().numpy()))


# This module prints the images that are predicted in good or bad (Top 5 and worst 5)
img_pred_diff_dict = {}
for img_name in y_pred_dict.keys():
    y_pred = y_pred_dict[img_name]
    y_true = y_true_dict[img_name]
    img_pred_diff_dict[img_name] = abs(y_pred - y_true)

sorted_dict = dict(sorted(img_pred_diff_dict.items(), key=lambda item: item[1]))
top_five = np.array(list(sorted_dict.keys()))[0:5]
worst_five = np.array(list(sorted_dict.keys()))[-5:]

print("TOP 5")
for i in range(len(top_five)):
    img_name = top_five[i]
    print(str(i + 1) + ".   Image: " + str(img_name) + "    Age: " + str(y_true_dict[img_name]) + "   Predicted: " + str(y_pred_dict[img_name]))

print("\nWORST 5")
for i in range(len(worst_five)):
    img_name = worst_five[i]
    print(str(i + 1) + ".   Image: " + str(img_name) + "    Age: " + str(y_true_dict[img_name]) + "   Predicted: " + str(y_pred_dict[img_name]))