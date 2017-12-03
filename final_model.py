from keras.utils.data_utils import get_file
import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss

#%%

database_path = 'train/'

dl_file='dl2017-image-proj.zip'
dl_url='https://www.cs.helsinki.fi/u/mvsjober/misc/'
get_file(dl_file, dl_url+dl_file, cache_dir='./', cache_subdir=database_path, extract=True)

#%% 


image_count = 20000
image_dim = (128, 128, 3) # 128 x 128; RGB

#%%

def form_image_data(normalize_values=False): 
    # normalizing range to be in 0.0 to 1.0
    my_dtype = "uint8"
    if normalize_values:
        my_dtype = np.float32
        
    x_color = np.zeros((image_count, *image_dim), dtype=my_dtype)
    for i in range(image_count):
        img = imread("train/images/im" + str(i+1) +".jpg")
        if normalize_values:
            img = img / 255
        if len(img.shape) == 3:
            x_color[i] = img
        else:    
            x_color[i] = np.repeat(img[:,:,np.newaxis], 3, axis=2) # B&W RGB representation
    return x_color
    
images_X = form_image_data(normalize_values=True)


#%%

num_classes = 14 + 1
UNCLASSIFIED_BIT = 14
labels = {
  0: 'baby',
  1: 'bird',
  2: 'car',
  3: 'clouds',
  4: 'dog',
  5: 'female',
  6: 'flower',
  7: 'male',
  8: 'night',
  9: 'people',
  10: 'portrait',
  11: 'river',
  12: 'sea',
  13: 'tree',
  UNCLASSIFIED_BIT: "unclassified"
}

labels_rev = {v: k for k, v in labels.items()}

# The labels are made into onehot vectors straight away
y = np.zeros((image_count, num_classes), dtype="uint8")

folder = "train/annotations/"
for num, label in labels.items():
    if num == UNCLASSIFIED_BIT:
        continue
    with open(folder + label +".txt", "r") as f:
        for line in f.readlines():
            # The indexing starts from 1 in dataset, but is converted to starting from 0
            y[int(line.strip())-1, num] = 1

for row in y:
    if sum(row) == 0:
        row[UNCLASSIFIED_BIT] = 1

#%%
        
def get_label_statistics():
    total = np.sum(y)
    count_per_label = y.sum(axis=0)
    unlabeled_images = count_per_label[UNCLASSIFIED_BIT]
    label_count = [(x, count_per_label[i]) for (i,x) in enumerate(labels_rev.keys())]
    print("Total amount of labels: %s" % total)
    print("Amount of images with no class: %d" % unlabeled_images)
    print("Amount of images per class:")
    print(label_count)
    l, c = zip(*label_count)
    plt.barh(np.arange(0, num_classes), c, tick_label=l)
        

get_label_statistics()
#%%
# Sanity check for black-and-white images

plt.subplot(1,2,1)
plt.imshow(imread("train/images/im7.jpg"), cmap='Greys_r', interpolation='none')
plt.title("Original B&W")

plt.subplot(1,2,2)
plt.imshow(np.repeat(imread("train/images/im7.jpg")[:,:,np.newaxis], 3, axis=2))
plt.title("Modified RGB B&W")

plt.tight_layout()

#%%
# Some plotting of first instances of each class

for l in range(num_classes):
    idx = np.argwhere(y[:,l]==1)[0]
    
    
    plt.subplot(3, 5, l+1)
    
    img = images_X[idx].reshape(128, 128, 3)
        
    plt.imshow(img)
    plt.title(labels[l])
    plt.axis('off')

#%% MODEL
    
x_train, x_test, y_train, y_test = train_test_split(images_X, y, test_size=0.2)

x_train.shape  

#%%

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, Dropout
from keras import backend as K

#%% Evaluators

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

# Definition of custom metrics as the project is ranked with F1 score
from sklearn.metrics import f1_score

def f1__score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


#%%
    
# bp mll loss function
# y_true, y_pred must be 2D tensors of shape (batch dimension, number of labels)
# y_true must satisfy y_true[i][j] == 1 iff sample i has label j
def bp_mll_loss(y_true, y_pred):
 
    # get true and false labels
    y_i = K.equal(y_true, K.ones_like(y_true))
    y_i_bar = K.not_equal(y_true, K.ones_like(y_true))
    
    # cast to float as keras backend has no logical and
    y_i = K.cast(y_i, dtype='float32')
    y_i_bar = K.cast(y_i_bar, dtype='float32')

    # get indices to check
    truth_matrix = pairwise_and(y_i, y_i_bar)

    # calculate all exp'd differences
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = K.exp(-sub_matrix)

    # check which differences to consider and sum them
    sparse_matrix = exp_matrix * truth_matrix
    sums = K.sum(sparse_matrix, axis=[1,2])

    # get normalizing terms and apply them
    y_i_sizes = K.sum(y_i, axis=1)
    y_i_bar_sizes = K.sum(y_i_bar, axis=1)
    normalizers = y_i_sizes * y_i_bar_sizes
    results = sums / normalizers

    # sum over samples
    return K.sum(results)

# compute pairwise differences between elements of the tensors a and b
def pairwise_sub(a, b):
    column = K.expand_dims(a, 2)
    row = K.expand_dims(b, 1)
    return column - row

# compute pairwise logical and between elements of the tensors a and b
def pairwise_and(a, b):
    column = K.expand_dims(a, 2)
    row = K.expand_dims(b, 1)
    return K.minimum(column, row)

#%%

batch_size = 512
epochs = 5

model = Sequential()

# Add layers here

model.add(Conv2D(32, (3, 3), input_shape=image_dim))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("sigmoid"))


model.compile(loss=bp_mll_loss, 
              optimizer='rmsprop', 
              metrics=['binary_accuracy', 'categorical_accuracy', f1])
print(model.summary())


# Use 1000 images for training at first because SLOW
hist = model.fit(x_train[:1000], 
                 y_train[:1000], 
                 batch_size=batch_size, 
                 epochs=epochs,
                 validation_data=(x_test, y_test),
                 shuffle=True)


#%% Model evaluation

scores = model.evaluate(x_test, y_test, verbose=2)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))


threshold = 0.5

pred = model.predict(x_test)
pred[np.where(pred > threshold)] = 1
pred[np.where(pred <= threshold)] = 0
y_pred = pred
print("F1:", f1_score(y_test, y_pred, average='micro'))
print("Hamming loss:", hamming_loss(y_test, y_pred))
#%%

def plot_train_metrics():
    plt.subplot(1,2,1)
    plt.plot(hist.history['binary_accuracy'])
    plt.title("binary_acc")
    plt.ylim((0, 1))
    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'])
    plt.title("loss")
    plt.subplot(1,2,1)
    plt.plot(hist.history['val_f1'])
    plt.title("bin_acc/f1")
    plt.ylim((0, 1))
    plt.tight_layout()
    
plot_train_metrics()

#%% Confusion-matrix

for i in range(y_train.shape[1]):
    print("Col {}".format(i))
    tn, fp, fn, tp = confusion_matrix(y_test[:,i], y_pred[:,i]).ravel()
    print("Label {}".format(labels[i]))
    print("True-Neg:", tn)
    print("False-Neg:", fn)
    print("True-Pos:", tp)
    print("False-Pos:", tn)
    print("")

# In[24]:

def get_readable_labels(y_i):
    return [labels[i] for (i,x) in enumerate(y_i) if x]





# In[ ]:

model.save('model.h5')


# In[ ]:

np.savetxt('results.txt', y, fmt='%d')

