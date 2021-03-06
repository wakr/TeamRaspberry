#%%
from keras.utils.data_utils import get_file
import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.utils import class_weight

#%%

database_path = 'train/'

dl_file='dl2017-image-proj.zip'
dl_url='https://www.cs.helsinki.fi/u/mvsjober/misc/'
get_file(dl_file, dl_url+dl_file, cache_dir='./', cache_subdir=database_path, extract=True)

print('download complete')
#%%


image_count = 20000
image_dim = (128, 128, 3) # 128 x 128; RGB

print('cell run')
#%% Form y
def get_readable_labels(y_i):
    return [labels[i] for (i,x) in enumerate(y_i) if x]

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

print('classes and y labels created')



#%%
# normalizing range to be in 0.0 to 1.0
my_dtype = "uint8"
normalize_values = True

my_dtype = "uint8"
if normalize_values:
    my_dtype = np.float32

def read_x(batch_size, begin_index, y):
    x_color = np.zeros((batch_size, *image_dim), dtype=my_dtype)
    end_index = begin_index + batch_size

    for i in range(begin_index, end_index):
        img = imread("train/images/im" + str(i+1) +".jpg")
        if normalize_values:
            img = img / 255
        if len(img.shape) == 3:
            x_color[i-begin_index] = img
        else:    
            x_color[i-begin_index] = np.repeat(img[:,:,np.newaxis], 3, axis=2) # B&W RGB representation

    return x_color, y[begin_index:end_index,:]

print('read_x defined')

#%% Sanity check x-reader

x_testing, y_testing = read_x(image_count, 0, y)
print(x_testing.shape)

#%% Mutexes
mutex_labels = [0,3] 

mutex_x = []
mutex_y = []
for i in range(x_testing.shape[0]):
    img = x_testing[i]
    labls = [i for (i, x) in enumerate(y[i]) if x]
    found = np.isin(labls, mutex_labels).any()
    if found:
        mutex_x.append(img)
        new_y = np.zeros(len(mutex_labels), dtype="uint8")
        for i in range(len(new_y)):
            if mutex_labels[i] in labls:
                new_y[i] = 1
            else:
                new_y[i] = 0
        mutex_y.append(new_y)

mutex_x = np.array(mutex_x)
mutex_y = np.array(mutex_y)


#%% MODEL

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
print('model imports done')

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
    return f1_score(y_true, y_pred, average='micro', labels=np.unique(y_pred))

print('evaluators and metrics defined')
    
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
    pseudo_f1 = f1(y_true, y_pred)
    pseudo_f1 = K.switch(pseudo_f1, K.epsilon(), pseudo_f1)
    # sum over samples
    sum_over_samples = K.sum(results)
    return sum_over_samples #+ (100 * (1 - pseudo_f1))

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


print('loss function(s)')
#%%

model = Sequential()

# Add layers here

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(len(mutex_labels)))
model.add(Activation("sigmoid"))


# binary_crossentropy
model.compile(loss=bp_mll_loss, 
              optimizer='rmsprop', 
              metrics=['binary_accuracy', "categorical_accuracy", f1])


print(model.summary())




#%%

x_train, x_test, y_train, y_test = train_test_split(mutex_x, mutex_y, test_size=0.2)



#%% Augmentor
train_datagen = ImageDataGenerator(
                samplewise_center=True,
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                )

print('data generator defined')

#%% Create class_weights
import math
def create_class_weight(labels_dict,mu=0.9):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1

    return class_weight

total_sum_y = mutex_y.sum(axis=0)
y_max = total_sum_y.max()

labels_dict = {i: (x) for (i, x) in enumerate(total_sum_y)}
cw = create_class_weight(labels_dict)

#%%


batch_size = 2000
overall_history = []
epochs = 5

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=2),
                                    steps_per_epoch=len(x_train) / 2, 
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    class_weight=cw)
overall_history.append(history)

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
    bin_acc_per_epoch = np.array([e.history['binary_accuracy'] for e in overall_history]).flatten()
    plt.plot(bin_acc_per_epoch)
    plt.title("binary_acc")
    plt.subplot(1,2,2)
    loss_per_epoch = np.array([e.history['loss'] for e in overall_history]).flatten()
    plt.plot(loss_per_epoch)
    plt.title("loss")
    plt.subplot(1,2,1)
    loss_per_epoch = np.array([e.history['val_f1'] for e in overall_history]).flatten()
    plt.plot(loss_per_epoch)
    plt.title("bin_acc/f1")
    plt.ylim((0,1))
    plt.tight_layout()
    
plot_train_metrics()

#%% Confusion-matrix

plt.figure(figsize=(20, 10))
true_pos = np.sum(y_test * y_pred, axis=0)
for i in range(len(true_pos)):
    
    plt.subplot(3, 5, i+1)
    
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test[:,i], y_pred[:,i]).ravel()
    
    matrix = np.zeros((2,2), dtype="uint32")
    matrix[0,0] = true_pos
    matrix[0,1] = false_neg
    matrix[1,1] = true_neg
    # sanity check: bottom left, predicted pos, true was neg
    matrix[1,0] = false_pos

    plt.imshow(matrix, 
               cmap=plt.cm.Blues, 
               interpolation='nearest',
               vmin=0, vmax=10000)

    for j in range(2):
        for k in range(2):
            plt.text(k, j, str(matrix[j,k]),
                        fontsize=15,
                        horizontalalignment='center',
                        verticalalignment='center')

    classes = ["Pos", "Neg"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title(labels[i], fontsize=15)
    

plt.tight_layout()
plt.show()




# In[ ]:

model.save('model.h5')


# In[ ]:

np.savetxt('results.txt', y, fmt='%d')

