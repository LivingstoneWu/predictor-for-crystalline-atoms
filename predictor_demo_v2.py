import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy
import pandas
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# The path of the csv file
dataPath='/Users/j.s.k/Desktop/FYP/data_extractor/CNOLiNaKAlSiPS_distance_data.csv'
distance_k=1
modelName='CNO_k_'+str(distance_k)
periodic_dictionary={"H":0, "He":1, "Li":2, "Be":3, "B":4, "C":5, "N":6, "O":7, "F":8, "Ne":9, "Na":10, "Mg":11, "Al":12, "Si":13, "P":14, "S":15, "Cl":16, "Ar":17, "K":18, "Ca":19, "Sc":20, "Ti":21, "V":22, "Cr":23, "Mn":24, "Fe":25, "Co":26, "Ni":27, "Cu":28, "Zn":29, "Ga":30, "Ge":31, "As":32, "Se":33, "Br":34, "Kr":35, "Rb":36, "Sr":37, "Y":38, "Zr":39, "Nb":40, "Mo":41, "Te":42, "Ru":43, "Rh":44, "Pd":45, "Ag":46, "Cd":47, "In":48, "Sn":49, "Sb":50, "Te":51, "I":52, "Xe":53, "Cs":54, "Ba":55, "La":56, "Ce":57, "Pr":58, "Nd":59, "Pm":60, "Sm":61, "Eu":62, "Gd":63, "Tb":64, "Dy":65, "Ho":66, "Er":67, "Tm":68, "Yb":69, "Lu":70, "Hf":71, "Ta":72, "W":73, "Re":74, "Os":75, "Ir":76, "Pt":77, "Au":78, "Hg":79, "Tl":80, "Pb":81, "Bi":82, "Po":83, "At":84, "Rn":85, "Fr":86, "Ra":87, "Ac":88, "Th":89, "Pa":90, "U":91, "Np":92, "Pu":93, "Am":94, "Cm":95, "Bk":96, "Cf":97, "Es":98, "Fm":99, "Md":100, "No":101, "Lr":102, "Rf":103, "Db":104, "Sg":105, "Bh":106, "Hs":107, "Mt":108, "Ds":109, "Rg":110, "Cn":111, "Nh":112, "Fl":113, "Mc":114, "Lv":115, "Ts":116, "Og":117, "Uue":118,}
tf.config.experimental.list_physical_devices('GPU')

# helper method to check whether the specific element is in the list or not
def element_in_list(element, list):
    for i in list:
        if (i==element): return True
    return False

# method turning Element types into labels represented by integers
def turnIntoLabel(list):
    list_labels=[]
    appeared_elements=[]
    for item in list:
        if (not element_in_list(item, appeared_elements)):
            appeared_elements.append(periodic_dictionary[item])
        list_labels.append(periodic_dictionary[item])
    return numpy.asarray(list_labels), appeared_elements


# read the data and generate training and testing sets
dataset=pandas.read_csv(dataPath, skiprows=[0]).to_numpy()
samples_label, appeared_elements=turnIntoLabel(dataset[:,0])
samples_data=dataset[:,1:distance_k+1].astype('float32')
X_train, X_test, y_train, y_test = train_test_split(samples_data, samples_label, random_state=0)

# preset the class weight to solve class_imbalance
class_weights=dict.fromkeys(range(118),118.)
for element in appeared_elements:
    class_weights[element]=1.

# first part of the model
model_1=keras.Sequential([
    layers.Dense(64, activation='sigmoid',name='layer_1',input_shape=(distance_k,)),
    layers.Dense(64, activation='sigmoid', name='layer_2'),
    layers.Dense(64, activation='sigmoid', name='layer_3'),
    layers.Dense(118, activation='sigmoid', name='output')
])

model_1.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history=model_1.fit(X_train,y_train,batch_size=64,epochs=60,class_weight=class_weights)
model_1.save('./'+modelName+'_first_part_model')

print("Evaluate on test data")
results = model_1.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# generate the data for the second part of the model
res_of_part_1=model_1.predict(samples_data)
samples_data=numpy.concatenate((samples_data,res_of_part_1),axis=1)
X_train, X_test, y_train, y_test = train_test_split(samples_data, samples_label, random_state=0)

# second part of the model, a softmax classifier that outputs probability distribution
model_2=keras.Sequential([
    layers.Dense(128, activation='sigmoid',name='layer_1',input_shape=(distance_k+118,)),
    layers.Dense(128, activation='sigmoid', name='layer_2'),
    layers.Dense(128, activation='sigmoid', name='layer_3'),
    layers.Dense(118, activation='softmax', name='output')
])

model_2.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

history=model_2.fit(X_train,y_train,batch_size=64,epochs=60)
model_2.save('./'+modelName+'_second_part_model')

print("Evaluate on test data")
results = model_2.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# generate the predicted labels
all_pred=model_2.predict(samples_data)
pred=[]
for i in range(len(all_pred)):
    pred.append(tf.argmax(all_pred[i]))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : The confusion matrix
    - classes : The classes corresponding to columns in the matrix
    - normalize : True: Percentage, False: Number
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(modelName+'.jpg', dpi=300)

cm=confusion_matrix(samples_label, pred)
classes=["Li","C","N","O","Na","Al","Si","P","S","K"]
plot_confusion_matrix(cm,classes,True)