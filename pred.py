import tensorflow as tf
import tensorflow.keras as keras
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataPath='/Users/j.s.k/Downloads/extractor/AlSiPS_distance_data.csv'
periodic_dictionary={"H":0, "He":1, "Li":2, "Be":3, "B":4, "C":5, "N":6, "O":7, "F":8, "Ne":9, "Na":10, "Mg":11, "Al":12, "Si":13, "P":14, "S":15, "Cl":16, "Ar":17, "K":18, "Ca":19, "Sc":20, "Ti":21, "V":22, "Cr":23, "Mn":24, "Fe":25, "Co":26, "Ni":27, "Cu":28, "Zn":29, "Ga":30, "Ge":31, "As":32, "Se":33, "Br":34, "Kr":35, "Rb":36, "Sr":37, "Y":38, "Zr":39, "Nb":40, "Mo":41, "Te":42, "Ru":43, "Rh":44, "Pd":45, "Ag":46, "Cd":47, "In":48, "Sn":49, "Sb":50, "Te":51, "I":52, "Xe":53, "Cs":54, "Ba":55, "La":56, "Ce":57, "Pr":58, "Nd":59, "Pm":60, "Sm":61, "Eu":62, "Gd":63, "Tb":64, "Dy":65, "Ho":66, "Er":67, "Tm":68, "Yb":69, "Lu":70, "Hf":71, "Ta":72, "W":73, "Re":74, "Os":75, "Ir":76, "Pt":77, "Au":78, "Hg":79, "Tl":80, "Pb":81, "Bi":82, "Po":83, "At":84, "Rn":85, "Fr":86, "Ra":87, "Ac":88, "Th":89, "Pa":90, "U":91, "Np":92, "Pu":93, "Am":94, "Cm":95, "Bk":96, "Cf":97, "Es":98, "Fm":99, "Md":100, "No":101, "Lr":102, "Rf":103, "Db":104, "Sg":105, "Bh":106, "Hs":107, "Mt":108, "Ds":109, "Rg":110, "Cn":111, "Nh":112, "Fl":113, "Mc":114, "Lv":115, "Ts":116, "Og":117, "Uue":118,}
# tf.config.experimental.list_physical_devices('GPU')

def element_in_list(element, list):
    for i in list:
        if (i==element): return True
    return False

def turnIntoLabel(list):
    list_labels=[]
    appeared_elements=[]
    for item in list:
        if (not element_in_list(item, appeared_elements)):
            appeared_elements.append(item)
        list_labels.append(periodic_dictionary[item])
    return np.asarray(list_labels), appeared_elements

dataset=pandas.read_csv(dataPath, skiprows=[0]).to_numpy()

samples_label, appeared_elements=turnIntoLabel(dataset[:,0])
samples_data=dataset[:,1:].astype('float32')
model_1=keras.models.load_model('./first_part_model')
res_of_part_1=model_1.predict(samples_data)
samples_data=np.concatenate((samples_data,res_of_part_1),axis=1)

print(samples_label.shape)
model_2=keras.models.load_model('./second_part_model')
pred_y=model_2.predict(samples_data)
pred=[]
for i in range(len(pred_y)):
    pred.append(tf.argmax(pred_y[i]))

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.figure(figsize=(48,60))
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')  

    # normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # set cells that have percentage under 1% to avoid confusion
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    # plot the grids
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    # rotate the labels for better view
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    # mark the percentages
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=300)
    plt.show()

labels_name=["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S"]
cm=confusion_matrix(samples_label, pred)
plot_Matrix(cm,labels_name)
plt.show()