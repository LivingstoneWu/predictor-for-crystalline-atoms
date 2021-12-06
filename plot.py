import itertools
import matplotlib.pyplot as plt
import numpy as np

# A plotter for the output data
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : the confusion matrix value
    - classes : The classes, corresponding to columns
    - normalize : True: persentage; False: number
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
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
    plt.savefig('cm.jpg', dpi=300)

classes=["Al","Si","P","S"]
cm=np.float32([[0.8161506181240785,0.05013042985142339,0.07281388227288194,0.0609050697516162],[0.0846,0.7673,0.1031,0.045],[0.0306,0.113,0.7742,0.0822],[0.0063006300630063005,0.0014001400140014,0.029302930293029304,0.962996299629963]])
plot_confusion_matrix(cm,classes,True)