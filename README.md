# predictor-for-crystalline-atoms
A predictor for atom types in crystals based on their k-NN distances.
The project proposal could be found in the files.

# 1. The model structure
Current model structure includes a sigmoid classifier, with 3 layers of 64 fully-connected neurons inside, and a weight of 118 applied to the true classes (the classes that presented in the dataset) to resolve class imbalance, and a softmax with input being the combination of the output of the sigmoid classifier and the k-NN distances. The model is then trained with 10000 samples on each element, which was obtained through CSD (Cambridge Structural Database). 

# 2. Updates
## 2.1 8 Dec 2021
### Current
![CNO_k_1](https://user-images.githubusercontent.com/52390858/145274061-66dfe235-a073-4410-8d03-ae9e66bbf47d.jpg)
![CNO_k_2](https://user-images.githubusercontent.com/52390858/145274073-ec1d8237-32f4-45b8-b3a0-5e04a9f0969e.jpg)
![CNO_k_3](https://user-images.githubusercontent.com/52390858/145274087-684cddae-17c3-44fc-b243-5591e27077ee.jpg)
![CNO_k_4](https://user-images.githubusercontent.com/52390858/145274094-c9216d0a-0f92-4a78-8b28-15ac76014982.jpg)
![CNO_k_5](https://user-images.githubusercontent.com/52390858/145274102-cbb57baa-e3e0-4d7b-9066-1b252f4d57dc.jpg)

### Next steps 
20 elements would be selected from the CSD by abundance, and all pairs of elements that the model make misclassifications would be picked out. In addition, the relationship between the selection of the hyperparameter k and the performance of the model would be investigated.

## 2.2 5 Jan 2022


## 5 Jan 2022
20 elements were picked out based on abundancy in the CSD (Cambridge Structural Database).

The model seems to have difficulty on distinguishing Aluminium from Lithium, which might be able to explain chemically. 


