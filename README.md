# predictor-for-crystalline-atoms
A predictor for atom types in crystals based on their [k-NN distances](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
The project proposal could be found in the files.

# 0. Introduction (updated on 15th Apr 2022)

## 0.0 Background Knowledge - Crystals and k-NN distances

Crystals are materials with repetitive atom arrangements in space. Since we only care about inter-atomic distances in this project, crystals are modelled as periodic point sets in 3-dimensional space.

k-NN distances: given a point $p$ in a point set $P$ and a fixed integer $k\ge 1$, k-NN distances is defined as the vector consisting of the increasingly ordered Euclidean distances $d_1\le...\le d_k$ measured from point $p$ to its first k nearest neighbours in the point set P.

## 0.1 Crystal Structure Prediction

Crystal structure prediction is the problem of predicting how molecules will crystalize given only their chemical diagrams, and perhaps, crystallization conditions.  

The current methods for CSP contain two crucial steps:

1. calculate the molecular structures based on chemical diagrams, and then generate possible crystal structures by searching the crystal packing phase space.
2. Evaluate the generated structures, and rank them in order of likelihood.

This machine learning model could serve as a quick filter applied in the second phase, to help narrow down the structures that need to be carefully assessed by suggesting very unlikely structures.



# 1. The model structure
The current model structure includes a sigmoid classifier, with 3 layers of 64 fully-connected neurons inside, and a weight of 118 applied to the true classes (the classes that are presented in the dataset) to resolve class imbalance, and a softmax with input being the combination of the output of the sigmoid classifier and the k-NN distances. The model structure was mainly inspired by [Crystal Structure Prediction via Deep Learning](https://pubs.acs.org/doi/abs/10.1021/jacs.8b03913), especially the idea of resolving class imbalance. The model is then trained with data obtained through [CSD (Cambridge Structural Database)](https://www.ccdc.cam.ac.uk/solutions/csd-core/components/csd/). 

# 2. The Dataset

## 2.1 Original dataset

The original dataset is extracted from the CSD. All entries of crystals are shuffled and iterated through. For each atom within the interested element types, 10-NN nearest neighbours, with their corresponding element types, were extracted. I gratefully acknowledge Mr. Daniel Widdowson in the group who provided the package and script required to calculate the nearest neighbour distances. (his package: [AMD](https://github.com/dwiddo/average-minimum-distance))

Then the indices (which identify every atom within the crystal???s asymmetric unit by its element type and an integer, e.g. ???H1???, ???H2???) were extracted from the CSD. The centre atoms are marked with indices from the database for further analysis. Those atoms are split into smaller files by crystals they belong to and stored with .pickle format for better performance.



# 3. Updates

## 3.1 8th Dec 2021
### Current results
Models were trained with k from 1 to 5 based on data of 10 elements in total extracted from the CSD. There is an upper limit of 10,000 samples for each element (otherwise the number of samples is simply the number of occurrences of the element in the database). The model's performance is shown in the form of confusion matrices in the figures below, ordered by k from 1 to 5.
![CNO_k_1](https://user-images.githubusercontent.com/52390858/145274061-66dfe235-a073-4410-8d03-ae9e66bbf47d.jpg)
![CNO_k_2](https://user-images.githubusercontent.com/52390858/145274073-ec1d8237-32f4-45b8-b3a0-5e04a9f0969e.jpg)
![CNO_k_3](https://user-images.githubusercontent.com/52390858/145274087-684cddae-17c3-44fc-b243-5591e27077ee.jpg)
![CNO_k_4](https://user-images.githubusercontent.com/52390858/145274094-c9216d0a-0f92-4a78-8b28-15ac76014982.jpg)
![CNO_k_5](https://user-images.githubusercontent.com/52390858/145274102-cbb57baa-e3e0-4d7b-9066-1b252f4d57dc.jpg)

The model seems to have difficulty on distinguishing Aluminium from Lithium, which might be able to be explained chemically. 

### Next steps 
20 elements would be selected from the CSD by abundance, and all pairs of elements that the model make misclassifications would be picked out. In addition, the relationship between the selection of the hyperparameter k and the performance of the model would be investigated.

## 3.2 5th Jan 2022
### Current results
Data of 20 most abundant elements in the CSD were extracted. There were 940k samples in total. The model was trained on k=5 and the result is shown below.
![top20_k=5](./ml_results/top20_elements/top20_k=5.png)

### Next steps
Since there are too much data that would be too slow to fit in the memory, I decided to switch to [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Later on, a figure showing how the performance (represented by confusion matrix) is developing with the increase of k shall be plotted. In addition the particular crystal IDs, and the elements where the model made misclassification shall be extracted and analyzed. 

## 3.3 25th Feb 2022

### Current results

The model has been trained on k=1-10. The results can be found in `/results/top20_elements/`

### Next steps

The model appeared to be predictive. 

We got curious about how the distribution of distances between possible pairs of elements being nearest neighbor would be like (e.g. C-C being nearest neighbors, 3 peaks expected in the distribution). This analysis could help explaining the model???s predictive power.

## 3.4 8th Apr 2022

### Current results

**Distribution of nearest neighbour distances by pairs of element types**:

To keep the research simple but not trivial, all atoms with their element types and first nearest neighbour within the types {C, H, O, N} have been extracted. The total number of samples is about 17 millions.

The distribution of the first nearest neighbour distances have been plotted. Then the plot is divided by possible atom type pairs. 

Here is the overall distribution of the first nearest neighbour distances: 
![all.png](./nn_distribution/all_1.png)

Then, these distributions are grouped by pairs of element types (e.g. center atom being Carbon and first nearest neighbour Carbon (C, C) ) and plotted:
![cc](./nn_distribution/CC_1.png)
It followed our expectation of 3 peaks, as there are 3 valences for C and they have different covalent bond lengths.

Histograms of other element types can be found in the directory `./nn_distribution`.

Then we developed an interest that how many shortest nearest neighbours are actually forming a bond, and if yes, how the distribution of these bond lengths are composed of various bond types? 

We expect that the shortest nearest neighbours are mainly composed of bonds and each bond type would form a peak. In such case, it suggest that the model???s being predictive is partially because the neighbour distances contains chemical information on bond type.

**Extracting bonds from nearest neighbours**

As mentioned before, each atom is labelled with index, while the neighbours are not. The problem is, the CSD only gave index to atoms within the asymmetric unit, therefore it is hard to distinguish atoms between different asymmetric units and unit cells. Nevertheless, the CSD provided bond types and length information with corresponding two atoms labelled with the indices. We assumed that, if the distance and the bond length provided by the database have a difference smaller than some particular threshold, the neighbour should be the atom that is forming the particular bond. Based on the above assumption, the bond lengths are extracted by atom, and then the difference between the shortest nearest neighbour distance and shortest bond length was computed and plotted in a histogram to find an appropriate threshold. H atoms were eliminated since they made up a big proportion while having only a few bond types. See extraction script in `/scripts/bond_extractor.py` and difference calculation script in `/scripts/difference_extractor.py`

For each atom above, the shortest bond is extracted (if there is one). Here atoms with some bond length not registered were thrown away to avoid possible mismatches. Then the difference of the shortest bond length and nearest neighbour distance is plotted:

![differences_none_removed_abs](./differences/differences_none_removed_abs.png)

There is a very high peak near 0, as expected. The threshold was initially set to 0.05??, and all shortest bonds with matching element types and having length difference less than the threshold were extracted. 

Here is how the overall distribution of shortest bond lengths look like in nearest neighbour distance:

![overall_bonds](./bond_within_neighbour/all.png)

Here are some examples plotting bonds within neighbours:

![CC_bond](./bond_within_neighbour/CC.png)

The plot above contains too much bond types and is not easy to distinguish between. See `./bond_plot_CC/` for breaked down plots. They have been only initially processed, a figure with better quality would probably be updated shortly. 

### distribution of single bond under nearest neighbours:

![bond_CC_Single](./bond_plots_CC/CCSingle.png)

### distribution of double bond under nearest neighbours: 

![cc_double](./bond_plots_CC/CCDouble.png)

## Limitations & Suggestions for future research(Updated 16th May)

As the project is approaching an end, the limitations and future directions are discussed here. 

### Limitations

* Machine learning model:

  The model 

