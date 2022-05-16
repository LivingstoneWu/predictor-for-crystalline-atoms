import random
import numpy as np
import pandas as pd
import amd
from amd.utils import ETA

# random.seed(1)

# path to the big data file
CSD_data_path = r'./CSD_data.hdf5'
reader = amd.SetReader(CSD_data_path)

# elements to find, can be any size
elems = ['H','B','C','N','O','F','Si','P','S','Cl','Fe','Co','Ni','Cu','Zn','Se','Br','Mo','Ru','I']

# number of distances/columns
k = 10
print("working...")
# get this many (random) samples for each element.
# So total samples = n_samples * len(elems)
# If None, get all possible samples for each element.
# Useful if there is a lot of some element
n_samples = 2000

reader = amd.SetReader(CSD_data_path)
elem_set = set(elems)
# create a set of elements

names = list(reader.keys())

random.shuffle(names)

labels = []
data = []
sample_counts = {elem: 0 for elem in elems}

eta = ETA(len(names))  # estimated time of arrival (execution)

for name in names:

    periodic_set = reader[name]

    if elem_set.intersection(set(periodic_set.types)):

        all_types = [periodic_set.types[i] for i in periodic_set.asymmetric_unit]  # all possible types in the crystal
        reduced_types_inds = np.array([i for i, t in enumerate(all_types) if t in elem_set])  # atoms being the specific type
        types_inds = periodic_set.asymmetric_unit[reduced_types_inds]  # index of these atoms in the crystal

        pdd, _, inds = amd.nearest_neighbours(periodic_set.motif, periodic_set.cell, k, asymmetric_unit=types_inds)
        # inds % len(periodic_set.motif
        # print([periodic_set.types[i] for i in )])

        for i, dist_row, inds_row in zip(reduced_types_inds, pdd, inds):
            label = all_types[i]
            if n_samples is None or sample_counts[label] < n_samples:
                labels.append(label)


                nn_types = [periodic_set.types[i] for i in inds_row % len(periodic_set.motif)]


                data.append((name, *dist_row, *nn_types))
                sample_counts[label] += 1

        if n_samples is not None:
            if all(count == n_samples for count in sample_counts.values()):
                break

    eta.update()

data = np.array(data)

print('Samples found for each element:', sample_counts)
print('Total samples:', sum(sample_counts.values()))

cols = ['ID'] + ['dist' + str(i) for i in range(1, k + 1)] + ['nn_type' + str(i) for i in range(1, k + 1)]
df = pd.DataFrame(data, index=labels, columns=cols)
filename = ''.join(elems)
df.to_csv(path_or_buf='./testTop20_distance_data.csv')