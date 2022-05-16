import pandas as pd
from matplotlib import pyplot as plt
import sys
import json

neighbour_no=sys.argv[1]
file_path="/Users/j.s.k/Downloads/distance_distribution_dat/CHON_distance_data.csv"
save_path="./distribution_fixed/"

def type(string):
    type_str=""
    for char in string:
        if (str.isalpha(char)):
            type_str+=char
        else:
            break
    return type_str

chunk_size=10 ** 6
elements=["C","H","O","N"]
n_bins=800

# create dictionary of pairs vectors, to store respective distance string.
hist_vectors={}
all_pairs=[]
for i in range(len(elements)):
    for j in range(i, len(elements)):
        pair_str=elements[i]+elements[j]
        all_pairs.append(pair_str)
        hist_vectors[pair_str]=[]

with pd.read_csv(file_path, chunksize=chunk_size) as reader:
    for chunk in reader:
        for index, row in chunk.iterrows():
            pair_string=type(row['type'])+row['nn_type'+str(neighbour_no)]
            if (pair_string not in all_pairs):
                continue
            hist_vectors[pair_string].append(row['dist'+str(neighbour_no)])

for key in hist_vectors:
    if (key[0:int(len(key)/2)]==key[int(len(key)/2):]):
        plt.hist(hist_vectors[key], weights=[0.5] * len(hist_vectors[key]), bins=n_bins)
    else:
        plt.hist(hist_vectors[key], bins=n_bins)
    plt.title(key+"_"+str(neighbour_no)+" distribution figure")
    plt.xlabel('nn_distance')
    plt.ylabel('Counts')
    plt.savefig(save_path+key+"_"+neighbour_no+".pdf")
    plt.cla()

dict_str=json.dumps(hist_vectors)
with open("/Users/j.s.k/Downloads/distance_distribution_dat/CHON_distance_data.csvvectors_data_neighbor_"+neighbour_no+".json", "w") as f:
    f.write(dict_str)
