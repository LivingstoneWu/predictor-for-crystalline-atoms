import pickle
import pandas as pd
import os
import json
import datetime

distances_path='../COMP390/distances_by_crystal/'
bonds_path='../../bond_info/'
crystal_ids_list_path='./crystal_ids_list.pickle'

difference_threshold=0.05
work_types=['C','O','N']

bond_distances_save_path='./bond_distances_noneRemoved_threshold=0.05.pickle'

def get_type(string):
    type_str=""
    for char in string:
        if (str.isalpha(char)):
            type_str+=char
        else:
            break
    return type_str

def get_shortest_bondinfo(list):
    if not list:
        return None
    else:
        #throw values containing None
        if list[0][2] is not None:
            return list[0]
    return None

with open(crystal_ids_list_path,'rb') as f:
    crystal_ids=pickle.load(f)

bond_distances_in_pair={}
for i in range(len(work_types)):
    for j in range(len(work_types)):
        bond_distances_in_pair[work_types[i]+work_types[j]]= {"Unknown":[],
                                                            "Single":[],
                                                            "Double":[],
                                                            "Triple":[],
                                                            "Quadruple":[],
                                                            "Aromatic":[],
                                                            "Delocalised":[],
                                                            "Pi":[]}


total_count=len(crystal_ids)
count=0
current_time=datetime.datetime.now()

for crystal_id in crystal_ids:
    count+=1
    try:
        crystal_nn_distances=pd.read_csv(distances_path+crystal_id+'.csv')
    except:
        continue
    with open(bonds_path+crystal_id+'.json') as f:
        crystal_bonds_dict=json.load(f)
    for index, row in crystal_nn_distances.iterrows():
        if row['type'] in work_types:
            shortest_bond_info = get_shortest_bondinfo(crystal_bonds_dict[row['center_atoms']])
            if shortest_bond_info is not None:
                # type of the other atom forming the bond
                the_other_type = get_type(shortest_bond_info[0])
                if the_other_type in work_types and the_other_type == row['nn_type1']:
                    # then, if the distance is shorter than the threshold the neighbour is considered forming a bond with this atom
                    if abs(row['dist1']-shortest_bond_info[2])<difference_threshold:
                        bond_distances_in_pair[row['type']+the_other_type][shortest_bond_info[1]].append(shortest_bond_info[2])
    # print an indication of running time
    if count%10000==0:
        print(str(count // 10000) + '/' + str(total_count // 10000))
        print("time consumed for 10000 crystals:", (datetime.datetime.now() - current_time).seconds)
        current_time = datetime.datetime.now()
with open(bond_distances_save_path,'wb') as f:
    pickle.dump(bond_distances_in_pair, f)
