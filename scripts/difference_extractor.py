import pandas as pd
import os
import json

distances_path='../COMP390/distances_by_crystal/'
bonds_path='../../bond_info/'
labels_path='../COMP390/labels/'
differences_path='./differences_notabs.json'

interested_types=['C','H','O','N']

work_types=['C','O','N']

CHON_count=[0]*4

# bond_length_by_type={}
# pairs_dict={}
# for i in range(len(interested_types)):
#     for j in range(i, len(interested_types)):
#         pairs_dict[interested_types[i]+interested_types[j]]=[]
#
# for i in range(len(interested_types)):


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
        for element in list:
            if element[2] is not None:
                return element
crystal_ids=os.walk(labels_path)

differences=[]

count=0

for path, dir_list, file_list in crystal_ids:
    # for each crystal
    for crystal_id in file_list:
        count+=1
        try:
            crystal_distances=pd.read_csv(distances_path+crystal_id+'.csv')
        except:
            continue
        with open(bonds_path+crystal_id+'.json') as f:
            crystal_bonds_dict=json.load(f)
        for index, row in crystal_distances.iterrows():
            # for i in range(len(CHON_flags)):
            #     if not CHON_flags:
            #         if row['type']==interested_types[i]:
            #             CHON_flags[i]=True
            #             CHON_count[i]+=1
            if row['type'] in work_types:
                shortest_bond_info=get_shortest_bondinfo(crystal_bonds_dict[row['center_atoms']])
                if shortest_bond_info is not None:
                    # type of the other atom forming the bond
                    the_other_type=get_type(shortest_bond_info[0])
                    if the_other_type in work_types and the_other_type==row['nn_type1']:
                        differences.append(abs(row['dist1']-shortest_bond_info[2]))
        if count==10000:
            print('mark')
# write difference
with open(differences_path, 'w') as f:
    str=json.dumps(differences)
    f.write(str)