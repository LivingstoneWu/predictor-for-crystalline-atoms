from ccdc import io
import os
from ccdc import molecule
import json

labels_path='./labels/'

bonds_info_path='/Volumes/Livingstone/bond_info/'

csd_reader=io.EntryReader('CSD')

interested_types=['C','H','O','N']

CHON_count=[0]*4

def get_bond_length(bond):
    try:
        return bond.length
    except:
        return float('-inf')

crystal_ids=os.walk(labels_path)

for path, dir_list, file_list in crystal_ids:
    # for each crystal
    for file_name in file_list:
        crystal=csd_reader.crystal(file_name)
        dict_of_crystal={}
        CHON_flags=[False]*4
        # for each atom in the atoms list within the interested types, sort its bonds by length and add it into the dict
        for atom in crystal.asymmetric_unit_molecule.atoms:
            if not atom.atomic_symbol in interested_types:
                continue
            for i in range(len(CHON_flags)):
                if not CHON_flags[i]:
                    if atom.atomic_symbol==interested_types[i]:
                        CHON_flags[i]=True
                        CHON_count[i]+=1
            bonds=list(atom.bonds)
            bonds.sort(key=get_bond_length)
            bonds_info_list=[]
            for bond in bonds:
                current_bond_info=[]
                if bond.atoms[0].label==atom.label:
                    current_bond_info.append(bond.atoms[1].label)
                else:
                    current_bond_info.append(bond.atoms[0].label)
                    current_bond_info.append(bond.bond_type.__str__())
                    try:
                        current_bond_info.append(bond.length)
                    except:
                        current_bond_info.append(None)
                    else:
                        pass
                    bonds_info_list.append(current_bond_info)
            dict_of_crystal[atom.label]=bonds_info_list
        with open(bonds_info_path+file_name+'.json','w') as f:
            str=json.dumps(dict_of_crystal)
            f.write(str)

print(CHON_count)