import pickle
import json
from matplotlib import pyplot as plt

nn_distances_path="./nn_1_distances.pickle"
bond_distances_path="./bond_distances_noneRemoved_threshold=0.05.pickle"
save_directory="./bond_plots_CC/"
differences_path='./difference_notabs_noneRemoved/differences_notabs.pickle'

n_bins=800

with open(nn_distances_path, 'rb') as f:
    nn_distances_dict=pickle.load(f)

with open(bond_distances_path, 'rb') as f:
    bond_distances_dict=pickle.load(f)

with open(differences_path, 'rb') as f:
    differences=pickle.load(f)

# all_distances_list=[]
# all_bonds_list=[]
# for key in bond_distances_dict:
#     all_distances_list+=nn_distances_dict[key]
#     # log_flag=len(nn_distances_dict[key])>10**4
#     log_flag=True
#     plt.hist(nn_distances_dict[key], bins=n_bins, alpha=0.3, label='nn_distances', log=log_flag)
#     for bond_type in bond_distances_dict[key]:
#         all_bonds_list+=bond_distances_dict[key][bond_type]
#         plt.hist(bond_distances_dict[key][bond_type], alpha=0.3, bins=n_bins, label=bond_type, log=log_flag)
#     plt.title("("+key[0]+","+key[1]+") distribution of nn_distances and bond lengths")
#     plt.ylabel("number of pairs at distance")
#     plt.xlabel("distance/bond length (Å)")
#     plt.legend(loc='upper left', prop={'size':6})
#     plt.savefig(save_directory+key+".png", dpi=400, bbox_inches="tight", pad_inches=0.5)
#     plt.cla()

# plt.hist(all_distances_list, bins=n_bins, alpha=0.3, label="nn_distances", log=True)
# plt.hist(all_bonds_list, bins=n_bins, alpha=0.3, label="bond_length", log=True)
# plt.ylabel("number of pairs at distance")
# plt.xlabel("distance/bond length (Å)")
# plt.title("distribution of all nn_distances and bond lengths (C,N,O)")
# plt.legend(loc='upper left', prop={'size':6})
# plt.savefig(save_directory+"all.png", dpi=400, bbox_inches="tight", pad_inches=0.5)

all_bonds=[]
for bond_type in bond_distances_dict['CC']:
    all_bonds+=bond_distances_dict['CC'][bond_type]
    print(bond_type+' '+str(len(bond_distances_dict['CC'][bond_type])))
    plt.hist(nn_distances_dict["CC"], bins=n_bins, alpha=0.3, label='nn_distances', log=True)
    plt.hist(bond_distances_dict['CC'][bond_type], alpha=0.3, bins=n_bins, label=bond_type, log=True)
    plt.title("(C,C) distribution of nn_distances and bond lengths")
    plt.ylabel("number of pairs at distance")
    plt.xlabel("distance/bond length (Å)")
    plt.legend(loc='upper left', prop={'size':6})
    plt.savefig(save_directory+"CC"+bond_type+".png", dpi=400, bbox_inches="tight", pad_inches=0.5)
    plt.cla()
plt.hist(nn_distances_dict["CC"], bins=n_bins, alpha=0.3, label='nn_distances', log=True)
plt.hist(all_bonds, alpha=0.3, bins=n_bins, label="bonds", log=True)
plt.title("(C,C) distribution of nn_distances and bond lengths")
plt.ylabel("number of pairs at distance")
plt.xlabel("distance/bond length (Å)")
plt.legend(loc='upper left', prop={'size':6})
plt.savefig(save_directory+"CC_all.png", dpi=400, bbox_inches="tight", pad_inches=0.5)
plt.cla()