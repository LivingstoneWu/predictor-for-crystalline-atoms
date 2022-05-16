import pandas as pd
from ccdc import io

# path to save labels
labels_path="./labels/"
# path to the distances file
distances_path="/Users/j.s.k/Downloads/distance_distribution_dat/CHON_distance_data.csv"
# path to save the absent ids
absent_ids_path="./absent_ids.txt"
chunk_size=100000

csd_reader=io.EntryReader("CSD")
absent_file=open(absent_ids_path, "w")

last_id=None
with pd.read_csv(distances_path, chunksize=chunk_size) as reader:
    for chunk in reader:
        for index, row in chunk.iterrows():
            if (row['ID']==last_id):
                continue
            else:
                last_id=row['ID']
                try:
                    crystal=csd_reader.crystal(last_id)
                    with open(labels_path+last_id, "w+") as f:
                        f.write(last_id+'\n')
                        # write labels of atoms within the asymmetric unit in order
                        for atom in crystal.asymmetric_unit_molecule.atoms:
                            f.write(atom.label+',')
                except RuntimeError:
                    absent_file.write(str(last_id)+'\n')
                else:
                    continue
absent_file.close()