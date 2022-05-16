import pandas as pd

# path to the directory storing labels
label_directory='./labels/'

# path to the distances file
distance_file='/Users/j.s.k/Downloads/distance_distribution_dat/CHON_distance_data.csv'

# output path
output_path='./distances_by_crystal/'

# error log
error_log_path='./indices_dont_match_log.txt'

# types of elements interested
interested_types=['C','H','O','N']

def get_element_type(string):
    result=''
    for char in string:
        if char.isalpha():
            result+=char
    return result

def remove_uninterested(list, interested_types=['C','H','O','N']):
    res=list.copy()
    for element in list:
        if not get_element_type(element) in interested_types:
            res.remove(element)
    return res


chunk_size=100000
last_id=None
current_series_dic={}
row_index=0
error_log=open(error_log_path,'w')
with pd.read_csv(distance_file, chunksize=chunk_size) as reader:
    for chunk in reader:
        for index, row in chunk.iterrows():
            # if this row is the starting point of a new crystal
            if row['ID']!=last_id:
                if current_series_dic!={}:
                    current_dataframe=pd.DataFrame.from_dict(current_series_dic, orient="index", columns=row.index)
                    try:
                        f=open(label_directory+last_id)
                        lines=f.readlines()
                        labels=lines[1].split(',')
                        labels.remove('')
                        f.close()
                    except:
                        pass
                    else:
                        try:
                            current_dataframe.insert(1, "center_atoms", labels)
                        except:
                            error_log.write('Crystal ID: '+last_id+'\n')
                            error_log.write('labels extracted from CSD: \n')
                            error_log.write(str(labels))
                            error_log.write('atom types in the dataset: \n')
                            error_log.write(str(current_dataframe['type'].tolist())+'\n')
                        else:
                            current_dataframe.to_csv(path_or_buf=output_path + last_id + '.csv')
                current_series_dic={}
                row_index=0
                current_series_dic[row_index] = row
                row_index+=1
                last_id=row['ID']
            else:
                current_series_dic[row_index]=row
                row_index+=1
error_log.close()