from tqdm import tqdm
import numpy as np

header = []
table = []
with open("microlensing_events.ipac", "a") as write_file:
    with open('metadata.ipac', 'r') as file:
        i = 0
        for line in tqdm(file):
            if i == 0:
                write_file.write(line.strip() + "\n")
                header = line[1:-2].replace(" ", "").split('|')
            elif i > 3:
                row = line.split()
                if row[19] != "null":
                    #print("event!")
                    table.append(row)
                    write_file.write(line.strip() + "\n")
            i+=1

def remove_nan_rows(sample):
    sample.remove_rows(np.where([c.data for c in sample.mask.itercols()])[-1])


