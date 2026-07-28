import csv
import os
import glob

all_results = []
for file in glob.glob("validation/results/*_results.csv"):
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_results.append(row)

if all_results:
    keys = all_results[0].keys()
    with open('validation/results/master_summary.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_results)
