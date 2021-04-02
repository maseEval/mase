'''
This script updates the paths in the Fluent Speech Commands synthetic data to use relative paths
instead of absolute paths. The original synthetic dataset (https://zenodo.org/record/3509828#.X8LKsxNKjOQ)
contains absolute paths to /home/lugosch/data/fluent_speech_commands_dataset/..., which this script updates
to be useful.

You don't need to run this if you downloaded data from my Google Drive.
'''

import csv

if __name__ == "__main__":
rows = []
synthetic_data_path = '/home/ec2-user/fluent_speech_commands_dataset/data/synthetic_data.csv'
with open(synthetic_data_path, newline='\n') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for row in spamreader:
    rows.append(row)

for i in range(len(rows)):
  if i is 0:
    continue
  rows[i][0] = "".join(["wavs_synth", rows[i][0].split("wavs")[1]])


with open('synthetic_data.csv', 'w', newline="\n") as out_csv:
  csvwriter = csv.writer(out_csv, delimiter=',')
  for r in rows:
    csvwriter.writerow(r)
