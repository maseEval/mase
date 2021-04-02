'''
python processing_scripts/join_error_analysis_and_word_csvs.py --error_analysis_path /home/ec2-user/Fluent-Speech-Commands/end-to-end-SLU/error_analysis.csv\ 
--word_transcripts_path /tmp/word_transcriptions.csv  --joined_file /home/ec2-user/Fluent-Speech-Commands/end-to-end-SLU/error_analysis_with_transcripts.csv
'''
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_analysis_path', required=True, help='Path to error analysis csv file')
    parser.add_argument('--word_transcripts_path', required=True, help='Path to word transcripts csv file')
    parser.add_argument('--joined_file', required=True, help='Path to csv file containing errors and word transcripts')
    args = parser.parse_args()
    error_analysis_path = args.error_analysis_path
    word_transcripts_path = args.word_transcripts_path
    joined_file = args.joined_file

    with open(error_analysis_path) as error_csvfile:
        error_analysis_csv = list(csv.reader(error_csvfile, delimiter=','))

    with open(word_transcripts_path) as transcripts_csvfile:
        word_transcripts_csv = list(csv.reader(transcripts_csvfile, delimiter=','))

    for i in range(len(error_analysis_csv)):
        if i == 0:
            continue
        error_path = error_analysis_csv[i][0]
        for w in word_transcripts_csv:
            if w[0] == error_path:
                error_analysis_csv[i].append(w[1])

    error_analysis_csv[0].append("Word Transcripts")

    with open(joined_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for row in error_analysis_csv:
            csvwriter.writerow(row)