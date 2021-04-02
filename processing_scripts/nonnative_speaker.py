import pickle
import pandas as pd
import os
from jiwer import wer
data_str="data/"
data_str=data_str+"original_splits"
data_dir="../fluent_speech_commands_dataset/"
train_file = os.path.join(data_dir, data_str, "train_data.csv")

test_file = os.path.join(data_dir, data_str, "test_data.csv")
test_set = pd.read_csv(test_file)
valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
train_set = pd.read_csv(train_file)    
valid_set = pd.read_csv(valid_file)
gold_transcription_dict={}
test_path=test_set.path.values
count=0
for sentence in test_set.transcription.values:
    if test_path[count] not in gold_transcription_dict:
        sent_arr=sentence.strip().lower()
        gold_transcription_dict[test_path[count]]=sent_arr
    count=count+1
train_path=train_set.path.values
count=0
for sentence in train_set.transcription.values:
    if train_path[count] not in gold_transcription_dict:
        sent_arr=sentence.strip().lower()
        gold_transcription_dict[train_path[count]]=sent_arr
    count=count+1
valid_path=valid_set.path.values
count=0
for sentence in valid_set.transcription.values:
    if valid_path[count] not in gold_transcription_dict:
        sent_arr=sentence.strip().lower()
        gold_transcription_dict[valid_path[count]]=sent_arr
    count=count+1

transcription_set = pd.read_csv("gcloud_transcription.csv")
transcription_dict={}
for k in transcription_set.values:
    string_arr=k[1]
    str2_arr=[]
    prev_str=""
    print(string_arr)
    if k[0] not in transcription_dict:
        transcription_dict[k[0]]=string_arr

insertion_dict=pickle.load(open("caches/original_splits_insert_complete_speaker_WER.pkl","rb"))
replace_dict=pickle.load(open("caches/original_splits_replace_complete_speaker_WER.pkl","rb"))
delete_dict=pickle.load(open("caches/original_splits_delete_complete_speaker_WER.pkl","rb"))
sum_WER=0
sum_length=0
for k in insertion_dict:
	speaker=k.split("/")[-2]
	if speaker not in ["OepoQ9jWQztn5ZqL","EExgNZ9dvgTE3928","9MX3AgZzVgCw4W4j"]:
		# print(k)
		sum_WER=sum_WER+(insertion_dict[k]*len(gold_transcription_dict[k]))
		sum_length=sum_length+len(gold_transcription_dict[k])
print(sum_WER/sum_length)

sum_WER=0
sum_length=0
for k in replace_dict:
	speaker=k.split("/")[-2]
	if speaker not in ["OepoQ9jWQztn5ZqL","EExgNZ9dvgTE3928","9MX3AgZzVgCw4W4j"]:
		# print(k)
		sum_WER=sum_WER+(replace_dict[k]*len(gold_transcription_dict[k]))
		sum_length=sum_length+len(gold_transcription_dict[k])
print(sum_WER/sum_length)

sum_WER=0
sum_length=0
for k in delete_dict:
	speaker=k.split("/")[-2]
	if speaker not in ["OepoQ9jWQztn5ZqL","EExgNZ9dvgTE3928","9MX3AgZzVgCw4W4j"]:
		# print(k)
		sum_WER=sum_WER+(delete_dict[k]*len(gold_transcription_dict[k]))
		sum_length=sum_length+len(gold_transcription_dict[k])
print(sum_WER/sum_length)


sum_WER=0
sum_length=0
for k in delete_dict:
	speaker=k.split("/")[-2]
	if speaker not in ["OepoQ9jWQztn5ZqL","EExgNZ9dvgTE3928","9MX3AgZzVgCw4W4j"]:
		# print(k)
		sum_WER=sum_WER+(wer(gold_transcription_dict[k], transcription_dict[k])*len(gold_transcription_dict[k]))
		sum_length=sum_length+len(gold_transcription_dict[k])
print(sum_WER/sum_length)
