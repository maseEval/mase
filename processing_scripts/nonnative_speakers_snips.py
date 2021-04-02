import pickle
import pandas as pd
import os
from jiwer import wer
import json
data_str="data/"
data_str=data_str+"original_splits"
data_dir="snips_slu_data_v1.0/close_field_splits/"
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
print(len(train_path))
print(len(valid_path))
print(len(test_path))
exit()

transcription_set = pd.read_csv("gcloud_snips_transcription.csv")
transcription_dict={}
for k in transcription_set.values:
	string_arr=k[1]
	str2_arr=[]
	prev_str=""
	print(string_arr)
	if k[0] not in transcription_dict:
		transcription_dict[k[0]]=string_arr

speaker_demographic=json.load(open("snips_slu_data_v1.0/smart-lights-en-close-field/speech_corpus/metadata.json","r"))
speaker_demo_dict={}
map_columns={}
# print(len(speaker_demographic))
transcript_speaker_dict={}
for k in speaker_demographic:
	transcript_speaker_dict["../smart-lights-en-close-field/speech_corpus/audio/"+str(k)+".wav"]=speaker_demographic[k]['worker']["id"] 
	if speaker_demographic[k]['worker']["id"] not in  speaker_demo_dict:
		speaker_demo_dict[speaker_demographic[k]['worker']["id"]]={}
	for j in speaker_demographic[k]['worker']:
		if j=="id":
			continue
		if j not in map_columns:
			map_columns[j]={}
		if j=="age":
			if speaker_demographic[k]['worker']['age']<35:
				speaker_class=1
			else:
				speaker_class=2
			if speaker_class  not in map_columns[j]:
				map_columns[j][speaker_class]=len(map_columns[j])  
			speaker_demo_dict[speaker_demographic[k]['worker']["id"]][j]=map_columns[j][speaker_class]
		else:
			if speaker_demographic[k]['worker'][j] not in map_columns[j]:
				map_columns[j][speaker_demographic[k]['worker'][j]]=len(map_columns[j])
			speaker_demo_dict[speaker_demographic[k]['worker']["id"]][j]=map_columns[j][speaker_demographic[k]['worker'][j]]
speaker_transcript_dict={}
for k in transcript_speaker_dict:
	if speaker_demo_dict[transcript_speaker_dict[k]]["country"] not in speaker_transcript_dict:
		speaker_transcript_dict[speaker_demo_dict[transcript_speaker_dict[k]]["country"] ]=[]
	if transcript_speaker_dict[k] not in speaker_transcript_dict[speaker_demo_dict[transcript_speaker_dict[k]]["country"]]:
		speaker_transcript_dict[speaker_demo_dict[transcript_speaker_dict[k]]["country"]].append(transcript_speaker_dict[k])
print(map_columns["country"])
# for j in speaker_transcript_dict:
# 	print(j)
# 	print(len(speaker_transcript_dict[j]))
# exit()

# for country in map_columns["country"]:
# 	print(country)
sum_WER=0
sum_length=0
	# store_native_value=map_columns["country"][country]
for k in gold_transcription_dict:
	if speaker_demo_dict[transcript_speaker_dict[k]]["age"]==1:
		sum_WER=sum_WER+(wer(gold_transcription_dict[k], transcription_dict[k])*len(gold_transcription_dict[k]))
		sum_length=sum_length+len(gold_transcription_dict[k])
print(sum_WER/sum_length)
# print()


