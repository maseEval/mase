import json
import pandas as pd
line2_array=[]
for line in open("run_gcloud_snips.sh"):
	if line.strip()=="":
		continue
	line2_array.append(line.strip().replace("gcloud ml speech recognize Documents/SEM-1/10701/Fluent-Speech-Commands/snips_slu_data_v1.0","..").replace(" --language-code=en-US",""))
	# print(line2_array)
	# exit()
line_complete_array=[]
line_array=[]
line_count=0
for line in open("run_gcloud_snips_close.json"):
	if line.strip()=="{}":
		# print(line)
		if len(line_array)!=0:
			if (" ").join(line_array)=="] } ] }":
				print(line_array)
				print(line)
				print(line_complete_array[-10:])
				exit()
			line_complete_array.append(("\n").join(line_array))
			print(line_array)
		
		line_complete_array.append(line.strip())
		line_array=[]
		line_count=0
		# exit()
		continue
	if line.rstrip()=="{":
		if len(line_array)!=0:
			# print(line_array)
			if (" ").join(line_array)=="] } ] }":
				print(line_array)
				print(line)
				print(line_complete_array[-1])
				exit()
			line_complete_array.append(("\n").join(line_array))
			line_array=[]
			line_count=0
		# exit()
	line_array.append(line.strip())
	line_count=line_count+1
line_complete_array.append(("\n").join(line_array))
line_array=[]
line_count=0
line_transcription_array=[]
for k in range(len(line_complete_array)):
	if "transcript" not in line_complete_array[k]:
		if line_complete_array[k]=="{}":
			print(line2_array[k])
		# print(line_complete_array[k-1])
print(len(line_complete_array))
print(len(line2_array))
for j in range(len(line_complete_array)):
	if line_complete_array[j]=="{}":
		line_transcription_array.append(" ")
	else:
		transcript_arr=line_complete_array[j].split("\n")
		for transcript in transcript_arr:
			if "transcript" in transcript:
				print(transcript.replace('"transcript": ','').replace('"',''))
				line_transcription_array.append(transcript.replace('"transcript": ','').replace('"',''))
				break
# exit()
print(len(line_transcription_array))
arr=[]
for k in range(len(line_transcription_array)):
	arr.append([line2_array[k],line_transcription_array[k]])

df = pd.DataFrame(arr,columns=["audio path","predicted_words"])
df.to_csv("gcloud_snips_transcription.csv",index=False)
