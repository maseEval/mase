import os
file=open("run_gcloud.sh","w")
for x in os.walk("../fluent_speech_commands_dataset/wavs/"):
	speaker_name=x[0].split("/")[-1]
	for j in x[2]:
		file.write("gcloud ml speech recognize Documents/SEM-1/10701/fluent_speech_commands_dataset/wavs/speakers/"+speaker_name+"/"+j+" --language-code=en-US\n")
