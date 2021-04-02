import pickle
import numpy as np
import matplotlib.pyplot as plt
# file_array=["speaker_or_utterance_closed_splits_utterance_error_bleu_weights_1,0,0,0.pkl",\
# "speaker_or_utterance_closed_splits_utterance_error_bleu_weights_0,1,0,0.pkl",\
# "speaker_or_utterance_closed_splits_utterance_error_bleu_weights_0,0,1,0.pkl",\
# "speaker_or_utterance_closed_splits_utterance_error_bleu_weights_0,0,0,1.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_error_bleu_weights_0,0,0,1.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_error_bleu_weights_0,0,1,0.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_error_bleu_weights_0,1,0,0.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_error_bleu_weights_1,0,0,0.pkl",\
# "utterance_closed_splits_utterance_error_bleu_weights_1,0,0,0.pkl",\
# "utterance_closed_splits_utterance_error_bleu_weights_0,1,0,0.pkl",\
# "utterance_closed_splits_utterance_error_bleu_weights_0,0,1,0.pkl",\
# "utterance_closed_splits_utterance_error_bleu_weights_0,0,0,1.pkl",\
# "utterance_closed_splits_utterance_bleu_weights_0,0,0,1.pkl",\
# "speaker_or_utterance_closed_splits_utterance_bleu_weights_0,0,0,1.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_bleu_weights_0,0,0,1.pkl",\
# "utterance_closed_splits_utterance_bleu_weights_0,0,1,0.pkl",\
# "speaker_or_utterance_closed_splits_utterance_bleu_weights_0,0,1,0.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_bleu_weights_0,0,1,0.pkl",\
# "speaker_or_utterance_closed_splits_utterance_bleu_weights_0,1,0,0.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_bleu_weights_0,1,0,0.pkl",\
# "utterance_closed_splits_utterance_bleu_weights_0,1,0,0.pkl",\
# "speaker_or_utterance_closed_splits_utterance_bleu_weights_1,0,0,0.pkl",\
# "utterance_closed_splits_utility_bleu_utterance_bleu_weights_1,0,0,0.pkl",\
# "utterance_closed_splits_utterance_bleu_weights_1,0,0,0.pkl"]

file_array=["speaker_or_utterance_closed_splits_utility_noBLEU_utterance_bleu_weights_0.25,0.25,0.25,0.25.pkl","speaker_or_utterance_closed_splits_utility_noBLEU_utterance_error_bleu_weights_0.25,0.25,0.25,0.25.pkl"]

d_array=[]
for file in file_array:
	f = open(file,"rb")
	utility_dict=pickle.load(f)
	d=[]
	print(utility_dict)
	for k in utility_dict:
		d=d+[round(utility_dict[k][0],4) for i in range(utility_dict[k][1])]
	print(len(d))
	print(d)
	d_array.append(d)
n, bins, patches = plt.hist(x=d_array, bins='auto',
                            alpha=0.7, rwidth=0.85, label=['actual', 'error'])
print(n)
print(bins)
print(patches)
for item in patches[0]:
    item.set_height(item.get_height()/sum(n[0]))
for item in patches[1]:
    item.set_height(item.get_height()/sum(n[1]))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Blue Score')
plt.ylabel('Percentage of Transcripts')
plt.ylim((0,1.0))
plt.legend()
# plt.savefig(file.replace(".pkl",".png"))
plt.savefig("compare_error_noBLEU.png")
plt.show()
