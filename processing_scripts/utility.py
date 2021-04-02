from collections import defaultdict
import numpy as np
import os
import pandas as pd

import json
import argparse
from string import punctuation
from probability_utils import KL_divergence, compute_intent_distribution, compute_intent_distribution_combine
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pickle
from jiwer import wer
from analysis_utils import Utils
import matplotlib.pyplot as plt


def get_uniCounts(train_dataset):
    unigram_table = {}
    count=0
    for all_tokens in train_dataset.transcription:
        all_tokens=all_tokens.strip().lower().strip(punctuation).split(" ")
        count=count+len(all_tokens)
        for token in all_tokens:
            if token in unigram_table:
                unigram_table[token] += 1
            else:
                unigram_table[token] = 1
    return unigram_table,count

def get_biCounts(train_dataset):
    # uniCounts, length = get_uniCounts(train_dataset)
    bigram_table = {}
    num_bigrams = 0
    bigram_count = 0
    for all_tokens in train_dataset.transcription:
        all_tokens=all_tokens.strip().lower().strip(punctuation).split(" ")
        bigram_count=bigram_count+len(all_tokens) -1
        for x in range(0, len(all_tokens) - 1):
            if all_tokens[x] in bigram_table:
                if all_tokens[x + 1] in bigram_table[all_tokens[x]]:
                    bigram_table[all_tokens[x]][all_tokens[x + 1]] += 1
                else:
                    bigram_table[all_tokens[x]][all_tokens[x + 1]] = 1
                    num_bigrams += 1
            else:
                bigram_table[all_tokens[x]] = {}
                bigram_table[all_tokens[x]][all_tokens[x + 1]] = 1
                num_bigrams += 1
    return bigram_table, num_bigrams, bigram_count

def get_triCounts(train_dataset):
    # uniCounts, length = get_uniCounts(train_dataset)
    trigram_table = {}
    num_trigrams = 0
    trigram_count = 0
    for all_tokens in train_dataset.transcription:
        all_tokens=all_tokens.strip().lower().strip(punctuation).split(" ")
        trigram_count=trigram_count+len(all_tokens) -2
        for x in range(0, len(all_tokens) - 2):
            if all_tokens[x] in trigram_table:
                if all_tokens[x + 1] in trigram_table[all_tokens[x]]:
                    if all_tokens[x+2] in trigram_table[all_tokens[x]][all_tokens[x+1]]:
                        trigram_table[all_tokens[x]][all_tokens[x + 1]][all_tokens[x+2]] += 1
                    else:
                        trigram_table[all_tokens[x]][all_tokens[x + 1]][all_tokens[x+2]] = 1
                        num_trigrams += 1
                else:
                    trigram_table[all_tokens[x]][all_tokens[x + 1]] = {}
                    trigram_table[all_tokens[x]][all_tokens[x + 1]][all_tokens[x + 2]] = 1
                    num_trigrams += 1
            else:
                trigram_table[all_tokens[x]]={}
                trigram_table[all_tokens[x]][all_tokens[x + 1]] = {}
                trigram_table[all_tokens[x]][all_tokens[x + 1]][all_tokens[x + 2]] = 1
                num_trigrams += 1
    return trigram_table, num_trigrams, trigram_count


def get_utterance_utility(train_dataset,test_dataset):
    if len(test_dataset)==0:
        return float('-inf')
    unigram_table, unigram_count=get_uniCounts(train_dataset)
    uni_prob=0
    for row in test_dataset.transcription:
        row=row.strip().lower().strip(punctuation).split(" ")
        for k in row:
            if k in unigram_table:
                # given_prob=(unigram_table[k]/unigram_count)
                uni_prob = uni_prob+ unigram_table[k]
    uni_utility=(uni_prob/(unigram_count*len(test_dataset.transcription)))
    # print(uni_prob)

    bigram_table, num_bigrams, bigram_count=get_biCounts(train_dataset)
    bi_prob=0
    for row in test_dataset.transcription:
        row=row.strip().lower().strip(punctuation).split(" ")
        for k in range(0,len(row)-1):
            if row[k] in bigram_table:
                if row[k+1] in bigram_table[row[k]]:
                    bi_prob = bi_prob+ bigram_table[row[k]][row[k+1]]
    bi_utility=(bi_prob/(bigram_count*len(test_dataset.transcription)))

    trigram_table, num_trigrams, trigram_count=get_triCounts(train_dataset)
    if trigram_count==0:
        tri_utility=0
    else:
        tri_prob=0
        for row in test_dataset.transcription:
            row=row.strip().lower().strip(punctuation).split(" ")
            for k in range(0,len(row)-2):
                if row[k] in trigram_table:
                    if row[k+1] in trigram_table[row[k]]:
                        if row[k+2] in trigram_table[row[k]][row[k+1]]:
                            tri_prob = tri_prob+ trigram_table[row[k]][row[k+1]][row[k+2]]
        tri_utility=(tri_prob/(trigram_count*len(test_dataset.transcription)))
    return -(uni_utility+bi_utility+tri_utility)

def compute_length_dict(avg_sent,max_sen_len=11):
    total_size=len(avg_sent)
    count_dict={}
    for k in avg_sent:
        if k not in count_dict:
            count_dict[k]=0
        count_dict[k]=count_dict[k]+1
    for t in range(1,max_sen_len):
        if t not in count_dict:
            count_dict[t]=10**-5
    sizes=[]
    for j in count_dict:
        sizes.append((j,count_dict[j]))
    sizes = sorted(sizes, key=lambda x: x[0])
    sizes = [(x, y / total_size) for (x,y) in sizes]
    return sizes

def get_utterance_utility_KL_length(train_set,test_set,single_intent=False):
    if single_intent:
        train_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in train_set.transcription.values],16)
        test_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in test_set.transcription.values],16)
    else:    
        train_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in train_set.transcription.values])
        test_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in test_set.transcription.values])
    return -KL_divergence(test_dist,train_dist)  

def get_utterance_utility_KL_intent(train_set,test_set,single_intent=False):
    if train_set is None:
        return -10^9
    if test_set is None:
        return -10^9
    train_set_intent_distribution =compute_intent_distribution(train_set,single_intent)
    test_set_intent_distribution =compute_intent_distribution(test_set,single_intent)
    train_set_intent_distribution,test_set_intent_distribution=compute_intent_distribution_combine(train_set_intent_distribution,test_set_intent_distribution)
    kld =0.5*(KL_divergence(train_set_intent_distribution, test_set_intent_distribution)+KL_divergence(test_set_intent_distribution, train_set_intent_distribution))
    return - kld

def get_utterance_utility_bleu_score(train_set,test_set, weights=None,save_file=False, length_normalise=False, length_normlise_ratio=0.1,autoweight=False,single_intent=False):
    train_sent_arr=[]    
    test_sent_dict={}
    for sentence in train_set.transcription.values:
        train_sent_arr.append(sentence.strip().lower().strip(punctuation).split(" "))
    utility=0
    utility_dict={}
    for sentence in test_set.transcription.values:
        sent_arr=sentence.strip().lower().strip(punctuation).split(" ")
        if tuple(sent_arr) not in test_sent_dict:
            test_sent_dict[tuple(sent_arr)]=0
        test_sent_dict[tuple(sent_arr)]=test_sent_dict[tuple(sent_arr)]+1
    total_test_sentences=0
    # print(test_sent_dict)
    for sent_arr_tuple in  test_sent_dict: 
        sent_arr=list(sent_arr_tuple)
        if weights is None:
            value=sentence_bleu(train_sent_arr,sent_arr)
        else:
            if (autoweight and len(sent_arr)<len(weights)):
                continue
            value=sentence_bleu(train_sent_arr,sent_arr,weights)
        # if length_normalise:
        #     # print(sent_arr)
        #     # print(len(sent_arr))
        #     value=value-(length_normlise_ratio*len(sent_arr))
        utility=utility+(test_sent_dict[sent_arr_tuple]*value)
        if save_file:
            utility_dict[sent_arr_tuple]=[value,test_sent_dict[sent_arr_tuple]]
        total_test_sentences=total_test_sentences+test_sent_dict[sent_arr_tuple]
    if save_file:
        return (-(utility/total_test_sentences),utility_dict)
    else:
        if single_intent:
            train_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in train_set.transcription.values],16)
            test_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in test_set.transcription.values],16)
        else:
            train_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in train_set.transcription.values])
            test_dist=compute_length_dict([len(sentence.strip().lower().split()) for sentence in test_set.transcription.values])
        # print(KL_divergence(test_dist,train_dist))
        if length_normalise:
            return -(utility/total_test_sentences)-(length_normlise_ratio*KL_divergence(test_dist,train_dist))
        else:
            return -(utility/total_test_sentences)

def get_speaker_utility_WER(train_set,test_set,transcript_file="gcloud_transcription.csv",subdivide_error= None):
    gold_transcription_dict={}
    test_path=test_set.path.values
    count=0
    for sentence in test_set.transcription.values:
        if test_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[test_path[count]]=sent_arr
        count=count+1

    transcription_set = pd.read_csv(transcript_file)
    transcription_dict={}
    for k in transcription_set.values:
        string_arr=k[1]
        str2_arr=[]
        prev_str=""
        if k[0] not in transcription_dict:
            transcription_dict[k[0]]=string_arr
    average_WER=0
    utility_dict={}
    sum_WER=0
    sum_length=0
    for k in gold_transcription_dict:
        if subdivide_error is not None:
            # print(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))
            error=(float(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))/len(gold_transcription_dict[k]))
        else:
            error = wer(gold_transcription_dict[k], transcription_dict[k])
        average_WER=average_WER+error
        utility_dict[tuple((gold_transcription_dict[k], transcription_dict[k]))]=error
        sum_WER=sum_WER+(error*len(gold_transcription_dict[k]))
        sum_length=sum_length+len(gold_transcription_dict[k])

    return (sum_WER/sum_length)

def get_speaker_utility_replace(train_set,test_set,transcript_file="gcloud_transcription.csv",subdivide_error= None,use_ins_del_diff=False, remove_absolute=False,single_intent=False):
    gold_transcription_dict={}
    test_path=test_set.path.values
    count=0
    for sentence in test_set.transcription.values:
        if test_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[test_path[count]]=sent_arr
        count=count+1

    transcription_set = pd.read_csv(transcript_file)
    transcription_dict={}
    for k in transcription_set.values:
        string_arr=k[1]
        str2_arr=[]
        prev_str=""
        if k[0] not in transcription_dict:
            transcription_dict[k[0]]=string_arr

    utility_dict={}
    sum_replace=0
    sum_insert=0
    sum_delete=0
    sum_length=0
    if single_intent:
        insertion_dict=pickle.load(open("caches/original_splits_insert_complete_snips_speaker_WER.pkl","rb"))
        replace_dict=pickle.load(open("caches/original_splits_replace_complete_snips_speaker_WER.pkl","rb"))
        delete_dict=pickle.load(open("caches/original_splits_delete_complete_snips_speaker_WER.pkl","rb"))
    else:
        insertion_dict=pickle.load(open("caches/original_splits_insert_complete_speaker_WER.pkl","rb"))
        replace_dict=pickle.load(open("caches/original_splits_replace_complete_speaker_WER.pkl","rb"))
        delete_dict=pickle.load(open("caches/original_splits_delete_complete_speaker_WER.pkl","rb"))
    for k in gold_transcription_dict:
        sum_replace=sum_replace+(replace_dict[k]*len(gold_transcription_dict[k]))
        sum_insert=sum_insert+(insertion_dict[k]*len(gold_transcription_dict[k]))
        sum_delete=sum_delete+(delete_dict[k]*len(gold_transcription_dict[k]))
        sum_length=sum_length+len(gold_transcription_dict[k])

    replace_error=(sum_replace/sum_length)
    insert_error=(sum_insert/sum_length)
    delete_error=(sum_delete/sum_length)
    # if (insert_error+delete_error)>0.015:
    #     replace_error=replace_error-(insert_error+delete_error)-1000
    # else:
    if use_ins_del_diff:
        if remove_absolute:
            if single_intent:
                replace_error=replace_error-0.05*np.abs(insert_error-delete_error)-0.05*insert_error-0.05*delete_error
            else:
                replace_error=replace_error-0.05*np.abs(insert_error-delete_error)-0.05*insert_error-0.4*delete_error
        else:
            replace_error=replace_error-0.1*np.abs(insert_error-delete_error)
    return replace_error

def speaker_info(speaker_demographic_file):
    speaker_demographic=pd.read_csv(speaker_demographic_file)
    speaker_demo_dict={}
    map_columns={}
    columns=speaker_demographic.columns
    for k in speaker_demographic.columns[1:]:
        if k not in map_columns:
            map_columns[k]={}
    for row in speaker_demographic.values:
        if row[0] not in speaker_demo_dict:
            speaker_demo_dict[row[0]]={}
        for j in range(1,len(row)):
            if row[j] not in map_columns[columns[j]]:
                map_columns[columns[j]][row[j]]=len(map_columns[columns[j]])
            speaker_demo_dict[row[0]][columns[j]]=map_columns[columns[j]][row[j]]
    for field in map_columns:
        if field=="Self-reported fluency level ":
            store_native_value=map_columns[field]['native']
            for k in speaker_demo_dict:
                if speaker_demo_dict[k][field]==store_native_value:
                    speaker_demo_dict[k][field]=0
                else:
                    speaker_demo_dict[k][field]=1
            map_columns[field]={'native':0,'non_native':1}
        elif field=="First Language spoken":
            store_native_value=map_columns[field]['English (United States)']
            for k in speaker_demo_dict:
                if speaker_demo_dict[k][field]==store_native_value:
                    speaker_demo_dict[k][field]=0
                else:
                    speaker_demo_dict[k][field]=1
            map_columns[field]={'English (United States)':0,'other':1}
        elif field=="Current language used for work/school":
            store_native_value=map_columns[field]['English (United States)']
            for k in speaker_demo_dict:
                if speaker_demo_dict[k][field]==store_native_value:
                    speaker_demo_dict[k][field]=0
                else:
                    speaker_demo_dict[k][field]=1
            map_columns[field]={'English (United States)':0,'other':1}

    return speaker_demo_dict,map_columns

def speaker_SNIPS_info(speaker_demographic_file):
    speaker_demographic=json.load(open(speaker_demographic_file,"r"))
    speaker_demo_dict={}
    map_columns={}
    # print(len(speaker_demographic))
    for k in speaker_demographic:
        if speaker_demographic[k]['worker']["id"] not in  speaker_demo_dict:
            speaker_demo_dict[speaker_demographic[k]['worker']["id"]]={}
        for j in speaker_demographic[k]['worker']:
            if j=="id":
                continue
            if j not in map_columns:
                map_columns[j]={}
            if j=="age":
                if speaker_demographic[k]['worker']['age']<41:
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

    for field in map_columns:
        if field=="country":
            store_native_value=map_columns[field]['United States']
            for k in speaker_demo_dict:
                if speaker_demo_dict[k][field]==store_native_value:
                    speaker_demo_dict[k][field]=0
                else:
                    speaker_demo_dict[k][field]=1
            map_columns[field]={'United States':0,'Other':1}
    return speaker_demo_dict,map_columns


def compute_speaker_distribution(df, speaker_demo_dict,map_columns):
    groups = df.groupby(['speakerId'])
    total_size = 0.0
    count_dict={}
    for k, g in groups:
        total_size=total_size+len(g)
        for j in speaker_demo_dict[k]:
            if j not in count_dict:
                count_dict[j]={}
            if speaker_demo_dict[k][j] not in count_dict[j]:
                count_dict[j][speaker_demo_dict[k][j]] = 0
            count_dict[j][speaker_demo_dict[k][j]] = count_dict[j][speaker_demo_dict[k][j]] + len(g)
    # print(count_dict)
    for j in count_dict:
        for value in map_columns[j]:
            if map_columns[j][value] not in count_dict[j]:
                count_dict[j][map_columns[j][value]]=10**-9
        sizes=[]
        for t in count_dict[j]:
            sizes.append((t,count_dict[j][t]))
        sizes = sorted(sizes, key=lambda x: x[0])
        sizes = [(x, y / total_size) for (x,y) in sizes]
        count_dict[j]=sizes
    return count_dict

class Speaker_Utility():
    """Coordinate Ascent"""

    def __init__(self, speaker_demo_dict,map_columns):
        self.speaker_demo_dict =speaker_demo_dict
        self.map_columns = map_columns

    def get_utility(self,train_dataset,test_dataset,use_WER=False,use_replace=False, use_ins_del_diff=False,remove_absolute=False,single_intent=False):
        if len(test_dataset)==0:
            return float('-inf')
        speaker_train_dict=compute_speaker_distribution(train_dataset,self.speaker_demo_dict,self.map_columns)
        speaker_test_dict=compute_speaker_distribution(test_dataset,self.speaker_demo_dict,self.map_columns)
        # print(speaker_train_dict)
        # print(speaker_test_dict)
        # exit()
        print(use_WER)
        Divergence=0
        for k in speaker_train_dict:
            Divergence=Divergence+ (KL_divergence(speaker_test_dict[k],speaker_train_dict[k])) + (KL_divergence(speaker_train_dict[k],speaker_test_dict[k]))
        if use_WER:
            if use_replace:
                if use_ins_del_diff:
                    if remove_absolute:
                        replace_error=get_speaker_utility_replace(train_dataset,test_dataset,use_ins_del_diff=use_ins_del_diff,remove_absolute=remove_absolute,single_intent=single_intent)
                        return ((-Divergence)/len(speaker_train_dict)+5*replace_error)
                    else:
                        replace_error=get_speaker_utility_replace(train_dataset,test_dataset,use_ins_del_diff=use_ins_del_diff)
                        return ((-Divergence)/len(speaker_train_dict)+0.6*replace_error)
                else:
                    replace_error=get_speaker_utility_replace(train_dataset,test_dataset)
                    return ((-Divergence)/len(speaker_train_dict)+1.0*replace_error)
            else:
                replace_error=get_speaker_utility_WER(train_dataset,test_dataset)
                return ((-Divergence)/len(speaker_train_dict)+0.4*replace_error)
        else:
            return (-Divergence)/len(speaker_train_dict)


def utility(data_dir, resplit_style,utility, utterance_lookup=False,noBLEU=False,seed=None):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    if noBLEU:
       data_str=data_str+"_noBLEU"
    if seed is not None:
        data_str=data_str+"_"+str(seed)
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        test_speaker_set = pd.read_csv(test_speaker_file)
        test_utterance_set = pd.read_csv(test_utterance_file)
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)

    speaker_demographic_file=os.path.join(data_dir, "data/", "speaker_demographics.csv")
    if "snips" in data_dir:
        speaker_demo_dict,map_columns=speaker_SNIPS_info("snips_slu_data_v1.0/smart-lights-en-close-field/speech_corpus/metadata.json")
    else:
        speaker_demo_dict,map_columns=speaker_info(speaker_demographic_file)
    speaker_utility_class=Speaker_Utility(speaker_demo_dict,map_columns)
    if resplit_style=="unseen" or resplit_style=="challenge":
        if utterance_lookup:
            speaker_utility=speaker_utility_class.get_utility(train_set,test_utterance_set)
        else:
            speaker_utility=speaker_utility_class.get_utility(train_set,test_speaker_set)
    else:
        speaker_utility=speaker_utility_class.get_utility(train_set,test_set)
    print(speaker_utility)

def error_analysis(data_dir, resplit_style,utility,error_file=None, weights=False, noBLEU=False, weights_array=None, length_normalise=False,length_normlise_ratio=0.05, utterance_lookup=True,autoweight=False):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    if noBLEU:
        data_str=data_str+"_noBLEU"
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        test_speaker_set = pd.read_csv(test_speaker_file)
        test_utterance_set = pd.read_csv(test_utterance_file)
        if utterance_lookup:
            test_set=test_utterance_set
        else:
            test_set=test_speaker_set
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)

    if weights_array is not None:
        weights_array_value=[float(i) for i in weights_array.split(",")]
    else:
        weights_array_value=None
    
    if error_file is not None:
        train_sent_arr=[]    
        test_sent_dict={}
        for sentence in train_set.transcription.values:
            train_sent_arr.append(sentence.strip().lower().strip(punctuation).split(" "))
        utility=0
        utility_dict={}
        error_set = pd.read_csv(error_file)
        audio_file_set = error_set.audio_path.values
        test_path=test_set.path.values
        count=0
        for sentence in test_set.transcription.values:
            if test_path[count] in audio_file_set:
                sent_arr=sentence.strip().lower().strip(punctuation).split(" ")
                if tuple(sent_arr) not in test_sent_dict:
                    test_sent_dict[tuple(sent_arr)]=0
                test_sent_dict[tuple(sent_arr)]=test_sent_dict[tuple(sent_arr)]+1
            count=count+1
        total_test_sentences=0

        for sent_arr_tuple in  test_sent_dict: 
            sent_arr=list(sent_arr_tuple)
            if weights:
                value=sentence_bleu(train_sent_arr,sent_arr, weights=weights_array_value)
            else:
                value=sentence_bleu(train_sent_arr,sent_arr)
            if length_normalise:
                value=value-(length_normlise_ratio*len(sent_arr))
            utility=utility+(test_sent_dict[sent_arr_tuple]*value)
            total_test_sentences=total_test_sentences+test_sent_dict[sent_arr_tuple]
            utility_dict[sent_arr_tuple]=[value,test_sent_dict[sent_arr_tuple]]
            
        utility=(utility/total_test_sentences)
    else:
        if weights:
            utility, utility_dict=get_utterance_utility_bleu_score(train_set, test_set, weights=weights_array_value,save_file=True, length_normalise=length_normalise,autoweight=autoweight)
        else:
            utility, utility_dict=get_utterance_utility_bleu_score(train_set, test_set,save_file=True, length_normalise=length_normalise,autoweight=autoweight)
    if error_file is not None:
        if weights:
           f = open(data_str.split("/")[-1]+"_utterance_error_bleu_weights_"+weights_array+".pkl","wb")
        else: 
            f = open(data_str.split("/")[-1]+"_utterance_error_bleu.pkl","wb")
    else:
        if weights:
           f = open(data_str.split("/")[-1]+"_utterance_bleu_weights_"+weights_array+".pkl","wb")
        else:
            f = open(data_str.split("/")[-1]+"_utterance_bleu.pkl","wb")
    pickle.dump(utility_dict,f)
    f.close()
    print(utility)
    return utility


def compute_entropy(data_dir, resplit_style,utility,error_file=None):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        test_speaker_set = pd.read_csv(test_speaker_file)
        test_utterance_set = pd.read_csv(test_utterance_file)
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)
    unigram_table, unigram_count=get_uniCounts(train_set)
    uni_entropy=0
    for k in unigram_table:
        prob=unigram_table[k]/unigram_count
        uni_entropy-=prob*np.log2(prob)
    print(uni_entropy)
    bigram_table, num_bigrams, bigram_count=get_biCounts(train_set)
    bi_entropy=0
    for k in bigram_table:
        for j in bigram_table[k]:
            prob=bigram_table[k][j]/bigram_count
            bi_entropy-=prob*np.log2(prob)
    print(bi_entropy)
    trigram_table, num_trigrams, trigram_count=get_triCounts(train_set)
    tri_entropy=0
    for k in trigram_table:
        for j in trigram_table[k]:
            for i in trigram_table[k][j]:
                prob=trigram_table[k][j][i]/trigram_count
                tri_entropy-=prob*np.log2(prob)
    print(tri_entropy)
    print((uni_entropy+bi_entropy+tri_entropy)/3)
    if resplit_style=="unseen" or resplit_style=="challenge":
        unigram_table, unigram_count=get_uniCounts(test_utterance_set)
        bigram_table, num_bigrams, bigram_count=get_biCounts(test_utterance_set)
        trigram_table, num_trigrams, trigram_count=get_triCounts(test_utterance_set)
    else:
        unigram_table, unigram_count=get_uniCounts(test_set)
        bigram_table, num_bigrams, bigram_count=get_biCounts(test_set)
        trigram_table, num_trigrams, trigram_count=get_triCounts(test_set)
    uni_entropy=0
    for k in unigram_table:
        prob=unigram_table[k]/unigram_count
        uni_entropy-=prob*np.log2(prob)
    print(uni_entropy)
    bi_entropy=0
    for k in bigram_table:
        for j in bigram_table[k]:
            prob=bigram_table[k][j]/bigram_count
            bi_entropy-=prob*np.log2(prob)
    print(bi_entropy)
    tri_entropy=0
    for k in trigram_table:
        for j in trigram_table[k]:
            for i in trigram_table[k][j]:
                prob=trigram_table[k][j][i]/trigram_count
                tri_entropy-=prob*np.log2(prob)
    print(tri_entropy)
    print((uni_entropy+bi_entropy+tri_entropy)/3)
    return

def compute_WER(data_dir, resplit_style,utility, noBLEU, transcript_file,utterance_lookup=False,error_file=None,subdivide_error= None, complete=False):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    if noBLEU:
        data_str=data_str+"_noBLEU"
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        if utterance_lookup:
            test_set = pd.read_csv(test_utterance_file)
        else:
            test_set = pd.read_csv(test_speaker_file)        
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)
    gold_transcription_dict={}
    test_path=test_set.path.values
    count=0
    if error_file is not None:
        error_set = pd.read_csv(error_file)
        audio_file_set = error_set.audio_path.values
    for sentence in test_set.transcription.values:
        if error_file is not None:
            if test_path[count] not in audio_file_set:
                count=count+1
                continue
        if test_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[test_path[count]]=sent_arr
        count=count+1

    if complete:
        train_path=train_set.path.values
        count=0
        for sentence in train_set.transcription.values:
            if error_file is not None:
                if train_path[count] not in audio_file_set:
                    count=count+1
                    continue
            if train_path[count] not in gold_transcription_dict:
                sent_arr=sentence.strip().lower()
                gold_transcription_dict[train_path[count]]=sent_arr
            count=count+1
        valid_path=valid_set.path.values
        count=0
        for sentence in valid_set.transcription.values:
            if error_file is not None:
                if valid_path[count] not in audio_file_set:
                    count=count+1
                    continue
            if valid_path[count] not in gold_transcription_dict:
                sent_arr=sentence.strip().lower()
                gold_transcription_dict[valid_path[count]]=sent_arr
            count=count+1

    transcription_set = pd.read_csv(transcript_file)
    transcription_dict={}
    for k in transcription_set.values:
        string_arr=k[1]
        str2_arr=[]
        prev_str=""
        print(string_arr)
        if k[0] not in transcription_dict:
            transcription_dict[k[0]]=string_arr
    average_WER=0
    utility_dict={}
    sum_WER=0
    sum_length=0
    for k in gold_transcription_dict:
        if subdivide_error is not None:
            # print(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))
            error=(float(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))/len(gold_transcription_dict[k]))
        else:
            error = wer(gold_transcription_dict[k], transcription_dict[k])
        average_WER=average_WER+error
        utility_dict[k]=error
        sum_WER=sum_WER+(error*len(gold_transcription_dict[k]))
        sum_length=sum_length+len(gold_transcription_dict[k])
        # print(utility_dict)
        # exit()
    if subdivide_error is not None:
        if subdivide_error==1:
            data_str=data_str+"_insert"
        elif subdivide_error==2:
            data_str=data_str+"_replace"
        elif subdivide_error==3:
            data_str=data_str+"_delete"
    if complete:
        data_str=data_str+"_complete"
    if "snips" in data_dir:
        data_str=data_str+"_snips"
    if error_file is not None:
        if utterance_lookup:
            f = open(data_str.split("/")[-1]+"_utterance_split_speaker_error_WER.pkl","wb")
        else:
            f = open(data_str.split("/")[-1]+"_speaker_error_WER.pkl","wb")
    else:
        if utterance_lookup:
            f = open(data_str.split("/")[-1]+"_utterance_split_speaker_WER.pkl","wb")
        else:
            f = open(data_str.split("/")[-1]+"_speaker_WER.pkl","wb")
    pickle.dump(utility_dict,f)
    f.close()
    print(average_WER/len(gold_transcription_dict))
    print(sum_WER/sum_length)
    # print(get_speaker_utility_WER(train_set,test_set,transcript_file))
    return

def compute_avg_WER_transcriptions(data_dir, resplit_style,utility,transcript_file,error_file=None):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        test_set = pd.read_csv(test_speaker_file)
        test_utterance_set = pd.read_csv(test_utterance_file)
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)
    gold_transcription_dict={}
    test_path=test_set.path.values
    count=0
    if error_file is not None:
        error_set = pd.read_csv(error_file)
        audio_file_set = error_set.audio_path.values
    for sentence in test_set.transcription.values:
        if error_file is not None:
            if test_path[count] not in audio_file_set:
                count=count+1
                continue
        if test_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[test_path[count]]=sent_arr
        count=count+1

    count=0
    train_path=train_set.path.values
    for sentence in train_set.transcription.values:
        if train_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[train_path[count]]=sent_arr
        count=count+1

    count=0
    valid_path=valid_set.path.values
    for sentence in valid_set.transcription.values:
        if valid_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[valid_path[count]]=sent_arr
        count=count+1


    transcription_set = pd.read_csv(transcript_file)
    transcription_dict={}
    for k in transcription_set.values:
        string_arr=k[1]
        str2_arr=[]
        prev_str=""
        print(string_arr)
        if k[0] not in transcription_dict:
            transcription_dict[k[0]]=string_arr
    average_WER=0
    utility_dict={}
    
    for k in gold_transcription_dict:
        error = wer(gold_transcription_dict[k], transcription_dict[k])
        average_WER=average_WER+error
        if gold_transcription_dict[k] not in utility_dict:
            utility_dict[gold_transcription_dict[k]]=[]
        utility_dict[gold_transcription_dict[k]].append(error)
    for transcript in utility_dict:
        utility_dict[transcript]=np.mean(utility_dict[transcript])
    if error_file is not None:
         f = open(data_str.split("/")[-1]+"_speaker_transcription_error_WER.pkl","wb")
    else:
        f = open(data_str.split("/")[-1]+"_speaker_transcription_WER.pkl","wb")
    pickle.dump(utility_dict,f)
    f.close()
    print(average_WER/len(gold_transcription_dict))
    return

def compute_intent_KL(data_dir, resplit_style,utility,single_intent=False,noBLEU=False,utterance_lookup=False,seed=None):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    if noBLEU:
        data_str=data_str+"_noBLEU"
    if seed is not None:
        data_str=data_str+"_"+str(seed)
    print(data_str)
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        if utterance_lookup:
            test_set = pd.read_csv(test_utterance_file)
        else:
            test_set = pd.read_csv(test_speaker_file)        
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)   
    valid_set = pd.read_csv(valid_file)
    train_set_intent_distribution =compute_intent_distribution(train_set,single_intent)
    test_set_intent_distribution =compute_intent_distribution(test_set,single_intent)
    print(train_set_intent_distribution)
    print(test_set_intent_distribution)
    kld =0.5*(KL_divergence(train_set_intent_distribution, test_set_intent_distribution)+KL_divergence(test_set_intent_distribution, train_set_intent_distribution))
    print(kld)

def utility_word_length_ratio(train_set,test_set):
    avg_sent=0
    for sentence in train_set.transcription.values:
        avg_sent=avg_sent+len(sentence.strip().lower().split())
    train_len=(avg_sent/len(train_set.transcription.values))
    avg_sent=0
    for sentence in test_set.transcription.values:
        avg_sent=avg_sent+len(sentence.strip().lower().split())
    test_len=(avg_sent/len(test_set.transcription.values))
    return (train_len/test_len)

def compute_word_length_ratio(data_dir, resplit_style,utility,error_file=None, perfect=False,noBLEU=False,utterance_lookup=False,seed=None):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    if perfect:
        data_str=data_str+"_perfect"
    if noBLEU:
        data_str=data_str+"_noBLEU"
    if seed is not None:
        data_str=data_str+"_"+str(seed)
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        if utterance_lookup:
            test_set = pd.read_csv(test_utterance_file)
        else:
            test_set = pd.read_csv(test_speaker_file)
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    if utility:
        resplit_style=resplit_style+"_utility"
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)
    avg_train_sent=[]
    for sentence in train_set.transcription.values:
        avg_train_sent.append(len(sentence.strip().lower().split()))
    print(np.mean(avg_train_sent))
    n, bins, patches = plt.hist(x=avg_train_sent, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Trancription Length')
    plt.ylabel('Percentage of Transcripts')
    plt.ylim((0,1.0))
    plt.savefig("train_"+resplit_style+".png")
    plt.show()
    avg_sent=[]
    for sentence in test_set.transcription.values:
        avg_sent.append(len(sentence.strip().lower().split()))
    print(np.mean(avg_sent))
    n, bins, patches = plt.hist(x=avg_sent, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Trancription Length')
    plt.ylabel('Percentage of Transcripts')
    plt.ylim((0,1.0))
    plt.savefig("test_"+resplit_style+".png")
    plt.show()
    if "snips" in data_dir:
        train_dist=compute_length_dict(avg_train_sent,16)
        test_dist=compute_length_dict(avg_sent,16)
    else:
        train_dist=compute_length_dict(avg_train_sent)
        test_dist=compute_length_dict(avg_sent)
    print(KL_divergence(test_dist,train_dist))
    if error_file is not None:
        error_set = pd.read_csv(error_file)
        audio_file_set = error_set.audio_path.values
        avg_sent=[]
        count=0
        test_path=test_set.path.values
        for sentence in test_set.transcription.values:
            if test_path[count] in audio_file_set:
                avg_sent.append(len(sentence.strip().lower().split()))
            count=count+1
        print(np.median(avg_sent))
        n, bins, patches = plt.hist(x=avg_sent, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        for item in patches:
            item.set_height(item.get_height()/sum(n))
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Trancription Length')
        plt.ylabel('Percentage of Transcripts')
        plt.ylim((0,1.0))
        plt.savefig("test_error.png")
        plt.show()

def compute_WER_KL(data_dir, resplit_style,utility, noBLEU, transcript_file,utterance_lookup=False,error_file=None,subdivide_error= None, complete=False):
    data_str="data/"
    data_str=data_str+f"{resplit_style}_splits"
    if utility:
        data_str=data_str+"_utility"
    if noBLEU:
        data_str=data_str+"_noBLEU"
    train_file = os.path.join(data_dir, data_str, "train_data.csv")
    if resplit_style=="unseen" or resplit_style=="challenge":
        test_speaker_file = os.path.join(data_dir, data_str, "speaker_test_data.csv")
        test_utterance_file = os.path.join(data_dir, data_str, "utterance_test_data.csv")
        if utterance_lookup:
            test_set = pd.read_csv(test_utterance_file)
        else:
            test_set = pd.read_csv(test_speaker_file)        
    else:
        test_file = os.path.join(data_dir, data_str, "test_data.csv")
        test_set = pd.read_csv(test_file)
    valid_file = os.path.join(data_dir, data_str, "valid_data.csv")
    train_set = pd.read_csv(train_file)    
    valid_set = pd.read_csv(valid_file)
    gold_transcription_dict={}
    test_path=test_set.path.values
    count=0
    if error_file is not None:
        error_set = pd.read_csv(error_file)
        audio_file_set = error_set.audio_path.values
    for sentence in test_set.transcription.values:
        if error_file is not None:
            if test_path[count] not in audio_file_set:
                count=count+1
                continue
        if test_path[count] not in gold_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_transcription_dict[test_path[count]]=sent_arr
        count=count+1

    gold_train_transcription_dict={}
    train_path=train_set.path.values
    count=0
    for sentence in train_set.transcription.values:
        if train_path[count] not in gold_train_transcription_dict:
            sent_arr=sentence.strip().lower()
            gold_train_transcription_dict[train_path[count]]=sent_arr
        count=count+1

    transcription_set = pd.read_csv(transcript_file)
    transcription_dict={}
    for k in transcription_set.values:
        string_arr=k[1]
        str2_arr=[]
        prev_str=""
        print(string_arr)
        if k[0] not in transcription_dict:
            transcription_dict[k[0]]=string_arr
    average_WER=0
    error_dist_dict={}
    error_dist_dict[0]=0
    error_dist_dict[0.1]=0
    error_dist_dict[0.2]=0
    error_dist_dict[0.3]=0
    error_dist_dict[0.4]=0
    error_dist_dict[0.5]=0
    error_dist_dict[0.6]=0
    error_dist_dict[0.7]=0
    error_dist_dict[0.8]=0
    error_dist_dict[0.9]=0
    error_dist_dict[1.0]=0
    sum_WER=0
    sum_length=0
    for k in gold_transcription_dict:
        if subdivide_error is not None:
            # print(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))
            error=(float(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))/len(gold_transcription_dict[k]))
        else:
            error = wer(gold_transcription_dict[k], transcription_dict[k])
        if error>=1.0:
            error_dist_dict[1.0]+=1
        elif error>=0.9:
            error_dist_dict[0.9]+=1
        elif error>=0.8:
            error_dist_dict[0.8]+=1
        elif error>=0.7:
            error_dist_dict[0.7]+=1
        elif error>=0.6:
            error_dist_dict[0.6]+=1
        elif error>=0.5:
            error_dist_dict[0.5]+=1
        elif error>=0.4:
            error_dist_dict[0.4]+=1
        elif error>=0.3:
            error_dist_dict[0.3]+=1
        elif error>=0.2:
            error_dist_dict[0.2]+=1
        elif error>=0.1:
            error_dist_dict[0.1]+=1
        else:
            error_dist_dict[0]+=1
    train_error_dist_dict={}
    train_error_dist_dict[0]=0
    train_error_dist_dict[0.1]=0
    train_error_dist_dict[0.2]=0
    train_error_dist_dict[0.3]=0
    train_error_dist_dict[0.4]=0
    train_error_dist_dict[0.5]=0
    train_error_dist_dict[0.6]=0
    train_error_dist_dict[0.7]=0
    train_error_dist_dict[0.8]=0
    train_error_dist_dict[0.9]=0
    train_error_dist_dict[1.0]=0
    for k in gold_train_transcription_dict:
        if subdivide_error is not None:
            # print(len(Utils.editops(gold_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))
            error=(float(len(Utils.editops(gold_train_transcription_dict[k], transcription_dict[k],look_error=subdivide_error)))/len(gold_train_transcription_dict[k]))
        else:
            error = wer(gold_train_transcription_dict[k], transcription_dict[k])
        if error>=1.0:
            train_error_dist_dict[1.0]+=1
        elif error>=0.9:
            train_error_dist_dict[0.9]+=1
        elif error>=0.8:
            train_error_dist_dict[0.8]+=1
        elif error>=0.7:
            train_error_dist_dict[0.7]+=1
        elif error>=0.6:
            train_error_dist_dict[0.6]+=1
        elif error>=0.5:
            train_error_dist_dict[0.5]+=1
        elif error>=0.4:
            train_error_dist_dict[0.4]+=1
        elif error>=0.3:
            train_error_dist_dict[0.3]+=1
        elif error>=0.2:
            train_error_dist_dict[0.2]+=1
        elif error>=0.1:
            train_error_dist_dict[0.1]+=1
        else:
            train_error_dist_dict[0]+=1

    WER1_arr=[]
    WER2_arr=[]
    for k in error_dist_dict:
        WER1_arr.append((k,max(10**-5,error_dist_dict[k]/len(gold_transcription_dict))))
        WER2_arr.append((k,max(10**-5,train_error_dist_dict[k]/len(gold_train_transcription_dict))))
    print(0.5*(KL_divergence(WER1_arr, WER2_arr)+KL_divergence( WER2_arr, WER1_arr)))

    
    # print(get_speaker_utility_WER(train_set,test_set,transcript_file))
    return