# Code adapted from https://github.com/rueycheng/CoordinateAscent
import pandas as pd
import numpy as np
from utility import utility_word_length_ratio, get_utterance_utility_KL_length, get_utterance_utility_KL_intent, get_utterance_utility_bleu_score, get_speaker_utility_WER

np.random.seed(0)

class CoordinateAscent():
    """Coordinate Ascent"""

    def __init__(self, n_restarts=5, max_iter=25, tol=0.0001, utility_scorer=None,lamda=1,ratio=0.15,weights=None,subdivide_error= None, mantain_length=False, use_WER=False,use_replace=False, use_ins_del_diff=False,remove_absolute=False,single_intent=False,add_utterance=False,intent_weight=0,baseline_bleu=0.627,baseline_WER = 0.187):
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.scorer = self.compute_score(utility_scorer,lamda=lamda,ratio=ratio, mantain_length=mantain_length)
        self.weights=weights
        self.subdivide_error=subdivide_error
        self.use_WER=use_WER
        self.use_replace=use_replace
        self.use_ins_del_diff=use_ins_del_diff
        self.remove_absolute=remove_absolute
        self.single_intent=single_intent
        self.add_utterance=add_utterance
        self.intent_weight=intent_weight
        self.baseline_BLEU=baseline_bleu
        self.baseline_WER = baseline_WER

    def compute_score(self, utility_scorer,lamda=1,ratio=0.15, mantain_length=False):
        def utility_func(coef,utterance_groups):
            train_utterance_group=[utterance_groups[i] for i in range(len(utterance_groups)) if coef[i]==False] 
            test_utterance_group=[utterance_groups[i] for i in range(len(utterance_groups)) if coef[i]==True] 
            if train_utterance_group==[]:
                return float('-inf')
            if test_utterance_group==[]:
                return float('-inf')
            train_groups =pd.concat(train_utterance_group)
            test_groups = pd.concat(test_utterance_group)
            if self.use_WER:
                if self.single_intent:
                    score =utility_scorer(train_groups,test_groups,use_WER=self.use_WER,use_replace=self.use_replace,use_ins_del_diff=self.use_ins_del_diff,remove_absolute=self.remove_absolute,single_intent=self.single_intent) - lamda*((len(test_groups.transcription)/len(train_groups.transcription))-ratio)**2
                else:
                    score = utility_scorer(train_groups,test_groups,use_WER=self.use_WER,use_replace=self.use_replace,use_ins_del_diff=self.use_ins_del_diff,remove_absolute=self.remove_absolute,single_intent=self.single_intent) - lamda*((len(test_groups.transcription)/len(train_groups.transcription))-ratio)**2
            elif (self.weights is not None and not(self.add_utterance)):
                print(self.weights)
                score = utility_scorer(train_groups,test_groups, weights=self.weights, length_normalise=mantain_length,single_intent=self.single_intent)
                score = score - lamda*((len(test_groups.transcription)/len(train_groups.transcription))-ratio)**2
            elif self.subdivide_error is not None:
                score = utility_scorer(train_groups,test_groups,subdivide_error=self.subdivide_error) - lamda*((len(test_groups.transcription)/len(train_groups.transcription))-ratio)**2
            elif self.single_intent:
                score = utility_scorer(train_groups,test_groups,single_intent=self.single_intent) - lamda*((len(test_groups.transcription)/len(train_groups.transcription))-ratio)**2
            else:
                score = utility_scorer(train_groups,test_groups) - lamda*((len(test_groups.transcription)/len(train_groups.transcription))-ratio)**2
            if self.add_utterance:
                if self.weights is not None:
                    print(self.intent_weight)
                    val=get_speaker_utility_WER(train_groups,test_groups,transcript_file="gcloud_snips_transcription.csv")
                    print(val)
                    score = 0.5*score -  3.0*abs(self.baseline_WER - val)+5.0*get_utterance_utility_bleu_score(train_groups,test_groups, weights=self.weights, length_normalise=mantain_length,single_intent=self.single_intent)+ self.intent_weight*get_utterance_utility_KL_intent(train_groups,test_groups,self.single_intent)
                else:
                    if self.use_WER:
                        score = score - 0.1*abs(self.baseline_BLEU + get_utterance_utility_bleu_score(train_groups,test_groups, weights=(0.25,0.25,0.25,0.25), length_normalise=mantain_length,single_intent=self.single_intent)) + get_utterance_utility_KL_length(train_groups,test_groups,self.single_intent) +  self.intent_weight*get_utterance_utility_KL_intent(train_groups,test_groups,self.single_intent)
                    else:
                        score = score + get_utterance_utility_KL_length(train_groups,test_groups,self.single_intent) +  self.intent_weight*get_utterance_utility_KL_intent(train_groups,test_groups,self.single_intent)
            return score
        return utility_func

    def fit(self, utterance_groups):
        """Fit a model to the data"""
        best_score, best_coef = float('-inf'), [True]+[False for i in range(len(utterance_groups)-1)]
        for restart_no in range(1, self.n_restarts + 1):
            coef = [True]+[False for i in range(len(utterance_groups)-1)]
            # train_utterance_group=[utterance_groups[i] for i in range(len(utterance_groups)) if coef[i]==False] 
            # test_utterance_group=[utterance_groups[i] for i in range(len(utterance_groups)) if coef[i]==True] 
            # print(train_utterance_group)
            # print(test_utterance_group)
            score = self.scorer(coef,utterance_groups)
            # print(score)
            n_fails = 0  # count the number of *consecutive* failures
            while n_fails < len(utterance_groups):
                for iter_no, fid in enumerate(np.random.permutation(len(utterance_groups))):
                    best_local_score, best_change = score, None
                    # pred_delta = X[:, fid]
                    # stepsize = 0.05 * np.abs(coef[fid]) if coef[fid] != 0 else 0.001

                    change = coef.copy()
                    change[fid]=not(change[fid])

                    new_score = self.scorer(change, utterance_groups)
                    # print(new_score)
                    if new_score > best_local_score:
                        best_local_score, best_change = new_score, change
                    
                    if best_change is not None:
                        score = best_local_score
                        coef[fid] = best_change [fid]
                        print('{}\t{}\t{}\t{}'.format(restart_no, iter_no, fid, score))
                        n_fails = 0
                    else:
                        n_fails += 1

            if score > best_score + self.tol:
                best_score, best_coef = score, coef.copy()

        return [utterance_groups[i] for i in range(len(utterance_groups)) if best_coef[i]==True],[utterance_groups[i] for i in range(len(utterance_groups)) if best_coef[i]==False]



