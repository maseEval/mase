from collections import defaultdict
import os
import pandas as pd
from coordinate_ascent import CoordinateAscent
from utility import get_utterance_utility_bleu_score, get_speaker_utility_replace, get_speaker_utility_WER, Speaker_Utility, speaker_info, get_utterance_utility_KL_length, speaker_SNIPS_info

from probability_utils import KL_divergence, compute_intent_distribution


'''
Run:
python processing_scripts/resplit_data.py \
--data_dir /home/ec2-user/slu_splits/fluent_speech_commands_dataset  \
--dataset {fluent_speech_commands, snips} \
--resplit_style {random, decomposable} {--challenge || --unseen}

Will create a directory called "unseen_splits", "challenge_splits", or "random_splits" in the supplied data directory.
Created "random", "unseen_splits", and "challenge_splits" splits have been included in splits_folder for reference
(for both Fluent Speech Commands and Snips).
'''

import argparse

def load_data(data_dir):
    train_file = os.path.join(data_dir, "data", "train_data.csv")
    test_file = os.path.join(data_dir, "data", "test_data.csv")
    valid_file = os.path.join(data_dir, "data", "valid_data.csv")
    if not (os.path.isfile(train_file) and os.path.isfile(test_file) and os.path.isfile(valid_file)):
        breakpoint()
        raise ValueError("Could not find train_data.csv, test_data.csv, or valid_data.csv in supplied data directory.")

    train_set = pd.read_csv(train_file)
    test_set = pd.read_csv(test_file)
    valid_set = pd.read_csv(valid_file)
    splits = [len(train_set), len(test_set), len(valid_set)]
    concatenated = pd.concat([train_set, test_set, valid_set])
    return [train_set, test_set, valid_set], concatenated, splits

def concatenate_if_exists(df_a, concat_dfs):
    if df_a is None:
        return pd.concat(concat_dfs)
    else:
        return pd.concat([df_a] + concat_dfs)

# Construct the new valid and test sets by choosing the N largest utterance groups for each intent.
def select_utterance_groups_to_add_to_test(utterance_groups, ratio_of_groups_to_add=8, utility = False,lamda=1,ratio=0.15, mantain_length=False,BLEU=True,single_intent=False):
    # Choose a number of utterance groups to add to the test set, proportional to the number of utterance
    # groups existing for that intent.
    # This is a heuristic, but it leads to a test set whose intent distribution closely matches the
    # original test set intent distribution. We also find that adding the smallest utterance groups helps with this.
    # TODO(Siddhanth): add a more sophisticated way to choose groups by a semantic complexity measure.
    # ratio_of_groups_to_add is the hyperaparameter trying to preserve the distribution of intent labels.
    # print(BLEU)
    if utility:
        if BLEU:
            if single_intent:
                n_restarts=1
            else:
                n_restarts=5
            utility_selector=CoordinateAscent(utility_scorer=get_utterance_utility_bleu_score,lamda=lamda,ratio=ratio, weights=(0.5,0.5,0,0), mantain_length=mantain_length,single_intent=single_intent,n_restarts=n_restarts)
        else:
            utility_selector=CoordinateAscent(utility_scorer=get_utterance_utility_KL_length,lamda=lamda,ratio=ratio,single_intent=single_intent)
        groups_to_add_to_test, groups_to_add_to_train_val = utility_selector.fit(utterance_groups)
    else:
        number_of_groups_to_include = max(1, int(len(utterance_groups) / ratio_of_groups_to_add))
        groups_to_add_to_test = utterance_groups[:number_of_groups_to_include]
        groups_to_add_to_train_val = utterance_groups[number_of_groups_to_include:]
    return groups_to_add_to_test, groups_to_add_to_train_val

def resplit_on_utterances(concatenated, splits, seed=5, ratio_parameter=8, mantain_length=False,utility=False,lamda=0.4,BLEU=True,ratio=0.12,single_intent=False):
    '''
    Re-split the concatenated data, to ensure that each unique utterance can only be found in one split
    (train, test, or valid), while maintaining equal distribution of intents and roughly maintaining splits.
    '''
    test_set = None
    valid_train_set = None
    intent_utterance_groups = defaultdict(list)
    for (utterance, group) in concatenated.groupby("transcription"):
        if single_intent:
            for k in ['intentLbl']:
                if len(set(group[k])) != 1:
                    # print(utterance)
                    # print(group)
                    raise ValueError("Utterance groups should be consistent on intent parse")
            intent_tuple = (group["intentLbl"].iloc[0])
        else:
            for k in ["action", "object", "location"]:
                if len(set(group[k])) != 1:
                    raise ValueError("Utterance groups should be consistent on intent parse")
            intent_tuple = (group["action"].iloc[0], group["object"].iloc[0], group["location"].iloc[0])
        intent_utterance_groups[intent_tuple].append(group)

    # sort utterance groups in each intent by length, from smallest to largest
    for intent in intent_utterance_groups:
        intent_utterance_groups[intent] = sorted(intent_utterance_groups[intent], key=len)

    for intent, utterance_groups in intent_utterance_groups.items():
        groups_for_test, groups_for_train_val = select_utterance_groups_to_add_to_test(utterance_groups, ratio_of_groups_to_add=ratio_parameter,utility=utility,lamda=lamda,ratio=ratio, mantain_length=mantain_length,BLEU=BLEU,single_intent=single_intent)
        test_set = concatenate_if_exists(test_set, groups_for_test)
        valid_train_set = concatenate_if_exists(valid_train_set, groups_for_train_val)

    valid_train_set = valid_train_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    [og_train_size, og_test_size, og_valid_size] = splits
    new_train_size = int(float(og_train_size) / (og_train_size + og_valid_size) * len(valid_train_set))
    train_set = valid_train_set.iloc[:new_train_size]
    utility=get_utterance_utility_bleu_score(train_set,test_set, weights=(0.5,0.5,0,0))
    print("Test BLEU Score")
    print(utility)
    valid_set = valid_train_set.iloc[new_train_size:]
    return train_set, test_set, valid_set

def resplit_decomposable(concatenated,
                       splits,
                       original_intent_distribution,
                       seed=5,
                       foreign_speakers = [],
                       length_tolerance=100,
                       utility=False,
                       speaker_demographic_file= None,
                       use_speaker_utility=False,
                       use_WER=False,
                       use_replace=False,
                       use_ins_del_diff=False,
                       remove_absolute=False,
                       BLEU=True,
                       mantain_length=True,
                       single_intent=False,
                       add_utterance=False):
    '''
    Re-split the concatenated data, to ensure that each unique utterance can only be found in one split
    (train, test, or valid), while maintaining equal distribution of intents and roughly maintaining splits.
    '''
    original_intent_distribution = dict(original_intent_distribution)
    train_set = None
    valid_test_sets = None
    intent_utterance_groups = defaultdict(list)

    original_test_size = splits[1]
    unseen_speaker_test_set = None
    unseen_speaker_train_val_set = None

    # Choose speaker groups uniformly
    concatenated_new=concatenated.copy(deep=True)
    load=False
    if utility:
        if load:
            speaker_demo_dict,map_columns=speaker_SNIPS_info(speaker_demographic_file)
            Speaker_Utility_class=Speaker_Utility(speaker_demo_dict,map_columns)
            concatenated_csv = pd.concat([pd.read_csv("snips_slu_data_v1.0/close_field_splits/decomposable_splits_utility/train_data.csv"), pd.read_csv("snips_slu_data_v1.0/close_field_splits/decomposable_splits_utility/valid_data.csv")])
            groups_to_add_to_train_val=[]
            for (speaker, group) in concatenated_csv.groupby("speakerId"):
                groups_to_add_to_train_val.append(group)
            test_csv=pd.read_csv("snips_slu_data_v1.0/close_field_splits/decomposable_splits_utility/test_data.csv")
            groups_to_add_to_test=[]
            for (speaker, group) in test_csv.groupby("speakerId"):
                groups_to_add_to_test.append(group)
            unseen_speaker_test_set = concatenate_if_exists(unseen_speaker_test_set, groups_to_add_to_test)
            unseen_speaker_train_val_set = concatenate_if_exists(unseen_speaker_train_val_set, groups_to_add_to_train_val)
        else:
            if use_speaker_utility:
                if single_intent:
                    speaker_demo_dict,map_columns=speaker_SNIPS_info(speaker_demographic_file)
                else:
                    speaker_demo_dict,map_columns=speaker_info(speaker_demographic_file)
                Speaker_Utility_class=Speaker_Utility(speaker_demo_dict,map_columns)
                if single_intent:
                    if use_WER:
                        utility_selector=CoordinateAscent(utility_scorer=Speaker_Utility_class.get_utility,use_WER=use_WER,use_replace=use_replace, use_ins_del_diff=use_ins_del_diff,remove_absolute=remove_absolute,lamda=100,n_restarts=1,ratio=0.08, single_intent= single_intent,add_utterance=add_utterance,intent_weight=1)
                    else:
                        utility_selector=CoordinateAscent(utility_scorer=Speaker_Utility_class.get_utility,use_WER=use_WER,use_replace=use_replace, use_ins_del_diff=use_ins_del_diff,remove_absolute=remove_absolute,lamda=100,n_restarts=5,ratio=0.11, single_intent= single_intent,add_utterance=add_utterance)
                else:
                    utility_selector=CoordinateAscent(utility_scorer=Speaker_Utility_class.get_utility,use_WER=use_WER,use_replace=use_replace, use_ins_del_diff=use_ins_del_diff,remove_absolute=remove_absolute,lamda=100,n_restarts=5)
            else:
                utility_selector=CoordinateAscent(utility_scorer=get_speaker_utility_replace,lamda=3,n_restarts=5)
            group_array=[]
            for (speaker, group) in concatenated.groupby("speakerId"):
                group_array.append(group)
            groups_to_add_to_test, groups_to_add_to_train_val = utility_selector.fit(group_array)
            print("yes")
            unseen_speaker_test_set = concatenate_if_exists(unseen_speaker_test_set, groups_to_add_to_test)
            unseen_speaker_train_val_set = concatenate_if_exists(unseen_speaker_train_val_set, groups_to_add_to_train_val)
    else:
        for (speaker, group) in sorted(concatenated.groupby("speakerId"), key=lambda x: len(x[1])):
            disjoint_utterance=False
            concatenated_new= concatenated_new[concatenated_new.speakerId != speaker]
            for k in group.transcription:
                if (k not in concatenated_new.transcription.values):
                    if unseen_speaker_train_val_set is None:
                        disjoint_utterance=True
                    elif (k not in unseen_speaker_train_val_set.transcription):
                        disjoint_utterance=True
            if ((unseen_speaker_test_set is None or len(unseen_speaker_test_set) < original_test_size - length_tolerance) and (not(disjoint_utterance))):
                unseen_speaker_test_set = concatenate_if_exists(unseen_speaker_test_set, [group])
            else:
                unseen_speaker_train_val_set = concatenate_if_exists(unseen_speaker_train_val_set, [group])
    # print(Speaker_Utility_class.get_utility(unseen_speaker_train_val_set ,unseen_speaker_test_set))
    if BLEU:
        if single_intent:
            final_train, unseen_utterance_test, final_valid = resplit_on_utterances(unseen_speaker_train_val_set, splits, seed=seed, mantain_length=mantain_length, utility=utility,lamda=10.0,BLEU=BLEU,single_intent=single_intent)
        else:
            final_train, unseen_utterance_test, final_valid = resplit_on_utterances(unseen_speaker_train_val_set, splits, seed=seed, mantain_length=mantain_length, utility=utility,lamda=5.0,BLEU=BLEU,single_intent=single_intent)
    else:
        if single_intent:
            valid_train_set = unseen_speaker_train_val_set.sample(frac=1, random_state=seed).reset_index(drop=True)
            [og_train_size, og_test_size, og_valid_size] = splits
            new_train_size = int(float(og_train_size) / (og_train_size + og_valid_size) * len(valid_train_set))
            train_set = unseen_speaker_train_val_set.iloc[:new_train_size]
            valid_set = unseen_speaker_train_val_set.iloc[new_train_size:]
            return train_set, valid_set, None, unseen_speaker_test_set
        else:
            final_train, unseen_utterance_test, final_valid = resplit_on_utterances(unseen_speaker_train_val_set, splits, seed=seed, mantain_length=mantain_length, utility=utility,lamda=50.0,BLEU=BLEU,single_intent=single_intent)
    unseen_speaker_test_set_filtered=unseen_speaker_test_set.copy()
    if not(single_intent):
        for k in unseen_speaker_test_set_filtered.transcription:
            if k not in final_train.transcription.values:
                unseen_speaker_test_set_filtered=unseen_speaker_test_set_filtered[unseen_speaker_test_set_filtered.transcription !=k]
    print("Test WER Score")
    if single_intent:
        utility=get_speaker_utility_WER(final_train,unseen_speaker_test_set_filtered,transcript_file="gcloud_snips_transcription.csv")
    else:
        utility=get_speaker_utility_WER(final_train,unseen_speaker_test_set_filtered)
    # print(utility)
    # utility=get_speaker_utility_WER(final_train,unseen_speaker_test_set_filtered,subdivide_error=2)
    print(utility)
    if use_speaker_utility:
        print("Test Speaker Utility")
        print(Speaker_Utility_class.get_utility(final_train,unseen_speaker_test_set_filtered))
    return final_train, final_valid, unseen_utterance_test, unseen_speaker_test_set_filtered



def resplit_shuffled(concatenated, splits, seed=0):
    '''
    Re-split the concatenated data, to ensure that each unique utterance can only be found in one split
    (train, test, or valid), while maintaining equal distribution of intents and roughly maintaining splits.
    '''
    shuffled = concatenated.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_set = shuffled.iloc[:splits[0]]
    test_set = shuffled.iloc[splits[0]:splits[0]+splits[1]]
    valid_set = shuffled.iloc[splits[0]+splits[1]:]
    return train_set, test_set, valid_set
        
def try_diff_train_valid(splits):
    train_set_arr=[]
    valid_set_arr=[]
    for seed in range(5):
        concatenated_csv = pd.concat([pd.read_csv("../fluent_speech_commands/data/decomposable_splits_utility/train_data.csv"), pd.read_csv("../fluent_speech_commands/data/decomposable_splits_utility/valid_data.csv")])
        valid_train_set = concatenated_csv.sample(frac=1, random_state=seed).reset_index(drop=True)
        [og_train_size, og_test_size, og_valid_size] = splits
        new_train_size = int(float(og_train_size) / (og_train_size + og_valid_size) * len(valid_train_set))
        train_set = valid_train_set.iloc[:new_train_size]
        valid_set = valid_train_set.iloc[new_train_size:]
        train_set_arr.append(train_set)
        valid_set_arr.append(valid_set)
    return train_set_arr, valid_set_arr

def main(data_dir, resplit_style, use_speaker_utility, dataset, utility=True, challenge=True, single_intent=False, mantain_length=False, replace=False,use_ins_del_diff=False,remove_absolute=False,add_utterance=False, try_diff_seed=False, seed=5):
    # Dataset is either fluent_speech_commands or snips_close_field.
    original_splits, concatenated_data, original_split_sizes = load_data(data_dir)
    [original_train, original_test, original_valid] = original_splits
    # print(single_intent)
    original_test_set_intent_distribution = compute_intent_distribution(original_test,single_intent)
    if resplit_style == "decomposable":
        if challenge:
            data_str="challenge_splits"
        else:
            data_str="unseen_splits"
    else:
        data_str=f"{resplit_style}_splits"

    if utility and not challenge:
        print(f"Using utility functions to optimize Unseen splits.")
    elif utility:
        print(f"Using utility functions to optimize Challenge splits.")
    
    if not os.path.isdir(os.path.join(data_dir, dataset)):
        os.makedirs(os.path.join(data_dir, dataset))
    output_dir = os.path.join(data_dir, dataset, data_str)
    if try_diff_seed:
        train_arr, valid_arr=try_diff_train_valid(original_split_sizes)
        for seed in range(5):
            output_dir_1=output_dir+"_"+str(seed)
            if not os.path.exists(output_dir_1):
                print(f"Created directory {output_dir_1}")
                os.makedirs(output_dir_1)
            new_train_path = os.path.join(output_dir_1, "train_data.csv")
            new_valid_path = os.path.join(output_dir_1, "valid_data.csv")
            new_splits = [train_arr[seed], valid_arr[seed]]
            new_paths = [new_train_path,new_valid_path]
            # print(f"New dataset split sizes: {[len(train), len(valid)]}")

            for (data, path) in zip(new_splits, new_paths):
                print(f"Wrote file to {path}.")
                data.to_csv(path, index=True)
        return
    if resplit_style == "decomposable":
        if single_intent:
            speaker_demographic_file=os.path.join("snips_slu_data_v1.0/smart-lights-en-close-field/speech_corpus/metadata.json")
        else:
            speaker_demographic_file=os.path.join(data_dir, "data/", "speaker_demographics.csv")
        train, valid, decomposable_utterance_test, decomposable_speaker_test = resplit_decomposable(concatenated_data, original_split_sizes, original_test_set_intent_distribution, seed=seed, utility=utility, speaker_demographic_file=speaker_demographic_file, use_speaker_utility=use_speaker_utility, use_WER=challenge,use_replace=replace, use_ins_del_diff=use_ins_del_diff,remove_absolute=remove_absolute, BLEU=challenge, mantain_length=mantain_length,single_intent=single_intent,add_utterance=add_utterance)
        if decomposable_utterance_test is None:
            test=decomposable_speaker_test
            new_test_set_intent_distribution = compute_intent_distribution(test,single_intent)
            kld = KL_divergence(original_test_set_intent_distribution, new_test_set_intent_distribution)
            print(f"KL divergence between original test distribution of intents and new test distribution of intents: {kld}\n")
            if not os.path.exists(output_dir):
                print(f"Created directory {output_dir}")
                os.makedirs(output_dir)
            new_train_path = os.path.join(output_dir, "train_data.csv")
            new_valid_path = os.path.join(output_dir, "valid_data.csv")
            new_test_path = os.path.join(output_dir, "test_data.csv")
            new_splits = [train, test, valid]
            new_paths = [new_train_path, new_test_path, new_valid_path]
            print(f"New dataset split sizes: {[len(train), len(test), len(valid)]}")

            for (data, path) in zip(new_splits, new_paths):
                print(f"Wrote file to {path}.")
                data.to_csv(path, index=True)

            print(f"Old dataset split sizes: {original_split_sizes}")
            return
        decomposable_speaker_test_set_intent_distribution = compute_intent_distribution(decomposable_speaker_test,single_intent)
        decomposable_utterance_test_set_intent_distribution = compute_intent_distribution(decomposable_utterance_test,single_intent)

        kld_speaker = KL_divergence(original_test_set_intent_distribution, decomposable_speaker_test_set_intent_distribution)
        print(f"KL divergence between original test distribution of intents and Unseen-Speaker test distribution of intents: {kld_speaker}\n")
        kld_utterance = KL_divergence(original_test_set_intent_distribution, decomposable_utterance_test_set_intent_distribution)
        print(f"KL divergence between original test distribution of intents and Unseen-Utterance test distribution of intents: {kld_utterance}\n")
    else:
        train, test, valid = resplit_shuffled(concatenated_data, original_split_sizes)
    
    if not os.path.exists(output_dir):
        print(f"Created directory {output_dir}")
        os.makedirs(output_dir)

    new_train_path = os.path.join(output_dir, "train_data.csv")
    new_valid_path = os.path.join(output_dir, "valid_data.csv")
    if resplit_style == "decomposable":
        speaker_test_path = os.path.join(output_dir, f"speaker_test_data.csv")
        utterance_test_path = os.path.join(output_dir, f"utterance_test_data.csv")
        new_splits = [train, decomposable_speaker_test, decomposable_utterance_test, valid]
        new_paths = [new_train_path, speaker_test_path, utterance_test_path, new_valid_path]
        print(f"New dataset split sizes: {[len(train), len(decomposable_speaker_test), len(decomposable_utterance_test), len(valid)]}")
    else:
        new_test_set_intent_distribution = compute_intent_distribution(test)
        kld = KL_divergence(original_test_set_intent_distribution, new_test_set_intent_distribution)
        print(f"KL divergence between original test distribution of intents and new test distribution of intents: {kld}\n")

        new_test_path = os.path.join(output_dir, "test_data.csv")
        new_splits = [train, test, valid]
        new_paths = [new_train_path, new_test_path, new_valid_path]
        print(f"New dataset split sizes: {[len(train), len(test), len(valid)]}")

    for (data, path) in zip(new_splits, new_paths):
        print(f"Wrote file to {path}.")
        data.to_csv(path, index=True)

    print(f"Old dataset split sizes: {original_split_sizes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to root of dataset directory (either FSC or Snips)')
    parser.add_argument('--dataset', required=True, choices=["fluent_speech_commands", "snips"], help="Type of dataset to generate splits for.")
    parser.add_argument('--resplit_style', required=True, choices=['random', 'decomposable'], help='Path to root of fluent_speech_commands_dataset directory')
    parser.add_argument('--unseen', required=False, action='store_true', help="Whether to create unseen splits")
    parser.add_argument('--challenge', required=False, action='store_true', help="Whether to create challenge splits")
    parser.add_argument('--utility', action='store_true')
    # The following are under-the-hood arguments, that you probably won't need to use unless
    # tweaking our presented splitting methods.
    parser.add_argument('--BLEU', action='store_true')
    parser.add_argument('--use_speaker_utility', action='store_true')
    parser.add_argument('--mantain_length', action='store_true')
    parser.add_argument('--WER', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--use_ins_del_diff', action='store_true')
    parser.add_argument('--remove_absolute', action='store_true')
    parser.add_argument('--add_utterance', action='store_true')
    parser.add_argument('--try_diff_seed', action='store_true')

    args = parser.parse_args()
    single_intent = args.dataset=="snips"
    if args.unseen and args.challenge:
        raise ValueError("Can only request either unseen or challenge splits - not both.")
    if args.unseen or args.challenge:
        args.use_speaker_utility = True
        args.mantain_length = True
        args.utility = True
        if args.challenge:
            args.BLEU = True
            args.WER = True
            args.replace = True
            args.use_ins_del_diff = True
            args.remove_absolute = True

    main(args.data_dir, args.resplit_style, args.use_speaker_utility, args.dataset, args.utility, args.challenge, single_intent,args.mantain_length,args.replace, args.use_ins_del_diff,args.remove_absolute,args.add_utterance, args.try_diff_seed)

