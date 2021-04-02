import numpy as np

def KL_divergence(f, g):
    # Compute KL divergence between two discrete distributions, each representing a joint probability
    # distribution over intents in the test set.
    kl = 0.0
    if len(f) != len(g):
        raise ValueError("Distributions are not over the same set of objects")
    for ((intent1, f_prob), (intent2, g_prob)) in zip(f,g):
        if intent1 != intent2:
            raise ValueError("Intents are not aligned!")
        kl += f_prob * (np.log2(f_prob) - np.log2(g_prob))
    return kl

def compute_intent_distribution(df,single_intent=False):
    # print(single_intent)
    if single_intent:
        try:
            groups = df.groupby(['intentLbl'])
        except:
            from IPython import embed; embed(); raise ValueError("!")
    else:
        groups = df.groupby(['action', 'object', 'location'])
    sizes = []
    total_size = 0.0
    for k, g in groups:
        sizes.append((k, len(g)))
        total_size += len(g)
    sizes = sorted(sizes, key=lambda x: x[0])
    sizes = [(x, y / total_size) for (x,y) in sizes]
    # print(sizes)
    return sizes

def compute_intent_distribution_combine(train_sizes,test_sizes):
    # print(single_intent)
    train_size_dict={}
    for (x,y) in train_sizes:
        train_size_dict[x]=y
    test_size_dict={}
    for (x,y) in test_sizes:
        test_size_dict[x]=y
    for x in train_size_dict:
        if x not in test_size_dict:
            test_size_dict[x]=10**-5
    for x in test_size_dict:
        if x not in train_size_dict:
            train_size_dict[x]=10**-5
    train_sizes=[]
    for x in train_size_dict:
        train_sizes.append((x,train_size_dict[x]))
    train_sizes = sorted(train_sizes, key=lambda x: x[0])
    
    test_sizes=[]
    for x in test_size_dict:
        test_sizes.append((x,test_size_dict[x]))
    test_sizes = sorted(test_sizes, key=lambda x: x[0])
    # print(sizes)
    return train_sizes,test_sizes