# Multi-Aspect Subtask Evaluation (MASE)

## Datasets
All the code in this repo will require you to have a copy of the Fluent Speech Commands dataset or the Snips SmartLights Dataset.

**Fluent Speech Commands** (FSC) was introduced by [Lugosch et al](https://arxiv.org/pdf/1904.03670.pdf) [1]), and can be downloaded [here](
https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/).

**Snips SmartLights** was introduced by [Coucke et al](https://arxiv.org/pdf/1810.12735.pdf), and access can be requested [here](https://github.com/sonos/spoken-language-understanding-research-datasets). To simplify things, we have converted the Snips dataset into the FSC format - this data can be found under `slu_splits/snips_close_field/original_splits`. To comply with their license, we do not include the actual audio files. When you download Snips data, move the `snips_slu_data_v1.0` folder into the root of this directory, and copy `slu_splits/snips_close_field/*` into `snips_slu_data_v1.0`.


## Resplit Data
To generate the "Unseen Splits" described in our paper for FSC, run:
`python processing_scripts/resplit_data.py --data_dir <PATH_TO_FSC_DATASET> --dataset fluent_speech_commands --resplit_style decomposable --challenge`

To generate "Challenge Splits" for FSC, run:  
`python processing_scripts/resplit_data.py --data_dir <PATH_TO_FSC_DATASET> --dataset fluent_speech_commands --resplit_style decomposable --unseen`

To generate these splits for Snips, replace `<PATH_TO_FSC_DATASET>` with the path to the Snips dataset, and replace the `--dataset` argument with `snips`, in the command.

Then, a new directory with name "unseen_splits" or "challenge_splits" will be created within the provided dataset path.

## Train end-to-end SLU models
Train end-to-end SLU models used in the papers "[Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670)" and "[Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models](https://arxiv.org/abs/1910.09463)".

On original splits for FSC, run-
`python main.py --train --config_path=<path to .cfg>  --resplit_style=original`

On "Unseen Splits" for FSC, run-
`python main.py --train --config_path=<path to .cfg>  --resplit_style=unseen`

On "Challenge Splits" for FSC, run-
`python main.py --train --config_path=<path to .cfg>  --resplit_style=challenge`

To test the model having best validation accuracy, use `--save_best_model` with the command. To run these models for Snips, add `--single_intent` to the command and use the corresponding Snips config file provided by use.

To add semantic word embeddings to the SLU system, use the following command-
`python main.py --train --config_path=<path to .cfg>  --resplit_style=<resplit style>  --use_FastText_embeddings --semantic_embeddings_path <FastText_EMBEDDING_PATH> --smooth_semantic --smooth_semantic_parameter K`

These commands can be run with any model training config (e.g. `no_unfreezing` or `unfreeze_word_layers`). Just make sure you update the training config such that `slu_path` points to your local dataset directory.

## References:
[1] Loren Lugosch,  Mirco Ravanelli,  Patrick Ignoto,  Vikrant Tomar,  and Yoshua Bengio.   Speech model pre-training for end-to-end spoken language understanding. pages 814–818, 09 2019. doi:10.21437/Interspeech.2019-2396.

[2] Alaa Saade, Alice Coucke, Alexandre Caulier, Joseph Dureau, Adrien Ball, Théodore Bluche, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, and Maël Primet.  Spoken  language  understanding  on  the  edge. Energy Efficient Machine Learning and Cognitive Computing workshop, NeurIPS. 2019

## Cite This Work
```
@inproceedings{Interspeech21mase,
    title = { Rethinking End-to-End Evaluation of Decomposable Tasks: A Case Study on Spoken Language Understanding},
    author = {Siddhant Arora and Alissa Ostapenko and Vijay Viswanathan and Siddharth Dalmia and Florian Metze and Shinji Watanabe and Alan W Black},
    booktitle = {22nd Annual Conference of the International Speech Communication Association (Interspeech 2021)},
    address = {Brno, Czech Republic},
    month = {August},
    year = {2021}
}
```

