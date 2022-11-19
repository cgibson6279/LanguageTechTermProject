# LanguageTechTermProject
T5 Paraphrase Generation Model

## Introduction

Paraphrase generation is a niche NLP task that takes a natural langauge sentence as input and generates a new sentence that maintains the same semantic quality of the original sentence, but is rearranged to have a different syntactic structure. In this project, I fine-tune a Text-to-Text Transfer Transformer (T5) to perform paraphrase generation on user data. 

The T5 model is an encoder-decorder model pre-trained for a variety of supervised and unsupervised tasks. In the case of a paraphrase generation task, T5 treats paraphrase generation as a fill-in-the-blank task where the model predicts missing words within a corupted piece of text. For example, if we give the model the input “I like to eat peanut butter and _4_ sandwiches,” we would train it to fill in the blank with approximately 4 words. 

## Methodology

I fine-tune my T5 model for paraphrase generation using the Google Paraphrase Adversaries from Word Scrambling (PAWS) dataset. The dataset contains 108,463 human-labeled and 656k noisily labeled pairs that feature the importance of modeling structure, context, and word order information for the problem of paraphrase identification. Specifically, I use the PAWS-Wiki Labeled (Final) subset of the dataset, which contains pairs that are generated from both word swapping and back translation methods. All pairs are labeled using human judgements on both paraphrasing and fluency. The data is split into train, dev, and test sets, and finally, I filter the data to only include positive paraphrase examples (label = 1). The model is trained for 10 epochs, and model output is evaluated using Levenshtein distance, sentence similarity based on co-sine similarity, and grammar checking.

## Conclusion



## Future Work

This model is meant to back-end utterance-generator scripts in the open source SCRAPI package, which is a high level Dialogflow CX wrapper, containing utilities that assist developers in building and maintaining custumer service AI agents in Dialogflow CX. While this model and these scripts a short term solution, the long term objective would be to register a model on Vertex AI and then build out a custom model building class for Vertex AI in SCRAPI, but this work is a good starting point for building out the utility for that class. 
