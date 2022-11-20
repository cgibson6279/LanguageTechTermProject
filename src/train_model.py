"""Utility functions for fine tuning pretrained T5 summarization model for paraphrasing."""

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#https://github.com/seduerr91/pawraphrase_public/blob/master/t5_pawraphrase_training.ipynb

import argparse
import os
import logging

import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# Construction 1
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
spacy_tokenizer = Tokenizer(nlp.vocab)

import pytorch_lightning as pl

from model import LoggingCallback, FineTuneT5Model

DATA_DIR = "data/final/"
MODEL_NAME = "t5-small"
TRAIN_FILE = "train"
VAL_FILE = "val"
TEST_FILE = "test"
FILE_TYPE = ".tsv"

OUTPUT_DIR = "cpk/"
FILE_TYPE = ".tsv"
MODEL_SAVE_PATH = "t5_small_paraphrase/"

#!pip install torch==1.4.0 -q
#!pip install transformers==2.9.0 -q
#!pip install pytorch_lightning==0.7.5 -q


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
        ) 

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        # precision=32,
        # amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(logger=logger)],
    )
    # Initialize Model
    print("Intializing Model...")
    model = FineTuneT5Model(
        adam_epsilon = args.adam_epsilon,
        eval_batch_size = args.eval_batch_size,
        file_type = FILE_TYPE,
        data_dir = DATA_DIR,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        model_name_or_path = args.model_name_or_path,
        n_gpu = args.n_gpu,
        num_train_epochs = args.num_train_epochs,
        tokenizer_name_or_path = args.tokenizer_name_or_path,
        train_batch_size = args.train_batch_size,
        warmup_steps = args.warmup_steps,
        weight_decay = args.weight_decay,
        train_dataset_file_name = TRAIN_FILE, 
        val_dataset_file_name = VAL_FILE,
        #test_data_file_name = TEST_FILE
    )
    #Initalize Trainer
    print("Initalizing Trainer...")
    trainer = pl.Trainer(**train_params)

    # Start Fine-Tuning
    print("Starting Fine Tuning...") # TODO: add timing
    trainer.fit(model)
    print("Fine Tuning Complete in: ")

    print("Saving model...")
    try:
        # TODO: Save tokenizer
        model.model.save_pretrained(MODEL_SAVE_PATH)
    except AssertionError:
        try:
            model.model.save_pretrained(MODEL_SAVE_PATH)
        except AssertionError:
            model.model.save_pretrained(os.getcwd())
    print('Saved model...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_dir", help="", nargs='?', type=str, const="data/final/", default="data/final/")
    parser.add_argument("output_dir", help="", nargs='?', type=str, const="cpk/", default="cpk/")
    parser.add_argument("model_name_or_path", help="", nargs='?', type=str, const="t5-small", default="t5-small")
    parser.add_argument("tokenizer_name_or_path", help="Debug", nargs='?', type=str, const="t5-small", default="t5-small")
    #parser.add_argument("max_seq_length", help="", nargs='?', type=int, const=512, default=512)
    parser.add_argument("learning_rate", help="", nargs='?', type=float, const=3e-4, default=3e-4)
    parser.add_argument("weight_decay", help="", nargs='?', type=float, const=0.0, default=0.0)
    parser.add_argument("adam_epsilon", help="", nargs='?', type=float, const=1e-8, default=1e-8)
    parser.add_argument("warmup_steps", help="", nargs='?', type=int, const=0, default=0)
    parser.add_argument("train_batch_size", help="", nargs='?', type=int, const=4, default=4)
    parser.add_argument("eval_batch_size", help="", nargs='?', type=int, const=4, default=4)
    parser.add_argument("num_train_epochs", help="Debug", nargs='?', type=int, const=10, default=10)
    parser.add_argument("gradient_accumulation_steps", help="", nargs='?', type=int, const=16, default=16)
    parser.add_argument("n_gpu", help="", nargs='?', type=int, const=1, default=1)
    parser.add_argument("early_stop_callback", help="", nargs='?', type=bool, const=False, default=False)
    parser.add_argument("fp_16", help="Debug", nargs='?', type=bool, const=False, default=False)
    parser.add_argument("opt_level", help="", nargs='?', type=str, const='O1', default='O1')
    parser.add_argument("max_grad_norm", help="", nargs='?', type=float, const=1.0, default=1.0)
    parser.add_argument("seed", help="", nargs='?', type=int, const=42, default=42)
    main(parser.parse_args())