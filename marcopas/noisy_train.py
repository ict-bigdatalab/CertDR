# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function
import sys, copy
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__)), '..'))




import argparse
import glob
import logging
import os
import random

import pickle
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME,
                          BertConfig,
                          BertTokenizer
                          )

from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import AdamW, WarmupLinearSchedule,  get_linear_schedule_with_warmup

from modeling import Ranking_BERT_Train
from marcopas.dataset import MSMARCONoisyDocDataset, get_collate_function
from utils import generate_rank, eval_results

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, Ranking_BERT_Train, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def train(args, model):
    """ Train the model """
    tb_writer = SummaryWriter() # 不指定路径时，会在当前目录下创建runs/日期_时间_host

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # perturbed training set

    train_dataset = MSMARCONoisyDocDataset("train", args.msmarco_dir, args.perturbed_collection_memmap_dir,
                                           args.query_tokenize_dir, args.bert_tokenizer_path,
                                           args.max_query_length, args.max_doc_length)
    # NOTE: Must Sequential! Pos, Neg, Pos, Neg, ...
    train_sampler = SequentialSampler(train_dataset)
    collate_fn = get_collate_function(mode="train")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=args.data_num_workers,
        collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility

    for epoch_idx, _ in enumerate(train_iterator):

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, _, _) in enumerate(epoch_iterator):

            batch = {k: v.to(args.device) for k, v in batch.items()}
            model.train()
            outputs = model(**batch)
            loss = outputs[0]  # model outputs are always t
            # uple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.evaluate_during_training and (global_step % args.training_eval_steps == 0):
                    mrr = evaluate(args, model, mode="dev", prefix="step_{}".format(global_step))
                    tb_writer.add_scalar('dev/MRR@100', mrr, global_step)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    cur_loss = (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(model, args.model_save_dir, 'ckpt-{}'.format(global_step), args)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, mode, prefix):
    eval_output_dir = args.eval_save_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_dataset = MSMARCONoisyDocDataset(mode, args.msmarco_dir, args.perturbed_collection_memmap_dir,
                                           args.query_tokenize_dir, args.bert_tokenizer_path,
                                           args.max_query_length, args.max_doc_length, small_test=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    collate_fn = get_collate_function(mode=mode)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                 num_workers=args.data_evaluate_num_workers, collate_fn=collate_fn)
    # 默认shuffle 是False，sampler 是SequentialSampler

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    output_file_path = f"{eval_output_dir}/{prefix}.{mode}.score.tsv"
    # i = 0
    with open(output_file_path, 'w') as outputfile:
        for batch, qids, docids in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            # if i>10: # 加速推断的流程，方便测试代码正确性，注意在正式代码中要注释掉
            #     break
            # i += 1
            with torch.no_grad():
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                # print(outputs)
                # print(outputs.shape)
                scores = outputs.detach().cpu().numpy()
                # print(qids, docids, scores)
                assert len(qids) == len(docids) == len(scores)
                for qid, docid, score in zip(qids, docids, scores):
                    outputfile.write(f"{qid}\t{docid}\t{score[0]}\n")

    rank_output = f"{eval_output_dir}/{prefix}.{mode}.rank.tsv"
    generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = eval_results(rank_output)
        return mrr


def main():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ")#.join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--msmarco_dir", type=str, default="/data/users/wuchen/pythonprojects/zjt_bert/data/msmarco_doc")
    parser.add_argument("--perturbed_collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--query_tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--bert_tokenizer_path", type=str, default="./data/tokenize")
    parser.add_argument("--data_num_workers", default=0, type=int)
    parser.add_argument("--data_evaluate_num_workers", default=0, type=int)
    parser.add_argument("--training_eval_steps", default=20000, type=int)

    parser.add_argument('--net_type', default='bert', type=str,
                        help='networktype: bert, textcnn, and so on')
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="")
    parser.add_argument("--max_doc_length", default=445, type=int,
                        help="The maximum document length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=30000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--similarity_threshold", default=0.5, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")
    parser.add_argument("--eval_ckpt", type=int, default=None, help='For evaluation')
    parser.add_argument("--eval_save_dir", type=str, default='')
    args = parser.parse_args()
    args.model_save_dir = f"{args.output_dir}/"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("Process device: %s, n_gpu: %s,",
                    device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Prepare task
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    if args.net_type == 'bert':
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    # Distributed and parallel training
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        logger.info("Saving model checkpoint to %s", args.model_save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        if args.net_type == 'bert':
            model_to_save.save_pretrained(args.model_save_dir)
        else:
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.model_save_dir, '/checkpoint.pth.tar'))
        tokenizer.save_pretrained(args.model_save_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.model_save_dir, 'training_args.bin'))

        if args.net_type == 'bert':
            # Load a trained model and vocabulary that you have fine-tuned 可能是load 一下看看对不对吧
            model = model_class.from_pretrained(args.model_save_dir)
            tokenizer = tokenizer_class.from_pretrained(args.model_save_dir)
            model.to(args.device)

    results = {}
    if args.do_eval:
        assert args.eval_ckpt is not None
        result = evaluate(args, model, args.task, prefix=f"ckpt-{args.eval_ckpt}")
        print(result)

    return results


if __name__ == "__main__":
    main()
