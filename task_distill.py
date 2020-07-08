# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from torch.nn import MSELoss

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_masks, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


def convert_examples_to_features(examples, tokenizer, max_seq_len):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        else:
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[:(max_seq_len - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(example.label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_masks=input_mask,
                          segment_ids=segment_ids,
                          label_id=example.label,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def read_examples(path):
    examples = []
    with open(path, 'r') as fr:
        for index, line in enumerate(fr):
            line = line.strip('\n').strip().split('\t')
            label = line[-1]
            text_a = line[0]
            if len(line) > 2:
                text_b = line[1]
            else:
                text_b = None
            try:
                label = int(label)
            except Exception:
                continue
            examples.append(InputExample(guid=index, text_a=text_a, text_b=text_b, label=label))
    return examples


class MyDataLoader(object):

    def __init__(self, data):
        self.data = data
        self.data_size = len(data)

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        return (torch.tensor(self.data[item].input_ids).long(),
                torch.tensor(self.data[item].input_masks).long(),
                torch.tensor(self.data[item].segment_ids).long(),
                torch.tensor(self.data[item].label_id).byte(),
                torch.tensor(self.data[item].seq_length).long())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files or the task.")
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where model checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_len",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length ")
    parser.add_argument("--num_labels", default=2, type=int, required=True, help="")
    parser.add_argument("--task_mode",
                        default='classification',
                        type=str,
                        required=False,
                        help="task type")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run train on the train set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate")

    # added arguments
    parser.add_argument('--aug_train', action='store_true')
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--pred_distill', action='store_true')
    parser.add_argument('--data_url', type=str, default="")
    parser.add_argument('--temperature', type=float, default=1.)

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(
            args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    num_labels = args.num_labels

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if args.do_train:
        train_path = os.path.join(args.data_dir, 'train.txt')
        eval_path = os.path.join(args.data_dir, 'eval.txt')
        train_examples = read_examples(train_path)
        eval_examples = read_examples(eval_path)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_len)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_len)

        train_features = MyDataLoader(train_features)
        eval_features = MyDataLoader(eval_features)

        train_dataloader = DataLoader(train_features,
                                      shuffle=True,
                                      batch_size=args.train_batch_size)
        # eval_dataloader = DataLoader(eval_features, shuffle=False, batch_size=args.eval_batch_size)

        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model,
                                                                          num_labels=num_labels)
        teacher_model.to(device)

    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model,
                                                                      num_labels=num_labels)
    student_model.to(device)
    # 只做预测
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        }, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # Prepare loss functions
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (-targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.

                student_logits, student_atts, student_reps = student_model(input_ids,
                                                                           segment_ids,
                                                                           input_mask,
                                                                           is_student=True)

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(
                        input_ids, segment_ids, input_mask)

                # 第一阶段
                if not args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [
                        teacher_atts[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)
                    ]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2,
                                                  torch.zeros_like(student_att).to(device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2,
                                                  torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [
                        teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)
                    ]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                # 第二阶段
                else:
                    if args.task_mode == "classification":
                        cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    elif args.task_mode == "regression":
                        loss_mse = MSELoss()
                        cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()    # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    result = {}
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss

                    result_to_file(result, output_eval_file)

            logger.info("***** Save model *****")
            model_to_save = student_model.module if hasattr(student_model,
                                                            'module') else student_model
            model_name = f'{epoch_}_{WEIGHTS_NAME}'
            output_model_file = os.path.join(args.output_dir, model_name)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)


if __name__ == "__main__":
    main()
