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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from sklearn.metrics import classification_report

from stancylib.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from stancylib.modeling import (BertConfig, WEIGHTS_NAME, CONFIG_NAME,
                                BertForSequenceClassificationDualLoss, BertForSequenceClassification)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from stancylib.processor import (ColaProcessor, MnliProcessor, Sst2Processor, StanceProcessor,
                                 MrpcProcessor, convert_examples_to_features, ProconProcessor)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def run_(data_dir: str, bert_model: str, task_name: str, output_dir: str,
                 cache_dir: str = '', max_seq_length: int = 128, do_train: bool = False,
                 do_eval: bool = False, do_lower_case: bool = False, train_batch_size: int = 32,
                 eval_batch_size: int = 8, learning_rate: float = 5e-5, num_train_epochs: float = 3.0,
                 warmup_proportion: float = 0.1, alpha: float = 0.5, no_cuda: bool = False,
                 local_rank: int = -1, seed: int = 42, gradient_accumulation_steps: int = 1,
                 fp16: bool = False, loss_scale: float = 0.0, server_ip: str = '', server_port: str = '',
                 dual_model: bool = True, pretrined=False):
    """
    Parameters-
      data_dir (str): Required, The input data dir. Should contain the .tsv files (or other data files) for the task.
      bert_model (str): Required, Bert pre-trained model selected in the list: bert-base-uncased,
                  bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
                  bert-base-multilingual-cased, bert-base-chinese.
      task_name (str): Required, The name of the task to train.
      output_dir (str): Required, The output directory where the model predictions and checkpoints will be written.
      cache_dir (str): Where do you want to store the pre-trained models downloaded from s3.
      max_seq_length (str): The maximum total input sequence length after WordPiece tokenization.
                            Sequences longer than this will be truncated, and sequences shorter
                            than this will be padded.
      do_train (bool): Whether to run training.
      do_eval (bool): Whether to run eval on the dev set.
      do_lower_case (bool):
      train_batch_size(int): Total batch size for training.
      eval_batch_size (int): Total batch size for eval.
      learning_rate (float): The initial learning rate for Adam.
      num_train_epochs (float): Total number of training epochs to perform.
      warmup_proportion (float): Proportion of training to perform linear learning rate warmup for.
                                 E.g., 0.1 = 10%% of training.
      alpha (float): Weight given to cross entropy loss. (1-alpha) weight for the cosine similarity.
      no_cuda (bool): Whether not to use CUDA when available.
      local_rank (int): local_rank for distributed training on gpus
      seed (int): random seed for initialization.
      gradient_accumulation_steps (int): Number of updates steps to accumulate before performing a backward/update pass.
      fp16 (bool): Whether to use 16-bit float precision instead of 32-bit.
      loss_scale (float): Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.
                          0 (default value): dynamic loss scaling.
                          Positive power of 2: static loss scaling value.
      server_ip (str): Can be used for distant debugging.
      server_port (str): Can be used for distant debugging.
      dual_model (bool): use dual model, paper SOTA algorithem for BERT.
      pretrined (bool): load pretrained model from output_dir when True, otherwise use default bert_model.
    """
    args = {
        'data_dir': data_dir,
        'bert_model': bert_model,
        'task_name': task_name,
        'output_dir': output_dir,
        'cache_dir': cache_dir,
        'max_seq_length': max_seq_length,
        'do_train': do_train,
        'do_eval': do_eval,
        'do_lower_case': do_lower_case,
        'train_batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'warmup_proportion': warmup_proportion,
        'alpha': alpha,
        'no_cuda': no_cuda,
        'local_rank': local_rank,
        'seed': seed,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'fp16': fp16,
        'loss_scale': loss_scale,
        'server_ip': server_ip,
        'server_port': server_port
    }
    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "stance": StanceProcessor,
        "procon": ProconProcessor
    }

    num_labels_task = {
        "cola": 2,
        "sst-2": 2,
        "mnli": 3,
        "mrpc": 2,
        "stance": 2,
        "procon": 2
    }

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
        if local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                         'distributed_{}'.format(local_rank))

    if dual_model:
        model = BertForSequenceClassificationDualLoss.from_pretrained(bert_model,
                                                                      cache_dir=cache_dir,
                                                                      num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels)

    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sim_label_ids = torch.tensor([f.sim_label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sim_label_ids)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        model.train()
        for _ in trange(int(num_train_epochs), desc="Epoch", position=0, leave=True, file=sys.stdout):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            process_bar = tqdm(train_dataloader, position=0, leave=True, file=sys.stdout)
            for step, batch in enumerate(process_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sim_label_ids = batch

                if dual_model:
                    loss = model(input_ids, segment_ids, input_mask, label_ids, sim_label_ids)
                else:
                    loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                process_bar.set_description("Loss: %0.8f" % (loss.sum().item()))

                if fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        # modify learning rate with special warm up BERT uses
                        # if fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                     warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            print("\nLoss: {}\n".format(tr_loss / nb_tr_steps))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)

        if dual_model:
            model = BertForSequenceClassificationDualLoss(config, num_labels=num_labels)
        else:
            model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForSequenceClassificationDualLoss.from_pretrained(bert_model, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file)) if pretrined else None
    model.to(device)

    if do_eval and (local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sim_label_ids = torch.tensor([f.sim_label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sim_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels = []
        predicted_prob = []
        gold_labels = []

        for input_ids, input_mask, segment_ids, label_ids, sim_label_ids in tqdm(eval_dataloader,
                                                                                 desc="Evaluating", position=0,
                                                                                 leave=True, file=sys.stdout):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            sim_label_ids = sim_label_ids.to(device)

            with torch.no_grad():
                if dual_model:
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, sim_label_ids)
                    logits = model(input_ids, segment_ids, input_mask)
                else:
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                predicted_prob.extend(torch.nn.functional.softmax(logits, dim=1))

            logits = logits.detach().cpu().numpy()

            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            tmp_predicted = np.argmax(logits, axis=1)
            predicted_labels.extend(tmp_predicted.tolist())
            gold_labels.extend(label_ids.tolist())

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_log_file = os.path.join(output_dir, "eval_details.txt")
        writer = open(output_eval_log_file, "w")
        for prob, pred_label, gold_label in zip(predicted_prob, predicted_labels, gold_labels):
            writer.write("{}\t{}\t{}\n".format(prob.cpu().tolist(), pred_label, gold_label))

        writer.close()

        print(classification_report(gold_labels, predicted_labels, target_names=label_list, digits=4))
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for k, v in sorted(args.items()):
                writer.write("{}={}\n".format(k, v))

            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write(classification_report(gold_labels, predicted_labels, target_names=label_list, digits=4))
