# source: https://github.com/huggingface/transformers/blob/b7439675b87dddac60091b7c4f574ee9d3b59e76/src/transformers/models/bert/modeling_bert.py#L1482

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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertPreTrainedModel, BertModel


class BertForSequenceClassificationDualLossV2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 1, config.num_labels)

        # addition
        self.cosine = nn.CosineSimilarity()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sim_labels=None  # addition, will be 1 or -1
    ):
        r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        # addition ---------------------------------------------------
        sen1_attention_mask = (1 - token_type_ids) * attention_mask
        outputs2 = self.bert(
            input_ids,
            attention_mask=sen1_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output_sen1 = outputs2[1]

        cos_sim = self.cosine(pooled_output, pooled_output_sen1).unsqueeze(1)
        combined = torch.cat([pooled_output, cos_sim], dim=1)
        # ---------------------------------------------------------------

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_bert = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # addition -----------------------------------
            loss_cosine = CosineEmbeddingLoss()
            loss_intent = loss_cosine(pooled_output, pooled_output_sen1, sim_labels.float())

            loss = loss_bert + loss_intent
            # --------------------------------------------
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
