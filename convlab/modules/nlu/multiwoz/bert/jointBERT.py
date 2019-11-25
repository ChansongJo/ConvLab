import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, device, slot_dim, intent_dim, intent_weight=None, context=False):
        super(JointBERT, self).__init__(config)
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.context = context
        if context:
            self.intent_classifier = nn.Linear(2 * config.hidden_size, self.intent_num_labels)
        else:
            self.intent_classifier = nn.Linear(config.hidden_size, self.intent_num_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, self.slot_num_labels)
        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
        outputs = self.bert(input_ids=word_seq_tensor,
                            attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        if self.context and context_seq_tensor is not None:
            context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, (slot_loss), (intent_loss),
