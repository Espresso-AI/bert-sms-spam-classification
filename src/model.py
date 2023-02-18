import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import AutoModel

Tensor = torch.Tensor


class SeqCls_Model(nn.Module):

    __doc__ = """
        It is a sequence classification model with adaptable loss function and classification head, 
        modified from transformers.BertForSequenceClassification.
        
        Args:
            base_checkpoint: BERT-structure checkpoints from huggingface
            num_classses: number of classes to predict
            loss_fn: loss function instance (eg. nn.CrossEntropyLoss())
            classifier_dropout: drop-out probability. 
                default: classifier_dropout if it is given in Config of the checkpoints,
                  else hidden_dropout_prob

        * classifier_dropout = 0 is not same with classifier_dropout = None.
        * Like the other model classes in transformers, model.training is set to False.
    """

    def __init__(
            self,
            base_checkpoint: str,
            num_classes: int,
            loss_fn: Optional[nn.Module] = None,
            classifier_dropout: Optional[float] = None,
    ):
        super().__init__()

        self.base_checkpoint = base_checkpoint
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.classifier_dropout = classifier_dropout

        self.base_model = AutoModel.from_pretrained(self.base_checkpoint)
        self.config = self.base_model.config
        self.find_dropout()
        self.head = self.classification_head()


    def find_dropout(self):
        if not self.classifier_dropout:
            p1, p2 = self.config.classifier_dropout, self.config.hidden_dropout_prob

            if any((p1, p2)):
                classifier_dropout = p1 or p2
            else:
                raise ValueError("dropout_prob for classification head is not given")
            self.classifier_dropout = classifier_dropout


    def classification_head(self):
        return nn.Sequential(
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.config.hidden_size, self.num_classes)
        ).eval()


    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Dict:

        outputs = self.base_model(
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
        logits = self.head(outputs[1])
        prediction = torch.argmax(logits, dim=-1)

        loss = None
        if not (self.loss_fn is None or labels is None):
            loss = self.loss_fn(logits, labels)

        return {
            'logits': logits,
            'prediction': prediction,
            'loss': loss,
        }