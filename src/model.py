import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from src.crf import CRF


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        # 此处全0初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        '''
            inputs: [b, s, h]
            cond: [b, h]
        '''
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'

        cond = torch.unsqueeze(cond, 1)  # (b, 1, h)
        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mu = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
#         std = torch.std(inputs, dim=-1, keepdim=True)  #  (b, s, 1)
        outputs = inputs - mu  # (b, s, h)
        var = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)  # (b, s, 1)
        outputs = weight * (inputs - mu) / (std + self.eps) + bias
        return outputs


class CasRel(nn.Module):
    def __init__(self, bert_path, num_relations=51, inner_code='lstm'):
        super(CasRel, self).__init__()
        self.num_relations = num_relations
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size

        crf_hidden_size = 128
        num_ner_labels = 3
        self.crf_fc = nn.Sequential(
            nn.Linear(hidden_size, crf_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(crf_hidden_size, num_ner_labels)
        )
        self.sub_crf = CRF(num_tags=num_ner_labels, batch_first=True)

        self.obj_heads_classifier = nn.Linear(hidden_size, num_relations)
        self.obj_tails_classifier = nn.Linear(hidden_size, num_relations)
        self.sub_cln = ConditionalLayerNorm(hidden_size, hidden_size)
        self.obj_cln = ConditionalLayerNorm(hidden_size, hidden_size)

        self.inner_code = inner_code
        self.sub_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(0.1)
        self.init_weights([
                           self.obj_heads_classifier, self.obj_tails_classifier])

    def init_weights(self, blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward_sub(self, embed, attention_mask, labels=None):
        '''
        :param embed:  [batch_size, seq_len, bert_dim(768)
        :return: [batch_sizee, seq_len]
        '''
        emissions = self.crf_fc(embed)
        sub_crf_loss = None
        if labels is not None:
            sub_crf_loss = -1. * self.sub_crf(
                emissions=emissions,
                tags=labels.long(),
                mask=attention_mask.byte(),
                reduction="mean"
            )
        sub_decode_seqs = None
        if not self.training:
            seq_length = embed.size(1)
            sub_decode_seqs = self.sub_crf.decode(emissions=emissions, mask=attention_mask.byte())

            for line in sub_decode_seqs:
                padding = [-1] * (seq_length - len(line))
                line += padding

            sub_decode_seqs = torch.tensor(sub_decode_seqs).to(embed.device)

        return sub_crf_loss, sub_decode_seqs


    def forward_obj(self, embed, sub_head, sub_tail):
        '''
        :param embed: [batch_size, seq_len, bert_dim(768)
        :param sub_head: [batch_size]
        :param sub_tail:
        :return:
        '''
        batch_size = embed.size(0)
        index = torch.arange(batch_size).cuda()

        if self.inner_code == 'lstm':
            sub_embed = []
            for idx in index:
                sub_head_tail_embed = embed[idx, sub_head[idx]:sub_tail[idx] + 1]  # [head:tail, bert_dim]
                _, (hn, cn) = self.sub_lstm(sub_head_tail_embed.unsqueeze(0))
                sub_embed.append(hn.squeeze())
            sub_embed = torch.stack(sub_embed, dim=0)  # [batch_size, bert_dim]
        else:
            sub_head_embed = embed[(index, sub_head)]
            sub_tail_embed = embed[(index, sub_tail)]
            sub_embed = torch.stack([sub_head_embed, sub_tail_embed]).mean(0)

        new_embed = self.obj_cln(embed, sub_embed)
        # [batch_size, seq_len, num_relations]
        pred_obj_heads_logits = torch.sigmoid(self.obj_heads_classifier(new_embed))
        # [batch_size, seq_len, num_relations]
        pred_obj_tails_logits = torch.sigmoid(self.obj_tails_classifier(new_embed))
        return pred_obj_heads_logits, pred_obj_tails_logits

    def obj_loss(self, obj_heads, obj_tails,
             pred_obj_heads_logits, pred_obj_tails_logits, input_mask):
        '''
        :param sub_heads: [batch_size, seq_len]
        :param sub_tails:
        :param obj_heads: [batch_size, seq_len, num_rel]
        :param obj_tails:
        :param pred_sub_biaffine_logits: [batch_size seq_len, seq_len]
        :param pred_obj_heads_logits: [batch_size, seq_len, num_rel]
        :param pred_obj_tails_logits:
        :param input_mask: [batch_size, seq_len]
        :return:
        '''

        obj_heads_loss = F.binary_cross_entropy(pred_obj_heads_logits, obj_heads.float(), reduction='none')
        obj_heads_loss = torch.sum(obj_heads_loss, dim=-1)  # [batch_size, seq_len]
        obj_heads_loss = torch.sum(obj_heads_loss * input_mask) / input_mask.sum()
        obj_tails_loss = F.binary_cross_entropy(pred_obj_tails_logits, obj_tails.float(), reduction='none')
        obj_tails_loss = torch.sum(obj_tails_loss, dim=-1)
        obj_tails_loss = torch.sum(obj_tails_loss * input_mask) / input_mask.sum()

        loss = (obj_heads_loss + obj_tails_loss)
        return loss

    def forward(self, input_ids, segment_ids, input_mask,
                sub_head, sub_tail, obj_heads, obj_tails, sub_bios_labels):

        # [batch_size, seq_len, 768]
        embed = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        embed = self.dropout(embed)

        # [batch_size, seq_len, seq_len]
        sub_crf_loss, _ = self.forward_sub(embed, input_mask, sub_bios_labels)

        # [batch_size, seq_len, num_relations]
        pred_obj_heads_logits, pred_obj_tails_logits = self.forward_obj(embed, sub_head, sub_tail)
        obj_loss = self.obj_loss(obj_heads, obj_tails,
             pred_obj_heads_logits, pred_obj_tails_logits, input_mask)

        return sub_crf_loss + obj_loss