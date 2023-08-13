import torch
import unicodedata
from typing import Optional, Tuple
import numpy as np


def decode(
        text: Optional[str] = None,
        pred_tokens: Tuple[np.ndarray, np.ndarray] = None,
        ent_type: Optional[str] = None,
        offset: Optional[int] = 0,
):
    id2ent = {0:"O", 1:"B+sub", 2:"I+sub"}
    pred_tokens = pred_tokens.tolist()[0]
    bios = [id2ent[item] for item in pred_tokens if item != -1]
    bios = bios[offset:-1]  # 除去 query, CLS SEP token

    pred_ents = []
    start_index, end_index = -1, -1
    ent_type = None
    for indx, tag in enumerate(bios):
        if tag.startswith("B+"):
            if end_index != -1:
                pred_ents.append(
                    {
                        "start_idx": start_index,
                        "end_idx": end_index,
                        "type": ent_type,
                        "entity": text[start_index:end_index + 1]
                    }
                )
            # 新的实体
            start_index = indx
            end_index = indx
            ent_type = tag.split('+')[1]
            if indx == len(bios) - 1:
                pred_ents.append(
                    {
                        "start_idx": start_index,
                        "end_idx": end_index,
                        "type": ent_type,
                        "entity": text[start_index:end_index + 1]
                    }
                )
        elif tag.startswith('I+') and start_index != -1:
            _type = tag.split('+')[1]
            if _type == ent_type:
                end_index = indx

            if indx == len(bios) - 1:
                pred_ents.append(
                    {
                        "start_idx": start_index,
                        "end_idx": end_index,
                        "type": ent_type,
                        "entity": text[start_index:end_index + 1]
                    }
                )
        else:
            if end_index != -1:
                pred_ents.append(
                    {
                        "start_idx": start_index,
                        "end_idx": end_index,
                        "type": ent_type,
                        "entity": text[start_index:end_index + 1]
                    }
                )
            start_index, end_index = -1, -1
            ent_type = None
    return pred_ents

def extract_triples(model, text, args, tokenizer, id2label, max_seq_len):

    encode_dict = tokenizer.encode_plus(text=text,
                                        max_length=max_seq_len,
                                        truncation=True,
                                        add_special_tokens=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        return_offsets_mapping=True,
                                        return_tensors='pt')
    input_ids = encode_dict['input_ids'].to(args.device)
    attention_masks = encode_dict['attention_mask'].to(args.device)
    token_type_ids = encode_dict['token_type_ids'].to(args.device)
    mapping = encode_dict['offset_mapping'].squeeze()   # token2char_mapping

    embed = model.bert(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_masks)[0]

    _, sub_decode_seqs = model.forward_sub(embed, attention_masks)

    pred_ents = decode(text, sub_decode_seqs)
    subjects = []
    for ent in pred_ents:
        subjects.append((ent["start_idx"], ent["end_idx"]))

    if subjects:
        spo_list = []
        all_sub_head = torch.tensor([sub[0] for sub in subjects])
        all_sub_tail = torch.tensor([sub[1] for sub in subjects])

        embed_repeat = embed.repeat(len(subjects), 1, 1)  # [num_subjects , seq_len, bert_dim(768)]
        obj_heads_logits, obj_tails_logits = model.forward_obj(embed_repeat, all_sub_head, all_sub_tail)

        # [num_subjects, seq_len, num_relations]
        for j, subject in enumerate(subjects):
            obj_heads = torch.where(obj_heads_logits[j] > args.h_bar)  # 返回两个向量，第一个向量是行数（obj），第二个向量是列数（rel)
            obj_tails = torch.where(obj_tails_logits[j] > args.t_bar)

            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        spo_list.append(((mapping[subject[0]][0],
                                          mapping[subject[1]][-1]),
                                          rel_head.item(),
                                         (mapping[obj_head][0],
                                          mapping[obj_tail][-1])))

                        break
        triples = []
        for s, p, o in spo_list:
            triples.append((
                text[s[0]:s[1]],
                id2label[p],
                text[o[0]:o[1]]
            ))
        return triples
    else:
        return []