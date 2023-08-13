# from models.casrel import CasRel
# import torch
# import os
# from tqdm import tqdm
# from transformers import BertTokenizerFast
# from utils.utils import extract_triples
# import json
# from process_data import RelationProcessor
# from collections import defaultdict
# import argparse
#
#
# class Ensemble():
#     def __init__(self, model_path_list, bert_dir, num_relations, device):
#         self.models = []
#         for idx, _path in enumerate(model_path_list):
#             print(f'Load model from {_path}')
#             model = CasRel(bert_dir, num_relations, inner_code='mean')
#             model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))
#             model.eval()
#             model = model.to(device)
#             self.models.append(model)
#         self.device = device
#         self.tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
#
#     def filter(self, dev_examples, id2relation, set_type='dev'):
#         add, remove = 0, 0
#         f = open(os.path.join(args.result_dir, 'CMeIE_' + set_type + '_filter.json'), 'w')
#         for example in tqdm(dev_examples, desc='Evaluating'):
#             text = example.text_a
#             gold_triples = [(triple['subject'], triple['predicate'], triple['object']['@value'])
#                             for triple in example.label]
#
#             cnt_dict = defaultdict(int)
#             for model in self.models:
#                 pred_triples = extract_triples(model, text, args, tokenizer, id2relation,
#                                                args.max_seq_length)
#                 pred = set(pred_triples)
#                 for p in pred:
#                     cnt_dict[p] += 1
#
#             for k,v in cnt_dict.items():
#                 if v == len(self.models) and k not in gold_triples:
#                     gold_triples.append(k)
#                     add += 1
#
#             for p in gold_triples:
#                 if p not in cnt_dict:
#                     gold_triples.remove(p)
#                     remove += 1
#
#             line = {}
#             line['text'] = text
#             tmp = []
#             for s, p, o in gold_triples:
#                 for l in example.label:
#                     if l['subject'] == s and l['predicate'] == p and l['object']['@value']:
#                         tmp.append(l)
#             line['entities'] = tmp
#             f.write(json.dumps(line, ensure_ascii=False) + '\n')
#
#         print(f'{set_type}: add {add} remove {remove}')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--raw_data_dir', type=str, default='./data/CMeIE')
#     parser.add_argument('--output_dir', type=str, default='./output/')
#     parser.add_argument('--result_dir', type=str, default='./data/CMeIE')
#     parser.add_argument('--bert_dir', type=str, default='../PTM/roberta-wwm-ext/')
#     parser.add_argument('--max_seq_len', type=int, default=512)
#     args = parser.parse_args()
#
#     processor = RelationProcessor()
#     pos2id = None
#     relation2id, id2relation = processor.get_relation2id(args.raw_data_dir)
#     num_relations = len(relation2id)
#
#     tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir, do_lower_case=True)
#     train_examples = processor.get_train_examples(args.raw_data_dir)
#     dev_examples = processor.get_dev_examples(args.raw_data_dir)
#
#     args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     paths = os.listdir(args.output_dir)
#     model_path_list = []
#     for p in paths:
#         if 'mean' in p:
#             model_path_list.append(os.path.join(args.output_dir, 'pytorch_model.bin'))
#
#     ensembel = Ensemble(model_path_list, args.bert_dir, num_relations, args.device)
#     ensembel.filter(dev_examples, id2relation, 'dev')
#     ensembel.filter(train_examples, id2relation, 'train')
#
#
import json
path = './raw_data/CMeIE/CMeIE_train.json'

with open(path, 'r') as f:
    data = [json.loads(l) for l in f]

cnt = 0
lents = []
for line in data:
    for ent in line['spo_list']:
        lents.append(len(ent['subject']))
        if len(ent['subject']) > 50:
           cnt += 1
print(max(lents))
print(cnt)