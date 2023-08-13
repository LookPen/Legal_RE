import torch
import json
import os
from random import choice, sample
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler


class InputExample(object):

    def __init__(self, sid, text_a, postag=None, text_b=None, label=None):
        self.sid = sid
        self.text_a = text_a
        self.text_b = text_b
        self.postag = postag
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def _read_jsonl(cls, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]


class RelationProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train_re.json"))["data"], "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test_re.json"))["data"], "dev")

    def get_test_examples(self, data_dir):
        return self._create_test_examples(
            self._read_json(os.path.join(data_dir, "test_re.json"))["data"])

    def get_relation2id(self, data_dir):
        relations = self._read_json(os.path.join(data_dir, "schemas.json"))
        relation2id = {}
        for id, rel in enumerate(relations):
            # rel = line['predicate']
            if rel not in relation2id:
                relation2id[rel] = len(relation2id)

        id2relation = {v: k for k, v in relation2id.items()}
        return relation2id, id2relation

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            sid = "%s-%s" % (set_type, i)
            text_a = line['text']
            text_b = None
            postag = None
            label = line['spo_list']
            examples.append(
                InputExample(sid=sid, text_a=text_a, postag=postag, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            sid = "%s-%s" % ('test', i)
            text_a = line['text']
            text_b = None
            postag = None
            label = None
            examples.append(
                InputExample(sid=sid, text_a=text_a, postag=postag, text_b=text_b, label=label))
        return examples


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def convert_examples_to_features(examples, pos2id, relation2id, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        s2ro_map = {}
        for label in example.label:
            rel = label['predicate']
            triple = (tokenizer.tokenize(label['subject']), rel, tokenizer.tokenize(label['object']))

            sub_head_idx = find_head_idx(tokens, triple[0])
            obj_head_idx = find_head_idx(tokens, triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                if sub not in s2ro_map:
                    s2ro_map[sub] = []

                s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, relation2id[rel]))

        if s2ro_map:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              labels=s2ro_map))
    return features


class Dataset(torch.utils.data.Dataset):

    def __init__(self, features, num_relations, max_seq_len, k):
        super(Dataset, self).__init__()
        self.features = features
        self.num_relations = num_relations
        self.max_length = max_seq_len
        self.k = k

    def __getitem__(self, index):

        return self.features[index]

    def __len__(self):
        return len(self.features)

    def my_collate(self, batch):

        batch_max = 0
        for line in batch:
            if len(line.input_ids) > batch_max:
                batch_max = len(line.input_ids)
        if batch_max > self.max_length:
            batch_max = self.max_length

        input_ids, input_mask, segment_ids = [], [], []
        sub_head, sub_tail, obj_heads, obj_tails =  [], [], [], []
        # sub_biaffine = []
        sub_bios = []
        for line in batch:
            padding = [0] * (batch_max - len(line.input_ids))
            input_ids.append(line.input_ids + padding)
            input_mask.append(line.input_mask + padding)
            segment_ids.append(line.segment_ids + padding)

            # sub_biaffine_ = torch.zeros(batch_max, batch_max)
            sub_bio = [0] * batch_max
            for sub in line.labels:
                # sub_biaffine_[sub[0], sub[1]] = 1
                sub_bio[sub[0]] = 1
                for i in range(sub[0]+1, sub[1]+1):
                    sub_bio[i] = 2

            sub_head_, sub_tail_ = choice(list(line.labels.keys()))
            obj_heads_, obj_tails_ = torch.zeros(batch_max, self.num_relations), torch.zeros(batch_max,
                                                                                             self.num_relations)
            for ro in line.labels.get((sub_head_, sub_tail_), []):
                obj_heads_[ro[0]][ro[2]] = 1
                obj_tails_[ro[1]][ro[2]] = 1

            sub_head.append(torch.tensor(sub_head_))
            sub_tail.append(torch.tensor(sub_tail_))
            obj_heads.append(obj_heads_)
            obj_tails.append(obj_tails_)
            # sub_biaffine.append(sub_biaffine_)
            sub_bios.append(sub_bio)


        # sub_biaffine = torch.stack(sub_biaffine, dim=0) # [batch_size, seq_len, seq_len]
        # if len(sub_biaffine.size()) == 2:
        #     sub_biaffine = sub_biaffine.unsqueeze(0)
        sub_bios = torch.tensor(sub_bios)
        sub_head = torch.stack(sub_head, dim=0)  # [batch_size]
        sub_tail = torch.stack(sub_tail, dim=0)
        obj_heads = torch.stack(obj_heads, dim=0)  # [batch_size, 200]
        obj_tails = torch.stack(obj_tails, dim=0)

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        segment_ids = torch.tensor(segment_ids)

        return input_ids, input_mask, segment_ids, sub_head, sub_tail, \
               obj_heads, obj_tails, sub_bios


def get_dataloader(examples, pos2id, relation2id, max_seq_length, tokenizer, args, set_type='Train'):
    features = convert_examples_to_features(examples, pos2id, relation2id, max_seq_length, tokenizer)
    dataset = Dataset(features, len(relation2id), max_seq_length, args.num_samples)
    if set_type == 'Train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=dataset.my_collate,
                            batch_size=args.train_batch_size, num_workers=4)
    return dataloader, dataset, len(features)