import os
import torch
import json
import logging
from tqdm import tqdm
from .utils.attack_train import *
from .utils.ema import EMA
from .utils.extract_triple import extract_triples
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, args, model, t_total):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.optimizer, self.scheduler = self.build_optimizer_and_scheduler(t_total)

    def build_optimizer_and_scheduler(self, t_total):
        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(self.model.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []
        for name, para in model_param:
            space = name.split('.')
            if space[0] == 'bert':
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert  module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps)
        if self.args.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
            )
        elif self.args.scheduler == 'cyc':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.args.lr, max_lr=5e-5,
                                                          step_size_up=t_total/100*2,
                                                          cycle_momentum=False, mode="triangular2")
        else:
            scheduler = None
        return optimizer, scheduler

    def train_model(self, train_dataloader, eval_examples, tokenizer, id2relation):
        # 权重滑动平均
        ema = EMA(self.model, 0.999)
        ema.register()
        # 对抗训练
        fgm, pgd = None, None
        attack_train_mode = self.args.attack_train.lower()
        if attack_train_mode == 'fgm':
            fgm = FGM(model=self.model)
        elif attack_train_mode == 'pgd':
            pgd = PGD(model=self.model)

        global_step = 0
        best_eval_f1 = 0.0
        patience = 1
        pgd_k = 3
        try:
            with tqdm(range(int(self.args.num_train_epochs)), desc="Epoch") as epochs:
                for epoch in epochs:
                    tr_loss = 0
                    self.model.train()

                    with tqdm(train_dataloader, desc="Iteration") as iters:
                        for step, batch in enumerate(iters):
                            batch = (b.to(self.device) for b in batch)
                            input_ids, input_mask, segment_ids, sub_head, \
                            sub_tail, obj_heads, obj_tails, sub_biaffine = batch

                            loss = self.model(input_ids, segment_ids, input_mask,
                                              sub_head, sub_tail, obj_heads, obj_tails, sub_biaffine)

                            iters.set_description("loss: {:f}".format(loss))

                            if self.args.gradient_accumulation_steps > 1:
                                loss = loss / self.args.gradient_accumulation_steps

                            self.optimizer.zero_grad()
                            loss.backward()
                            if fgm is not None:
                                fgm.attack()
                                loss_adv = self.model(input_ids, segment_ids, input_mask,
                                                      sub_head, sub_tail, obj_heads, obj_tails, sub_biaffine)
                                loss_adv.backward()
                                fgm.restore()
                            elif pgd is not None:
                                pgd.backup_grad()
                                for _t in range(pgd_k):
                                    pgd.attack(is_first_attack=(_t == 0))
                                    if _t != pgd_k - 1:
                                        self.model.zero_grad()
                                    else:
                                        pgd.restore_grad()
                                    loss_adv = self.model(input_ids, segment_ids, input_mask,
                                                          sub_head, sub_tail, obj_heads, obj_tails, sub_biaffine)
                                    loss_adv.backward()
                                pgd.restore()

                            self.optimizer.step()
                            if len(self.args.scheduler):
                                self.scheduler.step()
                            ema.update()

                            tr_loss += loss.item()
                            global_step += 1

                        print("lr : ", self.optimizer.param_groups[0]['lr'])
                        logger.info('-' * 50)
                        logger.info("lr : {}".format(self.optimizer.param_groups[0]['lr']))
                        logger.info("Train loss on epoch {} is {:.3f}".format(epoch, tr_loss / len(train_dataloader)))
                        # if epoch < 3:
                        #     continue

                        ema.apply_shadow()  # 应用ema
                        self.model.eval()
                        orders = ['subject', 'predicate', 'object']
                        correct_num, predicate_num, gold_num = 1e-10, 1e-10, 1e-10
                        import os
                        f = open(os.path.join(self.args.result_dir, 'CMeIE_dev.json'), 'w')
                        for example in tqdm(eval_examples, desc='Evaluating'):
                            text = example.text_a
                            gold_triples = [(triple['subject'], triple['predicate'], triple['object'])
                                            for triple in example.label]
                            pred_triples = extract_triples(self.model, text, self.args, tokenizer, id2relation,
                                                           self.args.test_max_seq_length)
                            gold = set(gold_triples)
                            pred = set(pred_triples)
                            gold_num += len(gold)
                            predicate_num += len(pred)
                            correct_num += len(pred & gold)

                            result = json.dumps({
                                'text': example.text_a,
                                'spo_list_gold': [
                                    dict(zip(orders, triple)) for triple in gold
                                ],
                                'spo_list_pred': [
                                    dict(zip(orders, triple)) for triple in pred
                                ],
                                'new': [
                                    dict(zip(orders, triple)) for triple in pred - gold
                                ],
                                'lack': [
                                    dict(zip(orders, triple)) for triple in gold - pred
                                ]
                            }, ensure_ascii=False, indent=4)
                            f.write(result + '\n')
                        logger.info(
                            "correct {:.0f} predict {:.0f} gold {:.0f} ".format(correct_num, predicate_num, gold_num))
                        precision = correct_num / predicate_num
                        recall = correct_num / gold_num
                        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

                        print("current F1 score: {:.3f},  best F1 score: {:.3f}".format(f1_score, best_eval_f1))
                        logger.info("current F1 score: {:.3f},  best F1 score: {:.3f}".format(f1_score, best_eval_f1))
                        if f1_score > best_eval_f1:
                            best_eval_f1 = f1_score
                            import os
                            save_file = os.path.join(self.args.output_dir, "pytorch_model.bin")
                            torch.save(self.model.state_dict(), save_file)
                            logger.info('*' * 50)
                            logger.info("Saved best F1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(
                                best_eval_f1, precision, recall))
                            patience = 0
                        else:
                            patience += 1
                            if patience >= self.args.patience:
                                break
                        ema.restore()  # 恢复原来的参数
        except KeyboardInterrupt:
            epochs.close()
            iters.close()
        epochs.close()
        iters.close()

    def eval_model(self, eval_examples, tokenizer, id2relation):
        import os
        save_file = os.path.join(self.args.output_dir, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(save_file))
        self.model.eval()

        orders = ['subject', 'predicate', 'object']
        correct_num, predicate_num, gold_num = 1e-10, 1e-10, 1e-10
        import os
        f = open(os.path.join(self.args.result_dir, 'CMeIE_dev.json'), 'w')
        for example in tqdm(eval_examples, desc='Evaluating'):
            text = example.text_a
            gold_triples = [(triple['subject'], triple['predicate'], triple['object'])
                            for triple in example.label]
            pred_triples = extract_triples(self.model, text, self.args, tokenizer, id2relation,
                                           self.args.test_max_seq_length)
            gold = set(gold_triples)
            pred = set(pred_triples)
            gold_num += len(gold)
            predicate_num += len(pred)
            correct_num += len(pred & gold)

            result = json.dumps({
                'text': example.text_a,
                'spo_list_gold': [
                    dict(zip(orders, triple)) for triple in gold
                ],
                'spo_list_pred': [
                    dict(zip(orders, triple)) for triple in pred
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in pred - gold
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in gold - pred
                ]
            }, ensure_ascii=False, indent=4)
            f.write(result + '\n')
        precision = correct_num / predicate_num
        recall = correct_num / gold_num
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        print(precision, recall, f1_score)

    def predicate(self, test_examples, tokenizer, id2relation):
        import os
        save_file = os.path.join(self.args.output_dir, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(save_file))
        self.model.eval()

        with open(os.path.join(self.args.result_dir, 'CMeIE_test.json'), 'w') as f:
            for example in tqdm(test_examples, desc="Predicating"):
                text = example.text_a
                pred_triples = extract_triples(self.model, text, self.args, tokenizer, id2relation,
                                               self.args.test_max_seq_length)
                pred_triples = list(set(pred_triples))
                line = {}
                line['text'] = text
                tmp = [{'subject': triple[0],
                        'predicate': triple[1],
                        'object': {'@value': triple[2]}} for triple in pred_triples]
                line['spo_list'] = tmp
                result = json.dumps(line, ensure_ascii=False)
                f.write(result + '\n')
