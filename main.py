import os
import json
import argparse
import logging
from src.model import CasRel
from src.process_data import *
from src.trainer import Trainer
from sklearn.model_selection import KFold
from src.utils.attack_train import *
from transformers import BertTokenizerFast

# 0903 禁用多线程 tokenization
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filemode='w',
                    filename='CMeIE.log',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    processor = RelationProcessor()
    pos2id = None  # TODO 没使用
    relation2id, id2relation = processor.get_relation2id(args.data_dir)
    args.num_relations = len(relation2id)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=args.do_lower_case)
    train_examples = processor.get_train_examples(args.data_dir)
    train_dataloader, _, train_num = get_dataloader(train_examples,
                                                    pos2id,
                                                    relation2id,
                                                    args.train_max_seq_length,
                                                    tokenizer,
                                                    args)
    num_train_steps = int(train_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    eval_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    logger.info("***** Data Summary *****")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num train steps = %d", num_train_steps)
    logger.info("  Num train examples = %d", len(train_examples))
    logger.info("  Num dev examples = %d", len(eval_examples))
    logger.info("  Num test examples = %d", len(test_examples))
    logger.info("  Num relations = %d", len(relation2id))

    model = CasRel(args.bert_path, args.num_relations, args.inner_code)

    trainer = Trainer(args, model, num_train_steps)
    if args.do_train:
        trainer.train_model(train_dataloader, eval_examples, tokenizer, id2relation)
    if args.do_eval:
        trainer.eval_model(eval_examples, tokenizer, id2relation)
    if args.do_test:
        trainer.predicate(test_examples, tokenizer, id2relation)


def stack():  # TODO 0903 ?
    processor = RelationProcessor()
    pos2id = None
    relation2id, id2relation = processor.get_relation2id(args.data_dir)
    args.num_relations = len(relation2id)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=args.do_lower_case)
    train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_dev_examples(args.data_dir)
    stack_examples = train_examples + eval_examples

    kf = KFold(8, shuffle=True, random_state=42)
    tmp_out_dir = args.output_dir
    for i, (train_ids, dev_ids) in enumerate(kf.split(stack_examples)):
        logger.info('*^'*80)
        logger.info(f'Start to train the {i} fold')

        args.output_dir = tmp_out_dir + f'_{i}'
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        args.output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

        train_examples = [stack_examples[_idx] for _idx in train_ids]
        eval_examples = [stack_examples[_idx] for _idx in dev_ids]
        test_examples = processor.get_test_examples(args.data_dir)

        train_dataloader, _, train_num = get_dataloader(train_examples,
                                                        pos2id,
                                                        relation2id,
                                                        args.train_max_seq_length,
                                                        tokenizer,
                                                        args)

        num_train_steps = int(
            train_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Data Summary *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num train steps = %d", num_train_steps)
        logger.info("  Num train examples = %d", len(train_examples))
        logger.info("  Num dev examples = %d", len(eval_examples))
        logger.info("  Num test examples = %d", len(test_examples))
        logger.info("  Num relations = %d", len(relation2id))

        model = CasRel(args.bert_path, args.num_relations, args.inner_code)
        trainer = Trainer(args, model, num_train_steps)
        if args.do_train:
            trainer.train_model(train_dataloader, eval_examples, tokenizer, id2relation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='./law_re', type=str)
    # parser.add_argument("--dataset", default='xx', type=str)
    parser.add_argument("--bert_path", default=r"D:\Code\huggingface\roberta-wwm-ext", type=str)
    parser.add_argument("--output_dir", default='./output/', type=str)
    parser.add_argument("--result_dir", default='./result/', type=str)

    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--do_test", action='store_true', default=True)
    parser.add_argument("--do_lower_case", action='store_false', default=True)

    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--num_train_epochs", default=40, type=float)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument('--inner_code', default='mean', type=str, choices=['lstm', 'mean'])
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'stack'])

    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument('--other_lr', default=1e-4, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--attack_train', default='pgd', type=str, choices=['', 'fgm', 'pgd'])
    parser.add_argument("--scheduler", default='', type=str, choices=['', 'linear', 'cyc'])

    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--h_bar', default=0.6, type=float)
    parser.add_argument('--t_bar', default=0.5, type=float)
    parser.add_argument("--train_max_seq_length", default=256, type=int)
    parser.add_argument("--test_max_seq_length", default=512, type=int)
    parser.add_argument('--patience', default=7, type=int)

    args = parser.parse_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # args.data_dir = args.data_dir + args.dataset + '/'

    # args.result_dir = os.path.join(args.result_dir, args.dataset)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    suffix =  args.inner_code
    if args.attack_train != '':
        suffix += f'_{args.attack_train}'
    if args.weight_decay:
        suffix += '_wd'
    if args.num_samples > 1:
        suffix += f'_{args.num_samples}sub'
    if args.scheduler != '':
        suffix += f'_{args.scheduler}'

    args.output_dir = os.path.join(args.output_dir, suffix)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == 'stack':
        stack()
    else:
        train()

