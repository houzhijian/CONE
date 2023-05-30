from __future__ import absolute_import, division, print_function
"""
Modified from https://github.com/jayleicn/TVRetrieval/blob/master/utils/text_feature/lm_finetuning_on_single_sentences.py
"""
"""
This script has been verified in the following framework
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers
$ git checkout e1b2949ae6cb34cc39e3934ca87423474f8c8d02
$ pip install .

References:
    https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
"""
import os
import argparse
import logging
import random
from easydict import EasyDict as edict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

import io

from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from utils.basic_utils import load_jsonl, flat_list_of_lists, save_jsonl, save_json

logger = logging.getLogger(__name__)

# only tested with roberta
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


class SingleSentenceDataset(Dataset):
    def __init__(self, tokenizer, input_datalist, block_size=512, add_extra_keys=False, debug=False):
        self.tokenizer = tokenizer
        self.max_length = block_size
        self.debug = debug
        self.debug_cnt = 100  # should be large than batch size
        self.add_extra_keys = add_extra_keys
        self.examples = self.read_examples(input_datalist, add_extra_keys)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def process_question(self, question):
        """Process the question to make it canonical."""
        return question.strip(".").strip(" ").strip("?").lower() + "?"

    def read_examples(self, input_datalist, add_extra_keys=False):
        """input_datalist, list(dict), each dict is,
        {"id": *str_id*,
        "text": raw consecutive text (e.g., a sentence or a paragraph) not processed in any way,}
        add_extra_keys: some additional keys that exist in the dict element of input_datalist,
            for example, "sen_lengths" for subtitles
        """
        examples = []
        for idx, line in tqdm(enumerate(input_datalist), desc="Loading data", total=len(input_datalist)):
            # print("before postprocessing: ",line["text"])
            line["text"] =  self.process_question(line["text"])
            # print("after postprocessing: ", line["text"])
            text_ids = self.single_segment_processor(line["text"], max_length=self.max_length)
            # print("text_ids: ", text_ids)
            line_example = edict(id=line["id"],
                                 text=line["text"],
                                 text_ids=text_ids)
            if add_extra_keys:
                for ek in line.keys():
                    if ek not in line_example:
                        line_example[ek] = line[ek]
            examples.append(line_example)
        return examples

    def single_segment_processor(self, single_segment_text, max_length):
        """
        single_segment_text: str, raw consecutive text (e.g., a sentence or a paragraph) not processed in any way
        Processing Steps:
        1) tokenize
        2) add special tokens
        # 3) pad to max length
        max_length: int, segment longer than max_length will be truncated
        """
        single_segment_ids = self.tokenizer.encode(single_segment_text,
                                                   add_special_tokens=True,
                                                   max_length=max_length)
        return single_segment_ids


def get_batch_token_embeddings(layer_hidden_states, attention_mask, rm_special_tokens=False):
    """ remove padding and special tokens
    Args:
        layer_hidden_states: (N, L, D)
        attention_mask: (N, L) with 1 indicate valid bits, 0 pad bits
        rm_special_tokens: bool, whether to remove special tokens, this is different for different model_type
            1) a RoBERTa sequence has the following format: <s> X </s>
    return:
        list(np.ndarray), each ndarray is (L_sentence, D), where L_sentence <= L
    """
    valid_lengths = attention_mask.sum(1).long().tolist()  # (N, )
    layer_hidden_states = layer_hidden_states.cpu().numpy()
    embeddings = [e[1:vl - 1] if rm_special_tokens else e[:vl]
                  for e, vl in zip(layer_hidden_states, valid_lengths)]
    return embeddings


def load_preprocess_tvr_query(tvr_file_path):
    return [dict(id=e["query_id"], text=e["query"]) for e in load_jsonl(tvr_file_path)]


def load_and_cache_examples(tokenizer, data_path, max_length
                            , add_extra_keys=False, debug=False):
    input_datalist = flat_list_of_lists([load_preprocess_tvr_query(e) for e in data_path])
    dataset = SingleSentenceDataset(tokenizer,
                                    input_datalist,
                                    block_size=max_length,
                                    add_extra_keys=add_extra_keys,
                                    debug=debug)
    return dataset


def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_collate(data):
    batch = edict()
    batch["text_ids"], batch["text_ids_mask"] = pad_sequences_1d([d["text_ids"] for d in data], dtype=torch.long)
    for k in data[0].keys():
        if k not in ["text_ids"]:
            batch[k] = [d[k] for d in data]
    batch["unique_id"] = [d["id"] for d in data]
    # batch["sen_lengths"] = [d["sen_lengths"] for d in data]
    return batch


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def extract(args, model, tokenizer, prefix=""):
    """Many of the extraction args are inherited from evaluation"""
    # extract_output_dir = args.output_dir

    extract_dataset = load_and_cache_examples(tokenizer, args.train_data_file, args.block_size,
                                              add_extra_keys=False,
                                              debug=args.debug)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    extract_sampler = SequentialSampler(extract_dataset) \
        if args.local_rank == -1 else DistributedSampler(extract_dataset)
    extract_dataloader = DataLoader(extract_dataset,
                                    sampler=extract_sampler,
                                    batch_size=args.eval_batch_size,
                                    collate_fn=pad_collate)

    # Eval!
    logger.info("***** Running extraction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(extract_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    caption_token_length_dict = dict()

    for idx, batch in tqdm(enumerate(extract_dataloader), desc="Extracting", total=len(extract_dataloader)):
        text_ids = batch.text_ids.to(args.device)
        text_ids_mask = batch.text_ids_mask.to(args.device)

        with torch.no_grad():
            outputs = model.roberta(text_ids, attention_mask=text_ids_mask)

            all_layer_hidden_states = outputs.hidden_states

            extracted_hidden_states = get_batch_token_embeddings(
                all_layer_hidden_states[-2], text_ids_mask, rm_special_tokens=True)
            if args.debug:
                logger.info("outputs {}, all_layer_hiddens {}, -2 {}"
                            .format(len(outputs), len(all_layer_hidden_states), all_layer_hidden_states[-2].shape))
                logger.info("last_to_second_layer_hidden_states {}"
                            .format([e.shape for e in extracted_hidden_states]))
        if args.debug and idx > 10:
            break

        for e_idx, (unique_id, text_feat) in enumerate(zip(batch.id, extracted_hidden_states)):
            # print("batch.text: ", batch.text[e_idx])
            # print("batch.text_ids: ",batch.text_ids[e_idx])
            # print("text_feat: ", text_feat.shape)
            assert text_feat.shape[0] == batch.text_ids_mask[e_idx].sum(0) - 2

            np.save(os.path.join(args.token_fea_dir, unique_id),text_feat)

    #save_json(caption_token_length_dict, args.lmdb_file + ".length.json")


def tokenize(args, tokenizer, prefix=""):
    """Many of the extraction args are inherited from evaluation"""
    # extract_output_dir = args.output_dir

    extract_dataset = load_and_cache_examples(tokenizer, args.train_data_file, args.block_size,
                                              add_extra_keys=False,
                                              debug=args.debug)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, nargs="+",
                        help="The input training data file (a text file)"
                             "When --do_extract and not --use_sub, extract feature from this file(s).")
    parser.add_argument("--token_fea_dir", default=None, type=str,
                        help=".lmdb file name to save the extracted file")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 "
                             "(instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")

    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--do_extract", action='store_true',
                        help="Whether to run extract")
    parser.add_argument("--do_tokenize", action="store_true",
                        help="Extract tokenized text. Typically used alone without loading models")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--debug', action='store_true', help="break all loops")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    if args.do_extract:
        config.output_hidden_states = True  # output hidden states from all layers
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation/Extraction parameters %s", args)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    model.to(args.device)
    if not os.path.exists(args.token_fea_dir):
        os.mkdir(args.token_fea_dir)
    # tokenize
    if args.do_tokenize:
        tokenize(args, tokenizer)

    # Extraction
    if args.do_extract and args.local_rank in [-1, 0]:
        extract(args, model, tokenizer)


if __name__ == "__main__":
    main()
