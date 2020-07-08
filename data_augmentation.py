import random
import sys
import os
import unicodedata
import re
import logging
import csv
import argparse

import torch
import numpy as np

from transformer import BertTokenizer, BertForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

StopWordsList = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't", "'s", "'re"
]


# valid string only includes al
def _is_valid(string):
    return True if not re.search('[^a-z]', string) else False


def prepare_embedding_retrieval(glove_file, vocab_size=100000):
    cnt = 0
    words = []
    embeddings = {}

    # only read first 100,000 words for fast retrieval
    with open(glove_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split()
            words.append(items[0])    # 将词加入words
            embeddings[items[0]] = [float(x) for x in items[1:]]    # 将词向量加入embeddings

            cnt += 1
            if cnt == vocab_size:
                break

    vocab = {w: idx for idx, w in enumerate(words)}
    ids_to_tokens = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(embeddings[ids_to_tokens[0]])
    emb_matrix = np.zeros((vocab_size, vector_dim))
    for word, v in embeddings.items():
        if word == '<unk>':
            continue
        emb_matrix[vocab[word], :] = v

    # normalize each word vector
    d = (np.sum(emb_matrix**2, 1)**0.5)
    emb_norm = (emb_matrix.T / d).T
    return emb_norm, vocab, ids_to_tokens


class DataAugmentor(object):

    def __init__(self, model, tokenizer, emb_norm, vocab, ids_to_tokens, M, N, p):
        self.model = model
        self.tokenizer = tokenizer
        self.emb_norm = emb_norm
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.M = M
        self.N = N
        self.p = p

    def _word_distance(self, word):
        if word not in self.vocab.keys():
            return []
        word_idx = self.vocab[word]
        word_emb = self.emb_norm[word_idx]

        # 点乘计算向量距离
        dist = np.dot(self.emb_norm, word_emb.T)
        # 让本身的向量负无穷，这样待会排序就不会取到
        dist[word_idx] = -np.Inf
        # 从大到小排序
        candidate_ids = np.argsort(-dist)[:self.M]
        return [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

    def _masked_language_model(self, sent, word_pieces, mask_id):
        tokenized_text = self.tokenizer.tokenize(sent)
        tokenized_text = ['[CLS]'] + tokenized_text
        tokenized_len = len(tokenized_text)
        # 为什么要把本身句子拼接到后面？是因为bert预训练都是用的两句训练的吗？
        tokenized_text = word_pieces + ['[SEP]'] + tokenized_text[1:] + ['[SEP]']

        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:512]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) - tokenized_len - 1)

        tokens_tensor = torch.tensor([token_ids]).to(device)
        segments_tensor = torch.tensor([segments_ids]).to(device)

        self.model.to(device)

        predictions = self.model(tokens_tensor, segments_tensor)
        # 直接取概率前self.M的索引, predictions : bsz x len x vocab_size
        word_candidates = torch.argsort(predictions[0, mask_id], descending=True)[:self.M].tolist()
        word_candidates = self.tokenizer.convert_ids_to_tokens(word_candidates)
        # 过滤掉带有##的token
        return list(filter(lambda x: x.find("##"), word_candidates))

    def _word_augment(self, sentence, mask_token_idx, mask_token):
        word_pieces = self.tokenizer.tokenize(sentence)
        word_pieces = ['[CLS]'] + word_pieces
        tokenized_len = len(word_pieces)

        token_idx = -1
        # 0 is [CLS], so start from 1
        # mask_token_idx指的是完整词的索引，而经过tokenizer之后，就变成了没有##的token的索引。因为wordpiece的token是类似
        # 如 ['word', '##love', 'happy', '##ness']
        for i in range(1, tokenized_len):
            if "##" not in word_pieces[i]:
                token_idx = token_idx + 1
                if token_idx < mask_token_idx:
                    word_piece_ids = []
                elif token_idx == mask_token_idx:
                    word_piece_ids = [i]
                else:
                    break
            else:
                word_piece_ids.append(i)

        # 是完整单词的话，长度为1，进行mask
        if len(word_piece_ids) == 1:
            word_pieces[word_piece_ids[0]] = '[MASK]'
            candidate_words = self._masked_language_model(sentence, word_pieces, word_piece_ids[0])
        elif len(word_piece_ids) > 1:
            candidate_words = self._word_distance(mask_token)
        else:
            logger.info("invalid input sentence!")

        if len(candidate_words) == 0:
            candidate_words.append(mask_token)

        return candidate_words

    def augment(self, sent):
        candidate_sents = [sent]

        tokens = self.tokenizer.basic_tokenizer.tokenize(sent)
        candidate_words = {}
        for (idx, word) in enumerate(tokens):
            if _is_valid(word) and word not in StopWordsList:
                candidate_words[idx] = self._word_augment(sent, idx, word)
        logger.info(candidate_words)
        cnt = 0
        while cnt < self.N:
            new_sent = list(tokens)
            for idx in candidate_words.keys():
                candidate_word = random.choice(candidate_words[idx])

                x = random.random()
                if x < self.p:
                    new_sent[idx] = candidate_word

            if " ".join(new_sent) not in candidate_sents:
                candidate_sents.append(' '.join(new_sent))
            cnt += 1

        return candidate_sents


class AugmentProcessor(object):

    def __init__(self, augmentor, data_path):
        self.augmentor = augmentor
        self.data_path = data_path

    def read_augment_write(self):
        filename = f'aug_{os.path.split(self.data_path)[1]}'
        aug_train_path = os.path.join(os.path.split(self.data_path)[0], filename)

        with open(aug_train_path, 'w', newline='') as fw, open(self.data_path, 'r') as fr:
            for (i, line) in enumerate(fr):
                sent = line.strip('\n').strip()
                if not line:
                    continue
                augmented_sents = self.augmentor.augment(sent)
                for augment_sent in augmented_sents:
                    fw.write(f'{augment_sent}\n')
                if (i + 1) % 1000 == 0:
                    logger.info("Having been processing {} examples".format(str(i + 1)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_bert_model",
                        default=None,
                        type=str,
                        required=True,
                        help="Downloaded pretrained model (bert-base-uncased) is under this folder")
    parser.add_argument("--glove_embs",
                        default=None,
                        type=str,
                        required=True,
                        help="Glove word embeddings file")
    parser.add_argument("--data_path", default=None, type=str, required=True, help="GLUE data dir")
    parser.add_argument("--N", default=30, type=int, help="How many times is the corpus expanded?")
    parser.add_argument("--M",
                        default=15,
                        type=int,
                        help="Choose from M most-likely words in the corresponding position")
    parser.add_argument("--p",
                        default=0.4,
                        type=float,
                        help="Threshold probability p to replace current word")

    args = parser.parse_args()
    # logger.info(args)

    # Prepare data augmentor
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)
    model = BertForMaskedLM.from_pretrained(args.pretrained_bert_model)
    model.eval()
    emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(args.glove_embs)
    data_augmentor = DataAugmentor(model, tokenizer, emb_norm, vocab, ids_to_tokens, args.M, args.N,
                                   args.p)

    # Do data augmentation
    processor = AugmentProcessor(data_augmentor, args.data_path)
    processor.read_augment_write()


if __name__ == "__main__":
    main()
