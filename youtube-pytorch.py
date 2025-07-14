from tqdm import tqdm
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange
from tensorboardX import SummaryWriter
from pytorch_transformers import *  # WEIGHTS_NAME
from pytorch_transformers import AdamW, WarmupLinearSchedule
import json
import os
import sys
import numpy as np
import pandas as pd


embedding_size = 128
item_num = 50
learn_rate = 1e-4
top_k = 50
max_item_len = 4
max_user_len = 19
model_path = 'model/'

class youtube_recommender_model(nn.Module):
    def __init__(self, config, max_item_len, max_user_len, embedding_size, freeze_transformer=False, **args):
        super(youtube_recommender_model, self).__init__()
        self.config = config

        if freeze_transformer:
            for p in self.transformer_layer.parameters():
                p.requires_grad = False
        self.max_item_len = max_item_len
        self.max_user_len = max_user_len
        self.embedding_size = embedding_size
        self.input_embedding = embedding_size*2 + max_user_len      # len(items) + len(users)

        self.visit_txt_embed = nn.Embedding(max_item_len, embedding_size)
        self.visit_img_embed = nn.Embedding(max_item_len, embedding_size)
        self.visit_txt_embed.weight.data.copy_(torch.from_numpy(item_txt_embedding_matrix))
        self.visit_txt_embed.weight.requires_grad = False
        self.visit_img_embed.weight.data.copy_(torch.from_numpy(item_img_embedding_matrix))
        self.visit_img_embed.weight.requires_grad = False
        self.visit_embedd = torch.cat([self.visit_txt_embed, self.visit_img_embed],1)

        self.linear1 = nn.ReLU(nn.Linear(128, 1, bias=True))
        self.linear2 = nn.ReLU(nn.Linear(64, 1, bias=True))
        self.linear3 = nn.ReLU(nn.Linear(self.embedding_size*2, 1, bias=True))


    def forward(self, input_ids, labels=None, attention_mask=None, token_type_ids= None,
                      position_ids=None, head_mask=None):

        visit_txt_embedding = nn.Embedding.from_pretrained(embeddings=item_txt_embedding_matrix, freeze=True)
        visit_img_embedding = nn.Embedding.from_pretrained(embeddings=item_img_embedding_matrix, freeze=True)


        layer1 = self.linear1(input_embedding)
        layer2 = self.linear2(layer1)
        layer3 = self.linear3(layer2)

        logit = torch.matmul(layer3, output_embedding)
        yhat = torch.nn.Softmax(logit)

        return yhat


