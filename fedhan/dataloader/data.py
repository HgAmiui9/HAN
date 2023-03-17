import datetime
import errno
import os
import pickle
import random
from pprint import pprint

import numpy as np
import torch
from scipy import io as sio
from scipy import sparse

import dgl
from dgl.data.utils import _get_dgl_url, download, get_download_dir

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = "dataset/ACM.mat"
    data_path = get_download_dir() + "/ACM.mat"
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data["PvsL"]  # paper-field
    p_vs_a = data["PvsA"]  # paper-author
    p_vs_t = data["PvsT"]  # paper-term, bag of words
    p_vs_c = data["PvsC"]  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    # 这里三种类型的结点都在同一张图
    # 论文ID = 原论文ID
    # 作者ID = 原作者ID + 论文数
    # 领域ID = 原领域ID + 论文数 + 作者数
    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): (p_vs_a.nonzero()[0], p_vs_a.nonzero()[1] + features.shape[0]),
            ("author", "ap", "paper"): (p_vs_a.transpose().nonzero()[0]+features.shape[0], p_vs_a.transpose().nonzero()[1]),
            ("paper", "pf", "field"): (p_vs_l.nonzero()[0], p_vs_l.nonzero()[1] + features.shape[0] + p_vs_a.shape[1]),
            ("field", "fp", "paper"): (p_vs_l.transpose().nonzero()[0]+features.shape[0]+p_vs_a.shape[1], p_vs_l.nonzero()[1]),
        }
    )

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = pc_c == conf_id
        float_mask[pc_c_mask] = np.random.permutation(
            np.linspace(0, 1, pc_c_mask.sum())
        )
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes("paper")
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return (
        hg,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )
