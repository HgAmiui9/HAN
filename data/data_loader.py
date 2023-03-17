

import numpy as np
import torch
from scipy import io as sio

import dgl
from dgl.data.utils import _get_dgl_url, download, get_download_dir


def download_data(data_name, url):
    path = get_download_dir() + data_name
    download(_get_dgl_url(url), path)
    return path


def load_armrow_data():
    data_name = '/ACM.mat'
    url = 'dataset/ACM.mat'
    data_path = download_data(data_name, url)

    data = sio.loadmat(data_path)

    p_vs_l = data["PvsL"]  # paper-field
    p_vs_a = data["PvsA"]  # paper-author
    p_vs_t = data["PvsT"]  # paper-term, bag of words
    p_vs_c = data["PvsC"]  # paper-conference, labels come from that

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

    features = torch.FloatTensor(p_vs_t.toarray())

    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): (p_vs_a.nonzero()[0], p_vs_a.nonzero()[1] + features.shape[0]),
            ("author", "ap", "paper"): (p_vs_a.transpose().nonzero()[0]+features.shape[0], p_vs_a.transpose().nonzero()[1]),
            ("paper", "pf", "field"): (p_vs_l.nonzero()[0], p_vs_l.nonzero()[1] + features.shape[0] + p_vs_a.shape[1]),
            ("field", "fp", "paper"): (p_vs_l.transpose().nonzero()[0]+features.shape[0]+p_vs_a.shape[1], p_vs_l.nonzero()[1]),
        }
    )

    # 将会议标签转化为三类标签
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3
    num_node_types = [p_vs_a.shape[0], p_vs_a.shape[1], p_vs_l.shapep[1]]

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = pc_c == conf_id
        float_mask[pc_c_mask] = np.random.permutation(
            np.linspace(0, 1, pc_c_mask.sum())
        )
    train_idx = np.where(float_mask <= 0.6)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.2)[0]
