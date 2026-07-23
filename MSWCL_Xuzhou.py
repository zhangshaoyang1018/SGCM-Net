# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.autograd import Variable
import numpy as np

import math
import argparse
import pickle
from sklearn import metrics
import time
import utils

import modelStatsRecord

from model.spa_branch import SpatialEncoder
from model.spe_branch import SpectralEncoder



use_gpu = torch.cuda.is_available()

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPT_BASENAME = os.path.basename(SCRIPT_DIR).lower()
if SCRIPT_BASENAME in ["ablation", "aaa_train_for_xuzhou"]:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    PROJECT_ROOT = SCRIPT_DIR

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "aaa_train_for_xuzhou")


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=128)
parser.add_argument("-c", "--src_input_dim", type=int, default=128)
parser.add_argument("-d", "--tar_input_dim", type=int, default=436)  # Xuzhou=436; Houston=144; SA=204; IP=200; UP=103
parser.add_argument("-n", "--n_dim", type=int, default=100)
parser.add_argument("-w", "--class_num", type=int, default=9)
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1)
parser.add_argument("-b", "--query_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=50)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)

# target
parser.add_argument("-m", "--test_class_num", type=int, default=9)
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help="5 4 3 2 1")

# test
parser.add_argument("-t", "--test_queries_num_per_class", type=int, default=5)

# 全局互信息域对齐 + 跨域类别锚点对比 + 目标域语义监督式对比学习
parser.add_argument("--da_warmup", type=int, default=500, help="先预热若干 episode，再加入全局域对齐")
parser.add_argument("--dda_weight", type=float, default=0.2, help="全局域对齐损失权重")
parser.add_argument("--lambda_loss", "--lambda_cross", dest="lambda_cross", type=float, default=0.5, help="内部跨域分布项权重")
parser.add_argument("--cross_warmup", type=int, default=500, help="CROSS 跨域对比")
parser.add_argument("--cca_weight", type=float, default=0.1, help="CROSS 跨域类别锚点对比损失权重")
parser.add_argument("--cross_temperature", type=float, default=0.1)
parser.add_argument("--scl_weight", type=float, default=1.5, help="目标域语义监督式对比学习损失权重")
parser.add_argument("--scl_temperature", type=float, default=0.1)
parser.add_argument("--scl_batch_size", type=int, default=64)
#TODO 读者们复现代码，需要先在自己的linux系统下放置一个预训练好的LLM模型，我这里是在hugging face下载的模型，你可以下载一样的模型，但是需要修改一下存放的路径
parser.add_argument("--text_encoder_model", type=str, default="/mnt/sda2/zsy/model/all-MiniLM-L6-v2")


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
TEST_CLASS_NUM = args.test_class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class
TEST_QUERIES_NUM_PER_CLASS = args.test_queries_num_per_class

DA_WARMUP = args.da_warmup
DDA_WEIGHT = args.dda_weight
LAMBDA_CROSS = args.lambda_cross
CROSS_WARMUP = args.cross_warmup
CCA_WEIGHT = args.cca_weight
CROSS_TEMPERATURE = args.cross_temperature
SCL_WEIGHT = args.scl_weight
SCL_TEMPERATURE = args.scl_temperature
SCL_BATCH_SIZE = args.scl_batch_size
TEXT_ENCODER_MODEL = args.text_encoder_model

RESULT_TXT_PATH = os.path.join(OUTPUT_DIR, f"SGCM-Net_{TEST_LSAMPLE_NUM_PER_CLASS}_shots.txt")
n_shot = TEST_LSAMPLE_NUM_PER_CLASS
n_ways = TEST_CLASS_NUM
n_queries = TEST_QUERIES_NUM_PER_CLASS
n_runs = 1
n_lsamples = n_ways * n_shot
n_usamples = n_ways * n_queries
n_samples = n_lsamples + n_usamples


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways


XUZHOU_CLASS_NAMES = [
    "a hyperspectral land-cover class of bare land 1",
    "a hyperspectral land-cover class of lakes",
    "a hyperspectral land-cover class of coals",
    "a hyperspectral land-cover class of cement",
    "a hyperspectral land-cover class of crops 1",
    "a hyperspectral land-cover class of trees",
    "a hyperspectral land-cover class of bare land 2",
    "a hyperspectral land-cover class of crops 2",
    "a hyperspectral land-cover class of red tiles",
]



class TextSemanticSimilarity:
    def __init__(self):
        self.device = device
        self.encoder = SentenceTransformer(TEXT_ENCODER_MODEL, device="cuda:0")

    @torch.no_grad()
    def build_similarity_matrix(self, class_names):
        emb = self.encoder.encode(
            class_names,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(self.device)
        sim = emb @ emb.t()
        return sim.clamp(-1.0, 1.0)


def _init_():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


_init_()

# =========================
# 加载源域Chikusei数据集
# =========================
with open(os.path.join(PROJECT_ROOT, "datasets", "Chikusei_imdb_128_7_7.pickle"), "rb") as handle:
    source_imdb = pickle.load(handle)

source_imdb["data"] = np.array(source_imdb["data"])
source_imdb["Labels"] = np.array(source_imdb["Labels"], dtype="int")
source_imdb["set"] = np.array(source_imdb["set"], dtype="int")
print(source_imdb.keys())
print(source_imdb["Labels"])

data_train = source_imdb["data"]
labels_train = source_imdb["Labels"]
print(data_train.shape)
print(labels_train.shape)

keys_all_train = sorted(list(set(labels_train)))
print(keys_all_train)

label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)

print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data)
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

print(np.array(source_imdb["data"]).shape)
source_imdb["data"] = source_imdb["data"].transpose((1, 2, 3, 0))
print(source_imdb["data"].shape)
print(source_imdb["Labels"])

del source_imdb

# =========================
# 加载目标域数据集
# =========================
last_dir = PROJECT_ROOT
test_data = last_dir + "/HSI_META_DATA/test/Xuzhou.mat"
test_label = last_dir + "/HSI_META_DATA/test/Xuzhou_gt.mat"

print("Xuzhou test_data:", test_data)
print("Xuzhou test_label:", test_label)
Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 3
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print("number of sample", nSample)

    train = {}
    test = {}
    da_train = {}
    m = int(np.max(G))
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print("labeled number per class:", nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print("the number of train_indices:", len(train_indices))
    print("the number of test_indices:", len(test_indices))
    print("the number of train_indices after data argumentation:", len(da_train_indices))
    print("labeled sample indices:", train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb["data"] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb["Labels"] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb["set"] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb["data"][:, :, :, iSample] = data[
            Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth + 1,
            Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1,
            :,
        ]
        imdb["Labels"][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb["Labels"] = imdb["Labels"] - 1
    imdb["set"] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print("Data is OK.")

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=class_num * shot_num_per_class,
        shuffle=False,
        num_workers=0,
    )
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    imdb_da_train = {}
    imdb_da_train["data"] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain], dtype=np.float32)
    imdb_da_train["Labels"] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train["set"] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        imdb_da_train["data"][:, :, :, iSample] = utils.radiation_noise(
            data[
                Row[da_RandPerm[iSample]] - HalfWidth: Row[da_RandPerm[iSample]] + HalfWidth + 1,
                Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1,
                :,
            ]
        )
        imdb_da_train["Labels"][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train["Labels"] = imdb_da_train["Labels"] - 1
    imdb_da_train["set"] = np.ones([da_nTrain]).astype(np.int64)
    print("ok")

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=class_num,
        shot_num_per_class=shot_num_per_class,
    )

    train_datas, train_labels = next(train_loader.__iter__())
    print("train labels:", train_labels)
    print("size of train datas:", train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train["data"].shape)
    print(imdb_da_train["Labels"])
    del Data_Band_Scaler, GroundTruth

    target_da_datas = np.transpose(imdb_da_train["data"], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train["Labels"]
    print("target data augmentation label:", target_da_labels)

    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    print(imdb_da_train["data"].shape)
    print(imdb_da_train["Labels"])

    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=SCL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain


# =========================
# Model
# =========================
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.spatial_encoder = SpatialEncoder()
        self.spectral_encoder = SpectralEncoder(in_channels=100)

    def encode_single(self, x, domain="source"):
        if domain == "target":
            x = self.target_mapping(x)
        elif domain == "source":
            x = self.source_mapping(x)
        else:
            raise ValueError("domain must be 'source' or 'target'")

        x_spa = self.spatial_encoder(x)
        x_spe = self.spectral_encoder(x)
        return 0.5 * (x_spa + x_spe)

    def forward(self, spt, qry, domain="source"):
        spt_mid = self.encode_single(spt, domain=domain)
        qry_mid = self.encode_single(qry, domain=domain)
        return spt_mid, qry_mid


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.ones_(m.bias)


crossEntropy = nn.CrossEntropyLoss().cuda()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def l2norm(x, dim=1, eps=1e-12):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)



#这个是计算信息熵的函数
def entropy_from_prob(prob, eps=1e-8):
    prob = prob.clamp(min=eps)
    return -(prob * torch.log(prob)).sum(dim=1).mean()

# 这个函数用来计算单个域内的MI信息 大点好
def mutual_information_objective(features, eps=1e-8):
    # 使用 batch 内特征维度分布估计互信息目标：I = H(mean p) - mean H(p)
    prob = F.softmax(features, dim=1)
    p_mean = prob.mean(dim=0, keepdim=True)
    h_mean = entropy_from_prob(p_mean, eps=eps)
    h_cond = entropy_from_prob(prob, eps=eps)
    return h_mean - h_cond


def mim_domain_alignment_loss(source_features, target_features, lambda_cross=0.5, eps=1e-8):
    i_src = mutual_information_objective(source_features, eps=eps)
    i_tar = mutual_information_objective(target_features, eps=eps)

    p_src = F.softmax(source_features, dim=1).mean(dim=0)
    p_tar = F.softmax(target_features, dim=1).mean(dim=0)
    i_cross = F.kl_div((p_src + eps).log(), p_tar.detach(), reduction="sum") + \
              F.kl_div((p_tar + eps).log(), p_src.detach(), reduction="sum")

    loss_mim = -1.0 * (i_src + i_tar) + lambda_cross * i_cross
    logs = {
        "I_src": i_src.detach(),
        "I_tar": i_tar.detach(),
        "I_cross": i_cross.detach(),
        "L_mim": loss_mim.detach(),
    }
    return loss_mim, logs


def compute_episode_anchors(features, labels, num_classes):
    labels = labels.long().to(features.device)
    centroids = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            centroids.append(torch.zeros(features.size(1), device=features.device, dtype=features.dtype))
        else:
            centroids.append(features[mask].mean(dim=0))
    return torch.stack(centroids, dim=0)

# 用当前 episode 的源域/目标域类别原型做跨域匹配与对齐
def cross_domain_class_anchor_contrast_loss(
    source_features,
    source_labels,
    target_features,
    target_labels,
    source_class_num,
    target_class_num,
    temperature=0.1,
    eps=1e-8,
):

    c_s = compute_episode_anchors(source_features, source_labels, source_class_num)
    c_t = compute_episode_anchors(target_features, target_labels, target_class_num)

    c_s_norm = F.normalize(c_s, p=2, dim=1, eps=eps)
    c_t_norm = F.normalize(c_t, p=2, dim=1, eps=eps)

    sim_st = torch.matmul(c_s_norm, c_t_norm.t())
    matched_src_for_tar = sim_st.argmax(dim=0)
    sim_tt = torch.matmul(c_t_norm, c_t_norm.t())

    loss_list = []
    for t in range(target_class_num):
        pos_s = matched_src_for_tar[t]
        pos_logit = sim_st[pos_s, t] / temperature

        src_logits = sim_st[:, t] / temperature

        tar_neg_mask = torch.ones(target_class_num, dtype=torch.bool, device=c_t.device)
        tar_neg_mask[t] = False
        tar_logits = sim_tt[tar_neg_mask, t] / temperature

        denom_logits = torch.cat([src_logits, tar_logits], dim=0)
        loss_t = -pos_logit + torch.logsumexp(denom_logits, dim=0)
        loss_list.append(loss_t)

    loss_cross_cl = torch.stack(loss_list).mean()
    logs = {
        "L_cross_cl": loss_cross_cl.detach(),
        "mean_match_sim": sim_st[
            matched_src_for_tar,
            torch.arange(target_class_num, device=sim_st.device),
        ].mean().detach(),
    }
    return loss_cross_cl, logs




# =========================
# 目标域文本语义负样本加权 SupCon
# =========================
class SemanticNegativeWeightedSupConLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super(SemanticNegativeWeightedSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, semantic_matrix=None):
        device = features.device
        if features.dim() < 3:
            raise ValueError("features must be [B, n_views, D]")
        if features.dim() > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1).long().to(device)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")

        pos_mask_base = torch.eq(labels, labels.T).float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = pos_mask_base.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * anchor_count, device=device).view(-1, 1), 0)
        mask = mask * logits_mask

        if semantic_matrix is not None:
            semantic_matrix = semantic_matrix.detach().float().to(device)
            label_vec = labels.view(-1)
            sem_pair = semantic_matrix[label_vec][:, label_vec].clamp(-1.0, 1.0)
            sem_pair = (sem_pair + 1.0) * 0.5
            neg_mask_base = 1.0 - pos_mask_base
            pair_weight_base = pos_mask_base + neg_mask_base * (1.0 + sem_pair)
            pair_weight = pair_weight_base.repeat(anchor_count, contrast_count)
        else:
            pair_weight = torch.ones_like(mask)

        exp_logits = torch.exp(logits) * logits_mask * pair_weight
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        pos_count = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (pos_count + 1e-12)
        loss_vec = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        valid_anchor = (pos_count > 0).float()
        loss = (loss_vec * valid_anchor).sum() / (valid_anchor.sum() + 1e-12)
        return loss


def save_classification_map(predict, G, RandPerm, Row, Column, nTrain, save_path):
    import matplotlib.pyplot as plt

    if predict is None or len(predict) == 0:
        print("No best prediction available; skip classification map.")
        return

    pred_map = np.zeros_like(G, dtype=np.int64)
    for i, pred_label in enumerate(predict):
        if nTrain + i >= len(RandPerm):
            break
        sample_index = RandPerm[nTrain + i]
        pred_map[Row[sample_index], Column[sample_index]] = int(pred_label) + 1

    if pred_map.shape[0] > 6 and pred_map.shape[1] > 6:
        pred_map = pred_map[3:-3, 3:-3]

    plt.figure(figsize=(8, 6))
    plt.imshow(pred_map, cmap="tab20")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("Best classification map saved to:", save_path)


# =========================
# 构建目标域文本语义相似度矩阵
# =========================
text_sim_builder = TextSemanticSimilarity()
target_text_sim_matrix = text_sim_builder.build_similarity_matrix(XUZHOU_CLASS_NAMES).to(device)
print(target_text_sim_matrix.detach().cpu().numpy())


# =========================
# Run 10 times
# =========================
nDataSet = 2
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_iDataset_all = -1
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
best_state_dict_all = None

seeds = [1337, 1220, 1336, 1330, 1224, 1236, 1226, 1235, 1233, 1229]

for iDataSet in range(nDataSet):
    np.random.seed(seeds[iDataSet])
    utils.same_seeds(seeds[iDataSet])

    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS,
    )

    feature_encoder = Network()
    print(get_parameter_number(feature_encoder))
    feature_encoder.apply(weights_init)
    feature_encoder.cuda()
    feature_encoder.train()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    supcon_criterion = SemanticNegativeWeightedSupConLoss(temperature=SCL_TEMPERATURE).cuda()

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    best_predict_dataset = None
    best_model_state_dataset = None
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    target_iter = iter(target_loader)
    train_start = time.time()

    for episode in range(EPISODE):
        try:
            target_data, target_label = next(target_iter)
        except Exception:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)

        # ---- 源域 few-shot ----
        task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader = utils.get_HBKC_data_loader(
            task,
            num_per_class=SHOT_NUM_PER_CLASS,
            split="train",
            shuffle=False,
        )
        query_dataloader = utils.get_HBKC_data_loader(
            task,
            num_per_class=QUERY_NUM_PER_CLASS,
            split="test",
            shuffle=True,
        )
        supports, support_labels = next(support_dataloader.__iter__())
        querys, query_labels = next(query_dataloader.__iter__())
        support_features_s, query_features_s = feature_encoder(supports.cuda(), querys.cuda(), domain="source")

        if SHOT_NUM_PER_CLASS > 1:
            support_proto_s = support_features_s.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
        else:
            support_proto_s = support_features_s

        logits_s = euclidean_metric(query_features_s, support_proto_s)
        f_loss_s = crossEntropy(logits_s, query_labels.long().cuda())
        loss_s = f_loss_s

        # ---- 目标域 few-shot ----
        task_t = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_t = utils.get_HBKC_data_loader(
            task_t,
            num_per_class=SHOT_NUM_PER_CLASS,
            split="train",
            shuffle=False,
        )
        query_dataloader_t = utils.get_HBKC_data_loader(
            task_t,
            num_per_class=QUERY_NUM_PER_CLASS,
            split="test",
            shuffle=True,
        )
        supports_t, support_labels_t = next(support_dataloader_t.__iter__())
        querys_t, query_labels_t = next(query_dataloader_t.__iter__())
        support_features_t, query_features_t = feature_encoder(supports_t.cuda(), querys_t.cuda(), domain="target")

        if SHOT_NUM_PER_CLASS > 1:
            support_proto_t = support_features_t.reshape(TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
        else:
            support_proto_t = support_features_t

        logits_t = euclidean_metric(query_features_t, support_proto_t)
        f_loss_t = crossEntropy(logits_t, query_labels_t.long().cuda())
        loss_t = f_loss_t

        # FSL: 源域 episode CE + 目标域 episode CE
        f_loss = loss_s + loss_t

        # MIM + CROSS + SCL: 保留全局互信息域对齐、跨域类别锚点对比和目标域语义监督式对比学习
        zero_da = torch.tensor(0.0, device=support_features_s.device)
        loss_mim = zero_da
        mim_logs = {
            "I_src": zero_da.detach(),
            "I_tar": zero_da.detach(),
            "I_cross": zero_da.detach(),
            "L_mim": zero_da.detach(),
        }
        loss_cross_cl = zero_da
        cross_logs = {
            "L_cross_cl": zero_da.detach(),
            "mean_match_sim": zero_da.detach(),
        }

        feat_s_all = torch.cat([support_features_s, query_features_s], dim=0)
        feat_t_all = torch.cat([support_features_t, query_features_t], dim=0)

        if (episode + 1) > DA_WARMUP:
            loss_mim, mim_logs = mim_domain_alignment_loss(
                feat_s_all,
                feat_t_all,
                lambda_cross=LAMBDA_CROSS,
            )

        if (episode + 1) > CROSS_WARMUP:
            label_s_all = torch.cat([support_labels.long().cuda(), query_labels.long().cuda()], dim=0)
            label_t_all = torch.cat([support_labels_t.long().cuda(), query_labels_t.long().cuda()], dim=0)
            loss_cross_cl, cross_logs = cross_domain_class_anchor_contrast_loss(
                source_features=feat_s_all,
                source_labels=label_s_all,
                target_features=feat_t_all,
                target_labels=label_t_all,
                source_class_num=CLASS_NUM,
                target_class_num=TEST_CLASS_NUM,
                temperature=CROSS_TEMPERATURE,
            )

        target_data = target_data.cuda().float()
        target_label = target_label.long().cuda()
        target_feature = feature_encoder.encode_single(target_data, domain="target")
        target_feature = F.normalize(target_feature, dim=1)
        supcon_feature = target_feature.unsqueeze(1)
        scl_loss_tar = supcon_criterion(
            supcon_feature,
            target_label,
            semantic_matrix=target_text_sim_matrix,
        )

        loss = f_loss + DDA_WEIGHT * loss_mim + CCA_WEIGHT * loss_cross_cl + SCL_WEIGHT * scl_loss_tar

        feature_encoder.zero_grad()
        loss.backward()
        feature_encoder_optim.step()

        total_hit += torch.sum(torch.argmax(logits_s, dim=1).cpu() == query_labels).item()
        total_hit += torch.sum(torch.argmax(logits_t, dim=1).cpu() == query_labels_t).item()
        total_num += querys.shape[0] + querys_t.shape[0]

        if (episode + 1) % 100 == 0:
            elapsed_time = time.time() - train_start
            train_loss.append(loss.item())
            print(
                "episode {:>3d}: f_loss: {:6.4f}, L_dda: {:6.4f}, L_cca: {:6.4f}, L_scl: {:6.4f}, total_loss: {:6.4f}, I_src: {:6.4f}, I_tar: {:6.4f}, I_cross: {:6.4f}, match_sim: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}, elapsed time: {:6.4f}".format(
                    episode + 1,
                    f_loss.item(),
                    loss_mim.item(),
                    loss_cross_cl.item(),
                    scl_loss_tar.item(),
                    loss.item(),
                    mim_logs["I_src"].item(),
                    mim_logs["I_tar"].item(),
                    mim_logs["I_cross"].item(),
                    cross_logs["mean_match_sim"].item(),
                    querys.shape[0],
                    total_hit / total_num,
                    elapsed_time,
                )
            )

        if (episode + 1) % 500 == 0 or episode == 0:
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            labels_al = np.array([], dtype=np.int64)

            train_datas, train_labels = next(train_loader.__iter__())
            test_datas, _ = next(test_loader.__iter__())
            train_features, _ = feature_encoder(
                Variable(train_datas).cuda(),
                Variable(test_datas).cuda(),
                domain="target",
            )

            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value + 1e-12)

            support_proto = train_features.reshape(-1, TEST_LSAMPLE_NUM_PER_CLASS, FEATURE_DIM).permute(1, 0, 2)
            support_proto = support_proto.mean(0)
            support_proto = support_proto / (support_proto.norm(dim=1, keepdim=True) + 1e-12)

            logits_pseudo = np.zeros((1, TEST_CLASS_NUM))

            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]
                _, test_features = feature_encoder(
                    Variable(train_datas).cuda(),
                    Variable(test_datas).cuda(),
                    domain="target",
                )
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value + 1e-12)
                logits = euclidean_metric(test_features, support_proto)

                predict_labels = torch.argmax(logits.detach(), dim=1).cpu()
                test_labels = test_labels.numpy()
                labels = np.append(labels, predict_labels)
                labels_al = np.append(labels_al, test_labels)
                logits_pseudo = np.append(logits_pseudo, logits.detach().cpu(), axis=0)
                counter += batch_size

            logits_pseudo = torch.from_numpy(logits_pseudo[1:, :])
            predict_labels = torch.argmax(logits_pseudo, dim=1).cpu()
            print("shape is", predict_labels.shape)

            rewards = [1 if predict_labels[j] == labels_al[j] else 0 for j in range(counter)]
            total_rewards = np.sum(rewards)
            predict = predict_labels
            accuracy = total_rewards / 1.0 / counter
            accuracies.append(accuracy)
            print(accuracy)

            test_accuracy = 100.0 * total_rewards / len(test_loader.dataset)
            print(
                "\t\tAccuracy: {}/{} ({:.2f}%)\n".format(
                    total_rewards,
                    len(test_loader.dataset),
                    100.0 * total_rewards / len(test_loader.dataset),
                )
            )
            test_end = time.time()

            feature_encoder.train()
            if test_accuracy > last_accuracy:
                print("update best result for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = total_rewards / len(test_loader.dataset)
                OA = acc[iDataSet]
                C = metrics.confusion_matrix(labels_al, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                k[iDataSet] = metrics.cohen_kappa_score(labels_al, predict)
                best_predict_dataset = predict.detach().cpu().numpy().astype(np.int64).copy()
                best_model_state_dataset = {k: v.detach().cpu().clone() for k, v in feature_encoder.state_dict().items()}

            print("best episode:[{}], best accuracy={}".format(best_episdoe + 1, last_accuracy))

    training_time[iDataSet] = train_end - train_start
    test_time[iDataSet] = test_end - train_end

    if last_accuracy > best_acc_all and best_predict_dataset is not None:
        best_acc_all = last_accuracy
        best_iDataset_all = iDataSet
        best_predict_all = best_predict_dataset.copy()
        best_G = G.copy()
        best_RandPerm = RandPerm.copy()
        best_Row = Row.copy()
        best_Column = Column.copy()
        best_nTrain = nTrain
        best_state_dict_all = best_model_state_dataset

    print("iter:{} best episode:[{}], best accuracy={}".format(iDataSet, best_episdoe + 1, last_accuracy))
    print("***********************************************************************************")


ELEMENT_ACC_RES_SS4 = np.transpose(A)
AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4, 0)
OA_RES_SS4 = np.transpose(acc)
KAPPA_RES_SS4 = np.transpose(k)
ELEMENT_PRE_RES_SS4 = np.transpose(P)
AP_RES_SS4 = np.mean(ELEMENT_PRE_RES_SS4, 0)
TRAINING_TIME_RES_SS4 = np.transpose(training_time)
TESTING_TIME_RES_SS4 = np.transpose(test_time)
classes_num = TEST_CLASS_NUM
ITER = nDataSet

meta = {
    "dataset": "Xuzhou",
    "shot": TEST_LSAMPLE_NUM_PER_CLASS,
    "way": TEST_CLASS_NUM,
    "seeds": seeds,
    "feature_dim": FEATURE_DIM,
    "text_encoder_model": TEXT_ENCODER_MODEL,
}




cur_time_stamp = str(int(time.time()))
modelStatsRecord.outputRecord_noP(
    ELEMENT_ACC=ELEMENT_ACC_RES_SS4,
    AA=AA_RES_SS4,
    OA=OA_RES_SS4,
    KAPPA=KAPPA_RES_SS4,
    TRAIN_TIME=TRAINING_TIME_RES_SS4,
    TEST_TIME=TESTING_TIME_RES_SS4,
    CATEGORY=TEST_CLASS_NUM,
    ITER=nDataSet,
    path1=RESULT_TXT_PATH,
    meta=meta,
    class_names=XUZHOU_CLASS_NAMES,
)

