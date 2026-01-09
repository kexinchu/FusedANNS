from this import d
import numpy as np
import time
import struct
import json
import re
import os
from pathlib import Path

def to_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', float(x))
                fp.write(a)

def to_ivecs(filename, data):
    # count = 0
    with open(filename, 'wb') as fp:
        for y in data:
            # count += 1
            # if count > 100:
            #     break
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('I', int(x))
                fp.write(a)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def ivecs_read_var(fname):
    data = []
    with open(fname, 'rb') as fp:
        tail = fp.seek(0, 2)
        fp.seek(0)
        while fp.tell() != tail:
            cur = fp.read(4)
            d, = struct.unpack('I', cur)
            tmp = []
            for i in range(d):
                cur = fp.read(4)
                id, = struct.unpack('I', cur)
                tmp.append(id)
            data.append(tmp)
    return data

def fvecs_read_var(fname):
    data = []
    with open(fname, 'rb') as fp:
        tail = fp.seek(0, 2)
        fp.seek(0)
        while fp.tell() != tail:
            cur = fp.read(4)
            d, = struct.unpack('I', cur)
            tmp = []
            for i in range(d):
                cur = fp.read(4)
                id, = struct.unpack('f', cur)
                tmp.append(id)
            data.append(tmp)
    return data

def fbin_read(fname):
    data = []
    with open(fname, 'rb') as fp:
        cur = fp.read(4)
        n, = struct.unpack('I', cur)
        cur = fp.read(4)
        d, = struct.unpack('I', cur)
        for i in range(n):
            tmp = []
            for i in range(d):
                cur = fp.read(4)
                id, = struct.unpack('f', cur)
                tmp.append(id)
            data.append(tmp)
    return data

def fvecs_read(filename):
    return ivecs_read(filename).view('float32')

def fvecs_read_head(filename, count):
    if count <= 0:
        return np.empty((0, 0), dtype='float32')
    with open(filename, 'rb') as fp:
        dim_bytes = fp.read(4)
        if not dim_bytes:
            return np.empty((0, 0), dtype='float32')
        dim, = struct.unpack('I', dim_bytes)
    vec_size = 1 + dim
    total = count * vec_size
    a = np.fromfile(filename, dtype='int32', count=total)
    if a.size == 0:
        return np.empty((0, dim), dtype='float32')
    return a.reshape(-1, vec_size)[:, 1:].copy().view('float32')

def txt_read(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    meta_info = re.split('[ \n]', lines[0])
    lines.pop(0)
    print(meta_info)
    data = []
    for line in lines:
        tmp = re.split('[ \n]', line)
        tmp.pop()
        tmp = list(map(float, tmp))
        data.append(tmp)

    return data

def to_txt(filename, data):
    f = open(filename, 'w')
    num = len(data)
    dim = len(data[0])
    f.write(str(num) + ' ' + str(dim) + '\n')
    for i in range(num):
        for j in range(dim):
            if j != (dim - 1):
                f.write(str(data[i][j]) + ' ')
            else:
                f.write(str(data[i][j]) + '\n')

def load_vecs(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".fvecs":
        data = fvecs_read_var(filename)
    elif ext == ".ivecs":
        data = ivecs_read_var(filename)
    elif ext == ".fbin":
        data = fbin_read(filename)
    else:
        raise TypeError("Unknown file type: {}".format(ext))

    return data

def _load_uqv_modal2(uqv_root):
    env_base = os.environ.get("UQV_MODAL2_BASE")
    env_query = os.environ.get("UQV_MODAL2_QUERY")
    if env_base and env_query:
        base_path = Path(env_base)
        query_path = Path(env_query)
    else:
        base_path = None
        query_path = None
        for ext in (".ivecs", ".fvecs"):
            candidate_base = uqv_root / f"uqv_modal2_base{ext}"
            candidate_query = uqv_root / f"uqv_modal2_query{ext}"
            if candidate_base.exists() and candidate_query.exists():
                base_path = candidate_base
                query_path = candidate_query
                break

    if not base_path or not query_path:
        return None, None
    if not base_path.exists() or not query_path.exists():
        return None, None

    if base_path.suffix == ".ivecs":
        return ivecs_read(base_path), ivecs_read(query_path)
    if base_path.suffix == ".fvecs":
        return fvecs_read(base_path), fvecs_read(query_path)
    return None, None

def DatasetLoader(dataset):
    """
    Load dataset vectors. Set DATASET_ROOT env var to override the default
    (which points to ../indexing_and_search/doc/dataset).
    """
    default_root = Path(__file__).resolve().parents[2] / "indexing_and_search" / "doc" / "dataset"
    dataset_root = Path(os.environ.get("DATASET_ROOT", default_root)).expanduser()
    dataset_suffix = "resnet_encode"
    is_composition = 0
    type = "/train/"

    # Initialize in case a branch does not set all.
    base_modal1 = base_modal2 = base_modal3 = base_modal4 = []
    query_modal1 = query_modal2 = query_modal3 = query_modal4 = []
    gt = []

    if dataset == 'siftsmall':
        base_modal1 = fvecs_read(dataset_root / dataset / f"{dataset}_modal1_base.fvecs")
        base_modal2 = ivecs_read(dataset_root / dataset / f"{dataset}_modal2_base.ivecs")
        query_modal1 = fvecs_read(dataset_root / dataset / f"{dataset}_modal1_query.fvecs")
        query_modal2 = ivecs_read(dataset_root / dataset / f"{dataset}_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root / dataset / f"{dataset}_gt.ivecs")
    elif dataset == 'mitstates':
        base_modal1 = fvecs_read(dataset_root / dataset / type.strip("/") / dataset_suffix / f"{dataset}_modal1_base.fvecs")
        base_modal2 = fvecs_read(dataset_root / dataset / type.strip("/") / dataset_suffix / f"{dataset}_modal2_base.fvecs")
        query_modal2 = fvecs_read(dataset_root / dataset / type.strip("/") / dataset_suffix / f"{dataset}_modal2_query.fvecs")
        gt = ivecs_read_var(dataset_root / dataset / type.strip("/") / dataset_suffix / f"{dataset}_gt.ivecs")
        if is_composition:
            query_modal1 = fvecs_read(dataset_root / dataset / type.strip("/") / dataset_suffix / f"{dataset}_modal1_2_query.fvecs")
        else:
            query_modal1 = fvecs_read(dataset_root / dataset / type.strip("/") / dataset_suffix / f"{dataset}_modal1_query.fvecs")
    elif dataset == 'msong1m':
        base_modal1 = fvecs_read(dataset_root / "msong1m" / "msong1m_modal1_base.fvecs")
        query_modal1 = fvecs_read(dataset_root / "msong1m" / "msong1m_modal1_query.fvecs")
        base_modal2 = ivecs_read(dataset_root / "msong1m" / "msong1m_modal2_base.ivecs")
        query_modal2 = ivecs_read(dataset_root / "msong1m" / "msong1m_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root / "msong1m" / "msong1m_modal2_gt.ivecs")
    elif dataset == 'uqv1m':
        base_modal1 = fvecs_read(dataset_root / "uqv1m" / "uqv_modal1_base.fvecs")
        query_modal1 = fvecs_read(dataset_root / "uqv1m" / "uqv_modal1_query.fvecs")
        base_modal2 = ivecs_read(dataset_root / "uqv1m" / "uqv_modal2_base.ivecs")
        query_modal2 = ivecs_read(dataset_root / "uqv1m" / "uqv_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root / "uqv1m" / "uqv_gt.ivecs")
    elif dataset == 'deep':
        base_modal1 = fvecs_read(dataset_root / "deep" / "deep_modal1_base.fvecs")
        query_modal1 = fvecs_read(dataset_root / "deep" / "deep_modal1_query.fvecs")
        base_modal2 = ivecs_read(dataset_root / "deep" / "deep_modal2_base.ivecs")
        query_modal2 = ivecs_read(dataset_root / "deep" / "deep_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root / "deep" / "deep_gt.ivecs")
    elif dataset == 'celeba+':
        base = dataset_root / "celeba+" / "test"
        base_modal1 = fvecs_read(base / "celeba_modal1_base.fvecs")
        query_modal1 = fvecs_read(base / "celeba_modal1_query.fvecs")
        base_modal2 = ivecs_read(base / "celeba_modal2_base.ivecs")
        query_modal2 = ivecs_read(base / "celeba_modal2_query.ivecs")
        base_modal3 = fvecs_read(base / "celeba_modal3_base.fvecs")
        query_modal3 = fvecs_read(base / "celeba_modal3_query.fvecs")
        base_modal4 = fvecs_read(base / "celeba_modal4_base.fvecs")
        query_modal4 = fvecs_read(base / "celeba_modal4_query.fvecs")
        gt = ivecs_read_var(base / "celeba_gt.ivecs")
    elif dataset == 'uqv':
        uqv_root = dataset_root / "uqv"
        if not uqv_root.exists():
            uqv_root = Path(__file__).resolve().parents[2] / "uqv"

        max_base = int(os.environ.get("MAX_BASE", 1000))
        max_query = int(os.environ.get("MAX_QUERY", 100))

        base_modal1 = fvecs_read_head(uqv_root / "uqv_base.fvecs", max_base)
        query_modal1 = fvecs_read_head(uqv_root / "uqv_query.fvecs", max_query)
        base_modal2, query_modal2 = _load_uqv_modal2(uqv_root)
        if base_modal2 is None or query_modal2 is None:
            siftsmall_base = dataset_root / "siftsmall" / "int_attribute_siftsmall_base.ivecs"
            siftsmall_query = dataset_root / "siftsmall" / "int_attribute_siftsmall_query.ivecs"
            if siftsmall_base.exists() and siftsmall_query.exists():
                base_modal2 = ivecs_read(siftsmall_base)
                query_modal2 = ivecs_read(siftsmall_query)
            else:
                base_modal2, query_modal2 = [], []

        gt_path = uqv_root / "uqv_groundtruth.ivecs"
        if not gt_path.exists():
            fallback_gt = Path(__file__).resolve().parents[2] / "uqv" / "uqv_groundtruth.ivecs"
            if fallback_gt.exists():
                gt_path = fallback_gt
        if gt_path.exists():
            gt = ivecs_read_var(gt_path)

        if len(gt) > len(query_modal1):
            gt = gt[:len(query_modal1)]

        if len(base_modal2) and len(query_modal2):
            target_base = min(len(base_modal1), len(base_modal2))
            target_query = min(len(query_modal1), len(query_modal2))
            if target_base < len(base_modal1):
                base_modal1 = base_modal1[:target_base]
            if target_base < len(base_modal2):
                base_modal2 = base_modal2[:target_base]
            if target_query < len(query_modal1):
                query_modal1 = query_modal1[:target_query]
                if len(gt) > target_query:
                    gt = gt[:target_query]
            if target_query < len(query_modal2):
                query_modal2 = query_modal2[:target_query]
            if gt and target_base:
                gt = [[idx for idx in row if idx < target_base] for row in gt]
                gt = [row if row else [0] for row in gt]

        # base_modal3 = np.zeros((len(base_modal1), 1), dtype='float32')
        # query_modal3 = np.zeros((len(query_modal1), 1), dtype='float32')
        # base_modal4 = np.zeros((len(base_modal1), 1), dtype='float32')
        # query_modal4 = np.zeros((len(query_modal1), 1), dtype='float32')
    else:  # supported options: "celeba", "shopping100k"
        base_modal1 = fvecs_read(dataset_root / dataset / "train" / dataset_suffix / f"{dataset}_modal1_base.fvecs")
        base_modal2 = ivecs_read(dataset_root / dataset / "train" / dataset_suffix / f"{dataset}_modal2_base.ivecs")
        query_modal2 = ivecs_read(dataset_root / dataset / "train" / dataset_suffix / f"{dataset}_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root / dataset / "train" / dataset_suffix / f"{dataset}_gt.ivecs")
        if is_composition:
            query_modal1 = fvecs_read(dataset_root / dataset / "train" / dataset_suffix / f"{dataset}_modal1_2_query.fvecs")
        else:
            query_modal1 = fvecs_read(dataset_root / dataset / "train" / dataset_suffix / f"{dataset}_modal1_query.fvecs")

    return base_modal1, base_modal2, base_modal3, base_modal4, query_modal1, query_modal2, query_modal3, query_modal4, gt
