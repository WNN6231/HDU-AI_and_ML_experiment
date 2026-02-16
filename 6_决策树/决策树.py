# decision_tree_c45.py

import math
from collections import Counter
import pprint

def load_data():
    data = [
        ['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否'],
    ]
    feature_names = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return data, feature_names

#计算熵
def entropy(dataset):
    labels = [row[-1] for row in dataset]
    counts = Counter(labels)
    total = len(labels)
    return -sum((cnt/total) * math.log2(cnt/total) for cnt in counts.values())

def split_data(dataset, axis, value):
    ret = []
    for row in dataset:
        if row[axis] == value:
            reduced = row[:axis] + row[axis+1:]
            ret.append(reduced)
    return ret

#计算信息增益比
def gain_ratio(dataset, axis):
    base_ent = entropy(dataset)
    total = len(dataset)
    vals = Counter(row[axis] for row in dataset)
    new_ent = 0.0
    split_info = 0.0
    for val, cnt in vals.items():
        prob = cnt / total
        new_ent += prob * entropy(split_data(dataset, axis, val))
        split_info -= prob * math.log2(prob) if prob > 0 else 0
    if split_info == 0:
        return 0
    info_gain = base_ent - new_ent
    return info_gain / split_info

#递归构建决策树
def build_tree(dataset, feature_names):
    class_list = [row[-1] for row in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return Counter(class_list).most_common(1)[0][0]

    num_features = len(dataset[0]) - 1
    ratios = [gain_ratio(dataset, i) for i in range(num_features)]
    best_idx = ratios.index(max(ratios))
    best_feat = feature_names[best_idx]

    tree = {best_feat: {}}
    unique_vals = set(row[best_idx] for row in dataset)
    for val in unique_vals:
        sub_names = feature_names[:best_idx] + feature_names[best_idx+1:]
        sub_data = split_data(dataset, best_idx, val)
        tree[best_feat][val] = build_tree(sub_data, sub_names)
    return tree

if __name__ == '__main__':
    data, features = load_data()
    tree = build_tree(data, features)
    print("生成的决策树：")
    pprint.pprint(tree, width=40, sort_dicts=False)