import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    if len(np.unique(feature_vector)) <= 1:
        return np.array([]), np.array([]), None, -1

    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    unique_values = np.unique(feature_sorted)
    if len(unique_values) <= 1:
        return np.array([]), np.array([]), None, -1

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    def gini_impurity(labels):
        if len(labels) == 0:
            return 0
        probabilities = np.bincount(labels) / len(labels)
        return 1 - np.sum(probabilities ** 2)

    parent_gini = gini_impurity(target_sorted)
    gini_gains = []

    for threshold in thresholds:
        left_mask = feature_sorted < threshold
        right_mask = ~left_mask

        left_target = target_sorted[left_mask]
        right_target = target_sorted[right_mask]

        if len(left_target) == 0 or len(right_target) == 0:
            gini_gains.append(-1)
            continue

        left_weight = len(left_target) / len(target_sorted)
        right_weight = len(right_target) / len(target_sorted)

        left_gini = gini_impurity(left_target)
        right_gini = gini_impurity(right_target)

        weighted_child_gini = left_weight * left_gini + right_weight * right_gini
        gini_gain = parent_gini - weighted_child_gini
        gini_gains.append(gini_gain)

    gini_gains = np.array(gini_gains)

    valid_gains = gini_gains[gini_gains != -1]
    if len(valid_gains) == 0:
        return np.array([]), np.array([]), None, -1

    best_idx = np.argmax(gini_gains)
    threshold_best = thresholds[best_idx]
    gini_best = gini_gains[best_idx]

    return thresholds, gini_gains, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if any(ft not in ["real", "categorical"] for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        n_samples = len(sub_y)

        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and n_samples < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, -np.inf, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            if len(np.unique(feature_vector)) <= 1:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None or gini == -1:
                continue

            if gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                split = feature_vector < threshold

        if feature_best is None or gini_best == -np.inf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        left_count = np.sum(split)
        right_count = n_samples - left_count
        if self._min_samples_leaf is not None and (left_count < self._min_samples_leaf or right_count < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_value = x[feature_idx]

        if feature_value < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
