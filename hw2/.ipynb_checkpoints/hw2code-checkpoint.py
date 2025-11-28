import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer


def find_best_split(feature_vector, target_vector):
    """
    Правильная реализация поиска лучшего разделения по критерию Джини
    """
    if len(np.unique(feature_vector)) <= 1:
        return np.array([]), np.array([]), None, -1

    # Сортируем признаки и цели
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
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in ["real", "categorical"] for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # Критерии остановки
        n_samples = len(sub_y)

        # 1. Все объекты одного класса
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # 2. Достигнута максимальная глубина
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # 3. Слишком мало samples для разделения
        if n_samples < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, -np.inf, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            # Пропускаем константные признаки
            if len(np.unique(feature_vector)) <= 1:
                continue

            # Ищем лучшее разделение
            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None or gini == -1:
                continue

            if gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                split = feature_vector < threshold

        # Если не нашли хорошего разделения
        if feature_best is None or gini_best == -np.inf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверяем min_samples_leaf
        left_count = np.sum(split)
        right_count = n_samples - left_count
        if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
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


# Функция для анализа параметров (Задание 8)
def analyze_hyperparameters():
    tic_tac_toe = pd.read_csv('datasets/tic-tac-toe-endgame.csv', header=None)

    X = tic_tac_toe.iloc[:, :-1].values
    y = tic_tac_toe.iloc[:, -1].values

    X_encoded = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        le = LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print("\n1. Анализ max_depth:")
    max_depths = [1, 2, 3, 5, 10, 15, 20, None]
    depth_results = []

    for max_depth in max_depths:
        tree = DecisionTree(feature_types=['categorical'] * X_encoded.shape[1],
                            max_depth=max_depth)

        scores = cross_val_score(tree, X_encoded, y_encoded, cv=5,
                                 scoring=make_scorer(accuracy_score))
        depth_results.append(scores.mean())
        print(f"max_depth={max_depth}: accuracy = {scores.mean():.4f}")

    print("\n2. Анализ min_samples_split:")
    min_splits = [2, 5, 10, 20, 50, 100]
    split_results = []

    for min_split in min_splits:
        tree = DecisionTree(feature_types=['categorical'] * X_encoded.shape[1],
                            min_samples_split=min_split)

        scores = cross_val_score(tree, X_encoded, y_encoded, cv=5,
                                 scoring=make_scorer(accuracy_score))
        split_results.append(scores.mean())
        print(f"min_samples_split={min_split}: accuracy = {scores.mean():.4f}")

    print("\n3. Анализ min_samples_leaf:")
    min_leaves = [1, 2, 5, 10, 20, 50]
    leaf_results = []

    for min_leaf in min_leaves:
        tree = DecisionTree(feature_types=['categorical'] * X_encoded.shape[1],
                            min_samples_leaf=min_leaf)

        scores = cross_val_score(tree, X_encoded, y_encoded, cv=5,
                                 scoring=make_scorer(accuracy_score))
        leaf_results.append(scores.mean())
        print(f"min_samples_leaf={min_leaf}: accuracy = {scores.mean():.4f}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot([str(d) for d in max_depths], depth_results, 'bo-')
    plt.title('Влияние max_depth на accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    plt.plot(min_splits, split_results, 'ro-')
    plt.title('Влияние min_samples_split на accuracy')
    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(min_leaves, leaf_results, 'go-')
    plt.title('Влияние min_samples_leaf на accuracy')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    return depth_results, split_results, leaf_results