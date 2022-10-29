## Séparation du jeu de données en plusieurs parties
# Avec tirage aléatoire (qui me semble plus puissant que la stratification).

#from sklearn import model_selection
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=.3) # 30 % pour test

import numpy as np
from numpy.random import default_rng

def pepper_dataset_split(X, y, k=2, sizes=None):
    """ Split the dataset in k parts.
    If sizes is None, part are of ~equal sizes (ex n, n + 1 or n, n, n + 1)
    If size is a list, list of relative sizes with a total of 1
    if size is a callable, maps relative size from index with a total of 1
    """
    # check if X and y are not None
    if X is None or y is None:
        raise ValueError('X and y cannot be None')
    # check if X and y are ndarrays
    if not isinstance(X, np.ndarray):
        raise TypeError('X is not a numpy ndarray')
    if not isinstance(y, np.ndarray):
        raise TypeError(y, 'y is not a numpy ndarray')
    # check if X and y have compatible shape
    if (X.shape[0] != y.shape[0]):
        raise ValueError('X and y have not the same first dimension', X.shape[0], y.shape[0])
    
    n = X.shape[0]

    # cannot divide the dataset in more parts than it's size
    if k > n:
        raise ValueError('k > n', k, n) 

    if sizes is not None and sum(sizes) != 1:
        raise ValueError('sum of sizes elements must be 1')

    X_y = np.concatenate((X, y), axis=1)

    rng = default_rng()
    rng.shuffle(X_y)

    if sizes is None:  # uniform distribution
        p = n // k       # base size of folds
        r = n % k        # remainder : size of p + 1 for the r first folds
        abs_sizes = [p + 1] * r + [p] * (k - r)

    else:              # distribution fixed by sizes
        abs_sizes = [round(n * size) for size in sizes]
        diff = sum(abs_sizes) - n
        if diff > 0:
            for i in range(diff):
                abs_sizes[-(i + 1)] -= 1
        else:
            for i in range(-diff):
                abs_sizes[i] += 1

    base = 0   # base index for the next fold
    X_parts, y_parts = [], []
    for size in abs_sizes:
        end = base + size
        X_y_part = X_y[base:end]
        X_parts += [X_y_part[:, :-1]]
        y_parts += [X_y_part[:, -1]]
        base = end

    return X_parts, y_parts
        
def test_pepper_dataset_split():
    X = np.arange(11 * 5).reshape(11, 5)
    y = np.arange(11).reshape(11, 1)
    print('X :', X)
    print('y :', y)
    X_parts, y_parts = pepper_dataset_split(X, y, k=3, sizes=[.5, .3, .2])
    print('X_parts :', X_parts)
    print('y_parts :', y_parts)
    print(y_parts[0].shape)

    print((y_parts[0][1:2] + y_parts[0][4:5]).shape)



## Grid search

# helpers pour mise au point de grid_search

from itertools import product
def demo_cartesian_product():
    a = [1, 2, 3, 4]
    b = ['a', 'b', 'c']
    c = [.5, 1.5, 7.5]
    print(list(product(a, b, c)))
""" > demo_cartesian_product()
[(1, 'a', 0.5), (1, 'a', 1.5), (1, 'a', 7.5), (1, 'b', 0.5), (1, 'b', 1.5), (1, 'b', 7.5), (1, 'c', 0.5), (1, 'c', 1.5), (1, 'c', 7.5), (2, 'a', 0.5), (2, 'a', 1.5), (2, 'a', 7.5), (2, 'b', 0.5), (2, 'b', 1.5), (2, 'b', 7.5), (2, 'c', 0.5), (2, 'c', 1.5), (2, 'c', 7.5), (3, 'a', 0.5), (3, 'a', 1.5), (3, 'a', 7.5), (3, 'b', 0.5), (3, 'b', 1.5), (3, 'b', 7.5), (3, 'c', 0.5), (3, 'c', 1.5), (3, 'c', 7.5), (4, 'a', 0.5), (4, 'a', 1.5), (4, 'a', 7.5), (4, 'b', 0.5), (4, 'b', 1.5), (4, 'b', 7.5), (4, 'c', 0.5), (4, 'c', 1.5), (4, 'c', 7.5)]
"""

def demo_cartesian_product_from_dict():
    d = {'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c'], 'c': [.5, 1.5, 7.5]}
    for e in d.values():
        print(e)
    args = [v for v in d.values()]
    print(args)
    print(list(product(*[v for v in d.values()])))
""" > demo_cartesian_product_from_dict()
[1, 2, 3, 4]
['a', 'b', 'c']
[0.5, 1.5, 7.5]
[[1, 2, 3, 4], ['a', 'b', 'c'], [0.5, 1.5, 7.5]]
[(1, 'a', 0.5), (1, 'a', 1.5), (1, 'a', 7.5), (1, 'b', 0.5), (1, 'b', 1.5), (1, 'b', 7.5), (1, 'c', 0.5), (1, 'c', 1.5), (1, 'c', 7.5), (2, 'a', 0.5), (2, 'a', 1.5), (2, 'a', 7.5), (2, 'b', 0.5), (2, 'b', 1.5), (2, 'b', 7.5), (2, 'c', 0.5), (2, 'c', 1.5), (2, 'c', 7.5), (3, 'a', 0.5), (3, 'a', 1.5), (3, 'a', 7.5), (3, 'b', 0.5), (3, 'b', 1.5), (3, 'b', 7.5), (3, 'c', 0.5), (3, 'c', 1.5), (3, 'c', 7.5), (4, 'a', 0.5), (4, 'a', 1.5), (4, 'a', 7.5), (4, 'b', 0.5), (4, 'b', 1.5), (4, 'b', 7.5), (4, 'c', 0.5), (4, 'c', 1.5), (4, 'c', 7.5)]
"""


def demo_parts_shapes(X, y):
    X_parts, y_parts = pepper_dataset_split(X, y, 5)
    print([p.shape for p in X_parts])
    print([len(p) for p in y_parts])
    #display(X_parts)
    #display(vtsack_parts(X_parts))
""" > demo_parts_shapes(X_train_std, y_train)
X shape (3428, 11) y shape (3428, 1)
[(686, 11), (686, 11), (686, 11), (685, 11), (685, 11)]
[686, 686, 686, 685, 685]
"""

# réutiliser le même objet classifier, ou le copier, ou le réinstancier connaissant son type
# après lecture doc sklearn -> base.clone : Construct a new unfitted estimator with the same parameters.
from sklearn import neighbors
def demo_instanciate_from_type():
    knn_class = type(neighbors.KNeighborsClassifier)
    knn = knn_class(n_neighbors=3)
    print(knn)
    # ça, c'est encore audelà de mes compétences python
r""" > demo_instanciate_from_type()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
c:\Users\franc\Projects\pepper_data-science_practising\OC DS\P4 C1 Evaluez les performances d'un modèle de machine learning\4297211_ml_model_perf_eval.ipynb Cellule 32 in <cell line: 3>()
      1 from sklearn import neighbors
      2 knn_class = type(neighbors.KNeighborsClassifier)
----> 3 knn = knn_class(n_neighbors=3)
      4 print(knn)

TypeError: __new__() missing 3 required positional arguments: 'name', 'bases', and 'namespace'
"""

# kwargs
def demo_dict_to_kwargs():
    def foo(a, b, c):
        print(a, b, c)
    foo(a=1, c=3, b=2)
    bar = {'a': 1, 'c':3, 'b': 2}
    foo(**bar)
    t = 3, 5, 7
    print(list(bar.keys()))
    print(list(zip(list(bar.keys()), t)))
    print({k: v for k, v in zip(list(bar.keys()), t)})
""" > demo_dict_to_kwargs()
1 2 3
1 2 3
['a', 'c', 'b']
[('a', 3), ('c', 5), ('b', 7)]
{'a': 3, 'c': 5, 'b': 7}
"""

# tests sur parties
# d'abord un simple appel au KNN
from sklearn import neighbors
def demo_grid_search_step(X_train, y_train, X_test, y_test):
    # X_train_std, y_train
    classifier = neighbors.KNeighborsClassifier
    param_grid = {'n_neighbors': [3, 4, 5]}
    print(len(param_grid))
    params = list(product(*[v for v in param_grid.values()]))
    print(params)
    kwargs = {k: v for k, v in zip(list(param_grid.keys()), params[3])}
    print(kwargs)
    c = classifier(**kwargs)
    y_train = y_train.reshape(X_train.shape[0], )
    c.fit(X_train, y_train)
    y_pred = c.predict(X_test)
    print(y_pred)
    dist = y_pred == y_test
    print(dist)
    print(dist.size)
    print(dist.sum())
    print(f'perf {round(100 * dist.sum() / dist.size, 2)} %')
""" > demo_grid_search_step(X_train_std, y_train, X_test_std, y_test)
1
[(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,)]
{'n_neighbors': 4}
[1 1 1 ... 1 1 1]
[ True  True False ... False  True  True]
1470
1083
perf 73.67 %
"""

# en principe, il faut que je crée une classe.. mais commençons avec une fonction unique
from itertools import product

def grid_search(X, y, classifier, param_grid, cv=5):
    # diviser X_train en cv folds (plis) : 2 listes de cv arrays
    X_parts, y_parts = pepper_dataset_split(X, y, k=cv)
    # produire les combinaisons des hyperparamètres
    params = list(product(*[v for v in param_grid.values()]))
    # pour chaque combinaison d'hyperparamètres
    scores_by_params = {'mean_test_score': [], 'std_test_score': [], 'params': []}
    for p in range(len(params)):
        # créer le classifier avec la combinaison d'hyperparamètres
        kwargs = {k: v for k, v in zip(list(param_grid.keys()), params[p])}
        c = classifier(**kwargs)
        # (voir empreinte mém : si pas trop gros, les créer au début et les réutiliser)
        # pour chaque fold
        scores_by_fold = []
        for i in range(cv):
            # TODO : il y a de l'optimisation à trouver pour ne pas refaire ces partitions pour chaque combi de params
            # fold est le jeu de test
            X_test, y_test = X_parts[i], y_parts[i]
            # composer le jeu d'entraînement comme l'ensemble complémentaire
            X_train = np.vstack(X_parts[:i] + X_parts[i + 1:])
            y_train = np.hstack(y_parts[:i] + y_parts[i + 1:])
            # entrainer le classifier sur le jeu d'entrainement
            c.fit(X_train, y_train)
            # tester ses résultats sur le jeu de test
            y_pred = c.predict(X_test)
            dist = y_pred == y_test
            # produire et enregistrer le score
            scores_by_fold += [dist.sum() / dist.shape[0]]
        scores_by_params['mean_test_score'] += [np.mean(scores_by_fold)]
        scores_by_params['std_test_score'] += [np.std(scores_by_fold)]
        scores_by_params['params'] += [kwargs]
    return scores_by_params



# le vrai test
from sklearn import neighbors


from sklearn import neighbors, model_selection

def sklearn_test_reference(X, y, classifier=neighbors.KNeighborsClassifier):
    param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}   # paramètres testés
    score = 'accuracy'    # le score que l'on cherche à optimiser : ici la proportion de prédictions ok

    # classifieur kNN avec recherche d'hyperparamètre par validation croisée
    clf = model_selection.GridSearchCV(
        classifier(),                      # un classifieur (default kNN)
        param_grid,                        # hyperparamètres à tester
        cv=5,                              # nombre de folds de validation croisée
        scoring=score                      # score à optimiser
    )

    # Optimiser ce classifieur sur le jeu d'entraînement
    clf.fit(X, y)

    # Afficher le(s) hyperparamètre(s) optimaux
    print('Meilleur(s) hyperparamètre(s) sur le jeu d\'entraînement:')
    print(clf.best_params_)

    # Afficher les performances correspondantes
    print('Résultats de la validation croisée :')
    for mean, std, params in zip(
            clf.cv_results_['mean_test_score'],  # score moyen
            clf.cv_results_['std_test_score'],   # écart-type du score
            clf.cv_results_['params']            # valeur de l'hyperparamètre
        ):
        print(f'{score} = {mean:.3f} (+/-{std * 2:.03f}) for {params}')
""" > sklearn_test_reference(X_train_std, y_train)
Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:
{'n_neighbors': 11}
Résultats de la validation croisée :
accuracy = 0.721 (+/-0.037) for {'n_neighbors': 2}
accuracy = 0.753 (+/-0.029) for {'n_neighbors': 3}
accuracy = 0.747 (+/-0.038) for {'n_neighbors': 4}
accuracy = 0.758 (+/-0.027) for {'n_neighbors': 5}
accuracy = 0.760 (+/-0.033) for {'n_neighbors': 6}
accuracy = 0.771 (+/-0.025) for {'n_neighbors': 7}
accuracy = 0.765 (+/-0.028) for {'n_neighbors': 8}
accuracy = 0.768 (+/-0.022) for {'n_neighbors': 9}
accuracy = 0.770 (+/-0.039) for {'n_neighbors': 10}
accuracy = 0.772 (+/-0.026) for {'n_neighbors': 11}
accuracy = 0.770 (+/-0.024) for {'n_neighbors': 12}
accuracy = 0.766 (+/-0.014) for {'n_neighbors': 13}
accuracy = 0.768 (+/-0.018) for {'n_neighbors': 14}
accuracy = 0.767 (+/-0.013) for {'n_neighbors': 15}
accuracy = 0.767 (+/-0.015) for {'n_neighbors': 16}
accuracy = 0.765 (+/-0.018) for {'n_neighbors': 17}
"""

def test_grid_search(X, y, classifier=neighbors.KNeighborsClassifier):
    param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}
    y = y.reshape(X.shape[0], 1)  # ça, c'est vraiment tordu : pourquoi y_train perd sa dim ?
    scores = grid_search(X, y, classifier, param_grid, cv=5)

    # Afficher le(s) hyperparamètre(s) optimaux
    #print('Meilleur(s) hyperparamètre(s) sur le jeu d\'entraînement:')
    #print(clf.best_params_)

    # Afficher les performances correspondantes
    print('Résultats de la validation croisée :')
    for mean, std, params in zip(
            scores['mean_test_score'],  # score moyen
            scores['std_test_score'],   # écart-type du score
            scores['params']            # valeur de l'hyperparamètre
        ):
        print(f'accuracy = {mean:.3f} (+/-{std * 2:.03f}) for {params}')
""" > test_grid_search(X_train_std, y_train)
Résultats de la validation croisée :
accuracy = 0.782 (+/-0.029) for {'n_neighbors': 1}
accuracy = 0.742 (+/-0.020) for {'n_neighbors': 2}
accuracy = 0.767 (+/-0.018) for {'n_neighbors': 3}
accuracy = 0.758 (+/-0.019) for {'n_neighbors': 4}
accuracy = 0.765 (+/-0.027) for {'n_neighbors': 5}
accuracy = 0.765 (+/-0.031) for {'n_neighbors': 6}
accuracy = 0.769 (+/-0.026) for {'n_neighbors': 7}
accuracy = 0.769 (+/-0.037) for {'n_neighbors': 8}
accuracy = 0.768 (+/-0.032) for {'n_neighbors': 9}
accuracy = 0.765 (+/-0.029) for {'n_neighbors': 10}
accuracy = 0.766 (+/-0.035) for {'n_neighbors': 11}
accuracy = 0.766 (+/-0.029) for {'n_neighbors': 12}
accuracy = 0.770 (+/-0.037) for {'n_neighbors': 13}
accuracy = 0.767 (+/-0.040) for {'n_neighbors': 14}
accuracy = 0.764 (+/-0.033) for {'n_neighbors': 15}
accuracy = 0.770 (+/-0.030) for {'n_neighbors': 16}
accuracy = 0.769 (+/-0.030) for {'n_neighbors': 17}
"""