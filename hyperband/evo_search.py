import copy
import numpy as np
import scipy.stats as sp

from sklearn.utils import check_random_state
from sklearn.model_selection._search import ParameterSampler
from .search import HyperbandSearchCV


def types_from_distribution(param_distribution):
    """"""
    items = sorted(param_distribution.items())
    numeric = []
    cat = []

    for k, v in items:
        if hasattr(v, "rvs"):
            numeric.append(k)
        else:
            cat.append(k)
    return numeric, cat


def dict_from_param_list(param_list):
    dic = {}
    for params in param_list:
        for key in params:
            if key not in dic:
                dic[key] = []
            dic[key].append(params[key])
    return dic


class CategoryKernel:
    def __init__(self, cat_vars):
        self.cat_vars = np.array(cat_vars, dtype=object)
        self.labels, counts = np.unique(self.cat_vars, return_counts=True)
        self.p = counts / counts.sum()

    def resample(self, size, seed):
        ret = seed.choice(self.labels, size=size, p=self.p)
        return [ret]


def get_child(population, param_distribution, kernel_dict=None, p=0.3, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)

    indices = rng.choice(len(population), size=2)
    a = population[indices[0]]
    b = population[indices[1]]
    child = {}
    items = sorted(param_distribution.items())
    for key, param in items:
        if np.random.uniform() >= p:
            idx = rng.choice(2)
            child[key] = (a[key], b[key])[idx]
        elif kernel_dict is not None and kernel_dict:
            child[key] = kernel_dict[key].resample(size=1, seed=rng)[0][0]
            if hasattr(param, "rvs"):
                child[key] = np.clip(child[key], param.a + 1e-2, param.b - 1e-2)
                if 'int' in str(param.dist):
                    child[key] = int(child[key])
        else:
            if hasattr(param, "rvs"):
                child[key] = param.rvs(random_state=rng)
            else:
                child[key] = param[rng.randint(len(param))]

    return child


class EHBSearchCV(HyperbandSearchCV):
    def __init__(self, estimator, param_distributions, brackets=None,
                 resource_param='n_estimators', eta=3, min_iter=1,
                 max_iter=81, skip_last=0, scoring=None, n_jobs=1,
                 refit=True, cv=None, verbose=0, p=0.3, sampling_mutation=False,
                 pre_dispatch='2*n_jobs', random_state=None, nu=2, chi=0.5,
                 error_score='raise', return_train_score=False):
        self.brackets = brackets
        self.p = p
        self.nu = nu
        self.chi = chi
        self.sampling_mutation = sampling_mutation

        super().__init__(
                 estimator=estimator, param_distributions=param_distributions,
                 resource_param=resource_param, eta=eta, min_iter=min_iter,
                 max_iter=max_iter, skip_last=skip_last, scoring=scoring, n_jobs=n_jobs,
                 refit=refit, cv=cv, verbose=verbose,
                 pre_dispatch=pre_dispatch, random_state=random_state,
                 error_score=error_score, return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        self._validate_input()

        s_max = int(np.floor(np.log(self.max_iter / self.min_iter) / np.log(self.eta)))
        B = (s_max + 1) * self.max_iter

        # TODO: fix multimetric

        # refit_metric = self.refit if self.multimetric_ else 'score'
        refit_metric = 'score'
        random_state = check_random_state(self.random_state)

        if self.skip_last > s_max:
            raise ValueError('skip_last is higher than the total number of rounds')

        reps = 1
        if self.brackets is None:
            self.brackets = s_max + 1

        if self.brackets > s_max + 1:
            reps = int(np.ceil(self.brackets / (s_max + 1)))
        
        kernel_dict = {}

        features_dict = {}

        results_list = []

        for round_index, s in enumerate(np.tile(np.arange(s_max, -1, -1), reps)[:self.brackets]):
            for key in features_dict:
                features_dict[key] = features_dict[key][-len(features_dict[key]) // 2:]
                

            n = int(np.ceil(int(B / self.max_iter / (s + 1)) * np.power(self.eta, s)))

            # initial number of iterations per config
            r = self.max_iter / np.power(self.eta, s)

            configurations = list(ParameterSampler(param_distributions=self.param_distributions,
                                                   n_iter=n,
                                                   random_state=random_state))

            if self.verbose > 0:
                print('Starting bracket {0} (out of {1}) of hyperband'
                      .format(round_index + 1, s_max + 1))

            for i in range((s + 1) - self.skip_last):

                n_configs = np.floor(n / np.power(self.eta, i))  # n_i
                n_iterations = int(r * np.power(self.eta, i))  # r_i
                n_to_keep = int(np.floor(n_configs / self.eta))

                children = []
                for _ in range(len(configurations), int(n_configs)):
                    children.append(get_child(configurations, self.param_distributions, kernel_dict=kernel_dict, p=self.p, rng=random_state))

                configurations += children

                if self.verbose > 0:
                    msg = ('Starting successive halving iteration {0} out of'
                           ' {1}. Fitting {2} configurations, with'
                           ' resource_param {3} set to {4}')

                    if n_to_keep > 0:
                        msg += ', and keeping the best {5} configurations.'

                    msg = msg.format(i + 1, s + 1, len(configurations),
                                     self.resource_param, n_iterations,
                                     n_to_keep)
                    print(msg)

                # Set the cost parameter for every configuration
                parameters = copy.deepcopy(configurations)
                for configuration in parameters:
                    configuration[self.resource_param] = n_iterations

                results = evaluate_candidates(parameters)

                if n_to_keep > 0:
                    top_configurations = [x for _, x in sorted(zip(results['rank_test_%s' % refit_metric],
                                                                   results['params']),
                                                               key=lambda x: x[0])]

                    configurations = top_configurations[:int(n_to_keep / self.nu) if int(n_to_keep / self.nu) > 1 else n_to_keep]

                    if self.sampling_mutation:
                        qual = list(sorted(results['rank_test_%s' % refit_metric]))

                        results_list.extend(qual)

                        num_features, cat_features = types_from_distribution(self.param_distributions)
                        new_features_dict = dict_from_param_list(top_configurations[-int(len(top_configurations) * (1 - self.chi)):])

                        for key, value in new_features_dict.items():
                            if key not in features_dict:
                                features_dict[key] = []
                            features_dict[key].extend(value)

                        results_array = np.array(results_list)
                        quantile = np.quantile(results_array, self.chi)
                        bool_results = (results_array > quantile)

                        for key, value in features_dict.items():
                            filtered_value = [x for i, x in enumerate(value) if bool_results[i]]
                            if key in num_features:
                                kernel_dict[key] = sp.gaussian_kde(filtered_value)
                            else:
                                kernel_dict[key] = CategoryKernel(filtered_value)

            if self.skip_last > 0:
                print('Skipping the last {0} successive halving iterations'
                      .format(self.skip_last))
