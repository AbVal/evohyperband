import copy
import numpy as np
import scipy.stats as sp

from sklearn.utils import check_random_state


def samples_from_distr(n_samples, param_distribution, rng):
    samples = []
    items = sorted(param_distribution.items())
    for n in range(n_samples):
        sample = {}
        for key, param in items:
            if hasattr(param, "rvs"):
                sample[key] = param.rvs(random_state=rng)
            else:
                sample[key] = param[rng.randint(len(param))]
        samples.append(sample)

    return samples


def get_child(population, param_distribution, rng, p=0.3):
    indices = rng.choice(len(population), size=2)
    a = population[indices[0]][0]
    b = population[indices[1]][0]
    child = {}
    items = sorted(param_distribution.items())
    for key, param in items:
        if np.random.uniform() >= p:
            idx = rng.choice(2)
            child[key] = (a[key], b[key])[idx]
        else:
            if hasattr(param, "rvs"):
                child[key] = param.rvs(random_state=rng)
            else:
                child[key] = param[rng.randint(len(param))]

    return child


def evolution(fun, param_distribution, iters, time_limit=100, pop_size=10, etta=2, mut_p=0.3, rng=None):
    if rng is None:
        print('rng is none')
        rng = np.random.RandomState(42)
        
    population_raw = samples_from_distr(pop_size, param_distribution, rng)
    start = time.time()
    
    population = []
    for child in population_raw:
        population.append((child, fun(child)))
        if time.time() - start > time_limit:
            warnings.warn('WARNING: reduce pop_size, working as RandomSearch')
            population = list(sorted(population, key=lambda x: x[1]))
            return population[-1]
        
    
    population = list(sorted(population, key=lambda x: x[1]))

    for i in range(iters):
        if time.time() - start > time_limit:
            if i == 0:
                warnings.warn('WARNING: reduce pop_size, working as RandomSearch')
            return population[-1]

        top = pop_size // etta
        population = population[-top:]
        while len(population) < pop_size:
            child = get_child(population, param_distribution, rng, p=mut_p)
            population.append((child, fun(child)))
            if time.time() - start > time_limit:
                if i == 0:
                    warnings.warn('WARNING: reduce pop_size, working as RandomSearch')
                break

        population = list(sorted(population, key=lambda x: x[1]))
    
    return population[-1]
