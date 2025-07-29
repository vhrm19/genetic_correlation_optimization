import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import correlate, hilbert, fftconvolve, correlation_lags, argrelmax
from scipy.stats import gmean
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import fftpack

symbol_set = np.load("symbol_set.npy")
symbol_size = symbol_set.shape[1]

trans_IR = np.load("IR_150_2wy.npy")
trans_IR = np.array([trans_IR])

def symbol_solutions_to_signal(solutions: np.array):
    """
    Function to convert the random symbol sequence to a signal.

    Parameters
    ----------
    rand_symbols : :class:`np.array`
        Symbol sequence of ints choosen randomly.

    Returns
    ----------
    symbol_sequence : :class:`np.array`
        Signal representation of the random generated symbol sequence.

    """
    return np.array([np.hstack([symbol_set[symbol] for symbol in rand_symbols]) for rand_symbols in solutions])


def fitness(solutions: np.array):

    global trans_IR, symbol_size

    signals = symbol_solutions_to_signal(solutions)

    convolved_signals = fftconvolve(signals, trans_IR, mode="full", axes=1) / (len(signals) + len(trans_IR) - 1)

    n_signals = signals.shape[0]
    signals_len = convolved_signals.shape[1]

    autocorrelation_peak = np.zeros(n_signals)
    side_lobes_peak = np.zeros((n_signals, n_signals))

    hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)))[:len(x)]

    for i in range(n_signals):
        for j in range(i, n_signals):

            corr = correlate(convolved_signals[i], convolved_signals[j], mode="full") / (2*signals_len - 1)

            enveloped_signals = np.abs(hilbert3(corr))
            # enveloped_signals = np.abs(corr)
            if i == j:

                peak = np.sort(enveloped_signals[argrelmax(enveloped_signals)[0]])  # sidelobe peak
                if (np.shape(peak) != () and len(peak) > 1):
                    autocorrelation_peak[i] = peak[-1]
                    side_lobes_peak[i, j] = peak[-2]
                else:
                    autocorrelation_peak[i] = peak
            else:
                side_lobes_peak[i, j] = np.max(enveloped_signals)

    # Relative Peak Side Lobe Ratio
    RPSLR = np.max(side_lobes_peak) / gmean(autocorrelation_peak)
    return 20*np.log10(RPSLR)

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            # bitstring[i] = 1 - bitstring[i]
            bitstring[i] = np.random.randint(0, 21)


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1) - 2)
        # perform crossover
        c1 = np.concatenate((p1[:pt], p2[pt:]))
        c2 = np.concatenate((p2[:pt], p1[pt:]))

    return [c1, c2]


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# genetic algorithm
def genetic_algorithm(objective, n_iter, n_pop, num_signals, num_symbols, r_cross, r_mut):
    n_bits = num_symbols*num_signals
    # initial population of random bitstring
    pop = np.random.randint(0, 21, size=(n_pop, n_bits))
    graph_gen = []
    graph_bests = []
    graph_worse = []
    graph_mean = []
    # keep track of best solution
    best, best_eval = 0, objective(pop[0].reshape((num_signals, num_symbols)))
    # enumerate generations
    for gen in tqdm(range(n_iter)):
        # evaluate all candidates in the population
        scores = Parallel(n_jobs=-1, backend="loky", verbose=0) \
            (delayed(objective)(c.reshape((num_signals, num_symbols))) for c in pop)
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]

        graph_gen.append(gen)
        graph_bests.append(np.min(scores))
        graph_worse.append(np.max(scores))
        graph_mean.append(np.mean(scores))

        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children

    plt.plot(graph_gen, graph_bests)
    plt.plot(graph_gen, graph_worse)
    plt.plot(graph_gen, graph_mean)
    plt.legend(['best', 'worse', 'mean'])
    return [best, best_eval]


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    solution_size = 287  # Num of symbols concatenated
    n_solutions = 100
    n_signals = 3
    n_generations = 100
    # crossover rate
    r_cross = 0.99
    # mutation rate
    r_mut = 1.0 / float(solution_size*n_signals)
    best, score = genetic_algorithm(fitness, n_generations, n_solutions, n_signals, solution_size, r_cross, r_mut)

    tristates = symbol_solutions_to_signal(best.reshape((n_signals, solution_size)))
    convolved_signals = fftconvolve(tristates, trans_IR, mode="full", axes=1) / (len(tristates.T) + len(trans_IR.T) - 1)
    corr_matrix = np.zeros([n_signals, n_signals])
    enveloped_signals = np.zeros([n_signals, n_signals, 2 * (len(tristates.T) + len(trans_IR.T) - 1) - 1])
    plt.figure()
    for i in range(n_signals):
        for j in range(n_signals):
            signal_i = convolved_signals[i]
            signal_j = convolved_signals[j]

            corr = correlate(signal_i, signal_j, mode="full") / (len(signal_i) + len(signal_j) - 1)
            enveloped_signals[i, j] = np.abs(hilbert(corr))

            corr_max = np.max(enveloped_signals[i, j])

            corr_matrix[i, j] = corr_max
    lags = correlation_lags(len(convolved_signals[0]), len(convolved_signals[0]), mode="full")
    enveloped_signals_dB = 20 * np.log10(enveloped_signals / np.max(enveloped_signals))
    for i in range(n_signals):
        for j in range(n_signals):
            if (i == j):
                plt.plot(lags, enveloped_signals_dB[i, j], "r")
            else:
                plt.plot(lags, enveloped_signals_dB[i, j], "b--")
    plt.ylabel("Correlation [dB]")
    plt.title(f"{n_signals} signals, {solution_size} symbols, RPSLR: {score:0.2f}\n"
              f"{n_generations} generations, {n_solutions} solutions")
    # plt.xlim([8396 / 2 - 2000, 8396 / 2 + 2000])
    ticks = np.arange(-60, 5, 5)
    plt.ylim([-60, 5])
    plt.yticks(ticks)
    plt.grid()
    plt.xlabel('Correlation lag')
    plt.legend(['Autocorrelations', 'Cross-correlations'])

    np.save(f"results/{solution_size}_symbols_{n_signals}_signals_seed{seed}", convolved_signals)
    plt.show()
