# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.4 64-bit
#     language: python
#     name: python3
# ---

# # Wright-Fisher model of mutation and random genetic drift

# A Wright-Fisher model has a fixed population size *N* and discrete non-overlapping generations. Each generation, each individual has a random number of offspring whose mean is proportional to the individual's fitness. Each generation, mutation may occur.

# ## Setup

import numpy as np
try:
    import itertools.izip as zip
except ImportError:
    import itertools
import pprint

# ## Make population dynamic model

# ### Basic parameters

pop_size = 100

seq_length = 10

alphabet = ['A', 'T', 'G', 'C']

base_haplotype = "AAAAAAAAAA"


def test_seqlength():
    assert len(base_haplotype) == seq_length, "The length of base_haplotype must be equal to the seq_length parameter."


# ### Setup a population of sequences

# Store this as a lightweight Dictionary that maps a string to a count. All the sequences together will have count *N*.

pop = {}

pop["AAAAAAAAAA"] = 40

pop["AAATAAAAAA"] = 30

pop["AATTTAAAAA"] = 30

pop["AAATAAAAAA"]


def test_popsize():
    assert sum(pop.values()) == pop_size, "The pop_size parameter must be equivalent to the sum of values in the pop dictionary"


pprint.pprint(pop, width=41, compact=True)

# ### Add mutation

# Mutations occur each generation in each individual in every basepair.

mutation_rate = 0.005 # per gen per individual per site

# Walk through population and mutate basepairs. Use Poisson splitting to speed this up (you may be familiar with Poisson splitting from its use in the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm)). 
#
#  * In naive scenario A: take each element and check for each if event occurs. For example, 100 elements, each with 1% chance. This requires 100 random numbers.
#  * In Poisson splitting scenario B: Draw a Poisson random number for the number of events that occur and distribute them randomly. In the above example, this will most likely involve 1 random number draw to see how many events and then a few more draws to see which elements are hit.

# First off, we need to get random number of total mutations

mutation_rate * pop_size * seq_length


def get_mutation_count():
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)


# Here we use Numpy's [Poisson random number](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.poisson.html).

get_mutation_count()

# We need to get random haplotype from the population.

pop.keys()

[x/float(pop_size) for x in pop.values()]


def get_random_haplotype():
    haplotypes = list(pop.keys()) 
    frequencies = [x/float(pop_size) for x in pop.values()]
    total = sum(frequencies)
    frequencies = [x / total for x in frequencies]
    return np.random.choice(haplotypes, p=frequencies)


# Here we use Numpy's [weighted random choice](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html).

get_random_haplotype()


# Here, we take a supplied haplotype and mutate a site at random.

def get_mutant(haplotype):
    site = np.random.randint(seq_length)
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = np.random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype


[get_mutant("AAAAAAAAAA") for _ in range(7)]


# Putting things together, in a single mutation event, we grab a random haplotype from the population, mutate it, decrement its count, and then check if the mutant already exists in the population. If it does, increment this mutant haplotype; if it doesn't create a new haplotype of count 1. 

def mutation_event():
    haplotype = get_random_haplotype()
    if pop[haplotype] > 1:
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1


pop

mutation_event()

pop


# To create all the mutations that occur in a single generation, we draw the total count of mutations and then iteratively add mutation events.

def mutation_step():
    mutation_count = get_mutation_count()
    # print(mutation_count)
    for i in range(mutation_count):
        mutation_event()


pop

mutation_step()

pop


# ### Add genetic drift

# Given a list of haplotype frequencies currently in the population, we can take a [multinomial draw](https://en.wikipedia.org/wiki/Multinomial_distribution) to get haplotype counts in the following generation.

def get_offspring_counts():
    haplotypes = list(pop.keys())
    frequencies = [x/float(pop_size) for x in pop.values()]
    return list(np.random.multinomial(pop_size, frequencies))


# Here we use Numpy's [multinomial random sample](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html).

get_offspring_counts()


# We then need to assign this new list of haplotype counts to the `pop` dictionary. To save memory and computation, if a haplotype goes to 0, we remove it entirely from the `pop` dictionary.

def offspring_step():
    haplotypes = list(pop.keys())
    counts = get_offspring_counts()
    for (haplotype, count) in zip(haplotypes, counts):
        if (count > 0):
            pop[haplotype] = count
        else:
            del pop[haplotype]


offspring_step()

pop


# ### Combine and iterate

# Each generation is simply a mutation step where a random number of mutations are thrown down, and an offspring step where haplotype counts are updated.

def time_step():
    mutation_step()
    offspring_step()


# Can iterate this over a number of generations.

generations = 5


def simulate():
    for i in range(generations):
        time_step()


simulate()

pop

# ### Record

# We want to keep a record of past population frequencies to understand dynamics through time. At each step in the simulation, we append to a history object.

pop = {"AAAAAAAAAA": pop_size}

history = []


def simulate():
    clone_pop = dict(pop)
    history.append(clone_pop)
    for i in range(generations):
        time_step()
        clone_pop = dict(pop)
        history.append(clone_pop)


simulate()

pop

len(history)

pprint.pprint(history)

history[0]

history[1]

history[2]

history[3]

history[4]

history[5]

# ## Analyze trajectories

# ### Calculate diversity

# Here, diversity in population genetics is usually shorthand for the statistic *&pi;*, which measures pairwise differences between random individuals in the population. *&pi;* is usually measured as substitutions per site.

pop


# First, we need to calculate the number of differences per site between two arbitrary sequences.

def get_distance(seq_a, seq_b):
    diffs = 0
    length = len(seq_a)
    assert len(seq_a) == len(seq_b)
    for chr_a, chr_b in zip(seq_a, seq_b):
        if chr_a != chr_b:
            diffs += 1
    return diffs / float(length)


get_distance("AAAAAAAAAA", "AAAAAAAAAB")


# We calculate diversity as a weighted average between all pairs of haplotypes, weighted by pairwise haplotype frequency.

def get_diversity(population):
    haplotypes = list(population.keys())
    haplotype_count = len(haplotypes)
    diversity = 0
    for i in range(haplotype_count):
        for j in range(haplotype_count):
            haplotype_a = haplotypes[i]
            haplotype_b = haplotypes[j]
            frequency_a = population[haplotype_a] / float(pop_size)
            frequency_b = population[haplotype_b] / float(pop_size)
            frequency_pair = frequency_a * frequency_b
            diversity += frequency_pair * get_distance(haplotype_a, haplotype_b)
    return diversity


get_diversity(pop)


def get_diversity_trajectory():
    trajectory = [get_diversity(generation) for generation in history]
    return trajectory


get_diversity_trajectory()

# ### Plot diversity

# Here, we use [matplotlib](http://matplotlib.org/) for all Python plotting.

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl

# Here, we make a simple line plot using matplotlib's `plot` function.

plt.plot(get_diversity_trajectory())


# Here, we style the plot a bit with x and y axes labels.

def diversity_plot(xlabel="generation",ylabel="diversity"):
    mpl.rcParams['font.size']=18
    trajectory = get_diversity_trajectory()
    plt.plot(trajectory, "#447CCD")    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel) 


diversity_plot()


# ### Analyze and plot divergence

# In population genetics, divergence is generally the number of substitutions away from a reference sequence. In this case, we can measure the average distance of the population to the starting haplotype. Again, this will be measured in terms of substitutions per site.

def get_divergence(population):
    haplotypes = population.keys()
    divergence = 0
    for haplotype in haplotypes:
        frequency = population[haplotype] / float(pop_size)
        divergence += frequency * get_distance(base_haplotype, haplotype)
    return divergence


def get_divergence_trajectory():
    trajectory = [get_divergence(generation) for generation in history]
    return trajectory


get_divergence_trajectory()


def divergence_plot(xlabel="generation",ylabel="divergence"):
    mpl.rcParams['font.size']=18
    trajectory = get_divergence_trajectory()
    plt.plot(trajectory, "#447CCD")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel) 


divergence_plot()


# ### Plot haplotype trajectories

# We also want to directly look at haplotype frequencies through time.

def get_frequency(haplotype, generation):
    pop_at_generation = history[generation]
    if haplotype in pop_at_generation:
        return pop_at_generation[haplotype]/float(pop_size)
    else:
        return 0


get_frequency("AAAAAAAAAA", 4)


def get_trajectory(haplotype):
    trajectory = [get_frequency(haplotype, gen) for gen in range(generations)]
    return trajectory


get_trajectory("AAAAAAAAAA")


# We want to plot all haplotypes seen during the simulation.

def get_all_haplotypes():
    haplotypes = set()   
    for generation in history:
        for haplotype in generation:
            haplotypes.add(haplotype)
    return haplotypes


get_all_haplotypes()

# Here is a simple plot of their overall frequencies.

haplotypes = get_all_haplotypes()
for haplotype in haplotypes:
    plt.plot(get_trajectory(haplotype))
plt.show()

colors = ["#781C86", "#571EA2", "#462EB9", "#3F47C9", "#3F63CF", "#447CCD", "#4C90C0", "#56A0AE", "#63AC9A", "#72B485", "#83BA70", "#96BD60", "#AABD52", "#BDBB48", "#CEB541", "#DCAB3C", "#E49938", "#E68133", "#E4632E", "#DF4327", "#DB2122"]

colors_lighter = ["#A567AF", "#8F69C1", "#8474D1", "#7F85DB", "#7F97DF", "#82A8DD", "#88B5D5", "#8FC0C9", "#97C8BC", "#A1CDAD", "#ACD1A0", "#B9D395", "#C6D38C", "#D3D285", "#DECE81", "#E8C77D", "#EDBB7A", "#EEAB77", "#ED9773", "#EA816F", "#E76B6B"]


# We can use `stackplot` to stack these trajectoies on top of each other to get a better picture of what's going on.

def stacked_trajectory_plot(xlabel="generation"):
    mpl.rcParams['font.size']=18
    haplotypes = get_all_haplotypes()
    trajectories = [get_trajectory(haplotype) for haplotype in haplotypes]
    plt.stackplot(range(generations), trajectories, colors=colors_lighter)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)


stacked_trajectory_plot()


# ### Plot SNP trajectories

def get_snp_frequency(site, generation):
    minor_allele_frequency = 0.0
    pop_at_generation = history[generation]
    for haplotype in pop_at_generation.keys():
        allele = haplotype[site]
        frequency = pop_at_generation[haplotype] / float(pop_size)
        if allele != "A":
            minor_allele_frequency += frequency
    return minor_allele_frequency


get_snp_frequency(3, 5)


def get_snp_trajectory(site):
    trajectory = [get_snp_frequency(site, gen) for gen in range(generations)]
    return trajectory


get_snp_trajectory(3)


# Find all variable sites.

def get_all_snps():
    snps = set()   
    for generation in history:
        for haplotype in generation:
            for site in range(seq_length):
                if haplotype[site] != "A":
                    snps.add(site)
    return snps


def get_snp_trajectories():
    snps = get_all_snps()
    trajectories = [(snp, get_snp_trajectory(snp)) for snp in snps]
    return trajectories


def snp_trajectory_plot(xlabel="generation"):
    mpl.rcParams['font.size']=18
    trajectories = get_snp_trajectories()
    data = []
    for trajectory, color in zip(trajectories, itertools.cycle(colors)):
        data.append(range(generations))
        data.append(trajectory[1])    
        data.append(color)
    plt.plot(*data)   
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)


snptrajectories = get_snp_trajectories()


def get_gen_snp_frequency(snptrajectories,gen):
    snp_frequencies = []
    genidx=gen-1
    for snp, trajectory in snptrajectories:
        if trajectory[genidx] > 0.0:
            snp_frequencies.append((snp,trajectory[genidx]))
    return snp_frequencies


get_gen_snp_frequency(snptrajectories, generations)

snptrajectories

snp_trajectory_plot()

# ## Scale up

# Here, we scale up to more interesting parameter values.

pop_size = 200
seq_length = 100
generations = 1000
mutation_rate = 2.5e-5 # per gen per individual per site

# In this case there are $\mu$ = 0.01 mutations entering the population every generation.

seq_length * mutation_rate

# And the population genetic parameter $\theta$, which equals $2N\mu$, is 1.

2 * pop_size * seq_length * mutation_rate

base_haplotype = ''.join(["A" for i in range(seq_length)])
pop.clear()
del history[:]
pop[base_haplotype] = pop_size

simulate()

history[-1]

# [(t,f) for t,f in enumerate(get_trajectory(list(history[-1].keys())[2])) if f>0]
[(t,f) for t,f in 
 enumerate(
    get_trajectory(
        max(history[-1], key=history[-1].get)
        )
    ) 
 if f>0]

snptrajectories = get_snp_trajectories()
get_gen_snp_frequency(snptrajectories, generations)


def get_alleles(locus,gen):
    return list(map(lambda x: x[locus], list(history[gen].keys())))


get_gen_snp_frequency(snptrajectories, 550)

get_alleles(48,5)

get_alleles(48,550)

get_alleles(48,1000)

plt.figure(num=None, figsize=(14, 14), dpi=150, facecolor='w', edgecolor='k')
plt.subplot2grid((3,2), (0,0), colspan=2)
stacked_trajectory_plot(xlabel="")
plt.subplot2grid((3,2), (1,0), colspan=2)
snp_trajectory_plot(xlabel="")
plt.subplot2grid((3,2), (2,0))
diversity_plot(ylabel="diversity/divergence")
plt.subplot2grid((3,2), (2,1))
divergence_plot(xlabel="",ylabel="")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "run wright-fisher simulation with mutation and genetic drift")
    parser.add_argument('--pop_size', type = int, default = 200, help = "population size")
    parser.add_argument('--mutation_rate', type = float, default = 0.000025, help = "mutation rate")
    parser.add_argument('--seq_length', type = int, default = 100, help = "sequence length")
    parser.add_argument('--generations', type = int, default = 1000, help = "generations")
    parser.add_argument('--summary', action = "store_true", default = False, help = "don't plot trajectories")
    parser.add_argument('--output', type = str, default = "fig_mutation_drift.png", help = "file name for figure output")

    params = parser.parse_args()
    pop_size = params.pop_size
    mutation_rate = params.mutation_rate
    seq_length = params.seq_length
    generations = params.generations
    output = params.output

    simulate()

    plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
    if params.summary:
        plt.subplot2grid((2,1), (0,0))
        diversity_plot()
        plt.subplot2grid((2,1), (1,0))
        divergence_plot()
    else:
        plt.subplot2grid((3,2), (0,0), colspan=2)
        stacked_trajectory_plot(xlabel="")
        plt.subplot2grid((3,2), (1,0), colspan=2)
        snp_trajectory_plot(xlabel="")
        plt.subplot2grid((3,2), (2,0))
        diversity_plot()
        plt.subplot2grid((3,2), (2,1))
        divergence_plot()
    plt.savefig(output, dpi=150)
