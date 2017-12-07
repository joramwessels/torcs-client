import numpy as np
import random
import run_torcs
from multiprocessing import TimeoutError
import time

ranges = [(-5,5,float),(-5,5,float),(-5,5,float),(-5,5,float),(-5,5,float),(10,360,int)]

def mutate_small(gene, ranges, mutation_probability=0.2, increment=0.01):
    new_gene = []
    for index, feature in enumerate(gene):
        print("working with:",feature)
        new_feature = feature
        # with probability we mutate
        if np.random.random_sample() >= (1 - mutation_probability):
            total_range = ranges[index][1] - ranges[index][0]
            change = increment * total_range
            sign = 1
            if np.random.random_sample() >= 0.5:
                sign = -1
            change *= sign
            new_feature += change
            if new_feature > ranges[index][1]:
                new_feature = ranges[index][1]
            elif new_feature < ranges[index][0]:
                new_feature = ranges[index][0]
            print("Mutated:", new_feature)
        new_gene.append(ranges[index][2](new_feature))

def mutate_random(gene, ranges, mutation_probability=0.1):
    new_gene = []
    for index, feature in enumerate(gene):
        print("working with:",feature)
        new_feature = feature
        # with probability we mutate
        if np.random.random_sample() >= (1 - mutation_probability):
            new_feature = np.random.uniform(ranges[index][0], ranges[index][1])
            print("Mutated:", new_feature)
        new_gene.append(ranges[index][2](new_feature))

def evaluate(gene):
    evaluation = -99999999
    gene = [0.21, 1.56, 0.68, 0.53, 1.25, 160]
    print(gene)
    try:
        client, server = run_torcs.run_on_all_tracks('scr_server', steering_values=gene[:5], max_speed=gene[5], timeout=5)
        for line in server:
            print(line)
        for line in client:
            print(line)
    except TimeoutError as e:
        pass
    print("done!")
    return evaluation

def select(population, evaluations, count):
    surviving_parents = []
    for x in range(count):
        index = np.argmax(evaluations)
        surviving_parents.append(population[index])
        del population[index]
        del evaluations[index]
    return surviving_parents

def get_random_gene(ranges):
    gene = []
    for low, high,_ in ranges:
        gene.append(np.random.uniform(low, high))
    return gene

def terminate(evaluation):
    maximum = max(evaluation)
    average = sum(evaluation)/len(evaluation)
    print_evalution(maximum, average, min(evaluation))
    if maximum - average <= 50:
        return True
    else:
        return False

def print_evalution(maximum, average, minimum):
    print("maximum:", maximum)
    print("average:", average)
    print("minimum:", minimum)

def main(population_size, ranges):
    population = []
    for index in range(population_size):
        population.append(get_random_gene(ranges))

    evaluation = [evaluate(gene) for gene in population]
    print("starting evaluation:", evaluation)

    survivor_count = 5
    while not terminate(evaluation):
        survivors = select(population=population, evaluations=evaluation, count=survivor_count)
        population = []
        for survivor in survivors:
            population.append(survivor)
            for child_i in range(0, int(population_size/survivor_count)):
                changed = False
                while not changed:
                    new_gene = mutate_small(gene=survivor, ranges=ranges, mutation_probability=0.2, increment=0.01)
                    new_gene = mutate_small(gene=new_gene, ranges=ranges, mutation_probability=0.2, increment=0.01)
                    if new_gene != survivor:
                        changed = True
                population.append(new_gene)
        evaluation = [evaluate(gene) for gene in population]


if __name__ == "__main__":
    main(1, ranges)