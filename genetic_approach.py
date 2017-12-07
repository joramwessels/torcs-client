import numpy as np
import random
import run_torcs
from multiprocessing import TimeoutError

ranges = [(-5,5,float),(-5,5,float),(-5,5,float),(-5,5,float),(-5,5,float),(10,360,int)]

def mutate_small(gene, ranges, mutation_probability=0.2, increment=0.01):
    new_gene = []
    for index, feature in enumerate(gene):
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
        new_gene.append(ranges[index][2](new_feature))

def mutate_random(gene, ranges, mutation_probability=0.1):
    new_gene = []
    for index, feature in enumerate(gene):
        new_feature = feature
        # with probability we mutate
        if np.random.random_sample() >= (1 - mutation_probability):
            new_feature = np.random.uniform(ranges[index][0], ranges[index][1])
        new_gene.append(ranges[index][2](new_feature))

def evaluate(gene):
    evaluation = -99999999
    client, server = run_torcs.run_on_ea_tracks('scr_server', steering_values=gene[:5], max_speed=int(gene[5]), timeout=3)
    time = []
    distance = []
    for track in client:
        distance.append(run_torcs.get_distance_covered(track))
        time.append(run_torcs.get_total_time_covered(track))
    print(distance)
    print(time)
    evaluation = sum(list(map((lambda x, y: x/y), distance, time)))/len(client)

    return evaluation

def select(population, evaluations, count):
    surviving_parents = []
    print(evaluations)
    for x in range(count):
        index = np.argmax(np.array(evaluations))
        surviving_parents.append(population[index])
        del population[index]
        del evaluations[index]
    return surviving_parents

def get_random_gene(ranges):
    gene = []
    for low, high,_ in ranges:
        gene.append(np.random.uniform(low, high))
    return gene

def terminate(max_generations, generation):
    if generation > max_generations:
        print("Maximum generation reached")
        return True
    # if maximum - average <= 1:
    #     print("Maximum and average are close")
    #     return True
    # else:
    return False

def print_generation_values(maximum, average, minimum):
    print("maximum:", maximum)
    print("average:", average)
    print("minimum:", minimum)

def print_generation(number):
    print("Generation={}".format(number))

def print_gene(gene, gene_index, fitness):
    floats_adjusted = ", ".join(["%.2f"%x for x in gene[:5]])
    print("gene_{}: {}, speed={}, fitness={}".format(gene_index, floats_adjusted, int(gene[5]), fitness))

def print_survivors(survivors, evaluations):
    print("selecting survivors:")
    for survivor_index, survivor in enumerate(survivors):
        print_gene(survivor, survivor_index, evaluations[survivor_index])

def main(population_size, ranges, max_generations=100, survivor_count=5):
    population = []
    for index in range(population_size):
        population.append(get_random_gene(ranges))
    generation = 0

    print("max_generations={}, population_size={}, survivor_count={}".format(max_generations, population_size, survivor_count))
    while not terminate(max_generations, generation):
        generation += 1
        print_generation(generation)
        evaluation = [evaluate(gene) for gene in population]
        for gene_index, gene in enumerate(population):
            print_gene(gene, gene_index, evaluation[gene_index])

        survivors = select(population=population, evaluations=evaluation, count=survivor_count)
        print_survivors(survivors=survivors, evaluations=evaluation)

        population = []
        for survivor in survivors:
            population.append(survivor)
            for child_i in range(0, int(population_size/survivor_count)):
                changed = False
                while not changed:
                    new_gene = mutate_small(gene=survivor, ranges=ranges, mutation_probability=0.2, increment=0.01)
                    print(new_gene)
                    new_gene = mutate_random(gene=new_gene, ranges=ranges, mutation_probability=0.2)
                    if new_gene != survivor:
                        changed = True
                population.append(new_gene)


if __name__ == "__main__":
    main(population_size=2, ranges=ranges, max_generations=3, survivor_count=1)
