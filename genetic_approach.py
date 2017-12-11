import numpy as np
import random
import run_torcs
import matplotlib.pyplot as plt
from multiprocessing import TimeoutError

DATA_FILE = "gen.data"
PLOT_NAME = "evolutionary.png"
ranges = [(-5,5,float),(-5,5,float),(-5,5,float),(-5,5,float),(-5,5,float),(10,360,int)]

def mutate_small(gene, ranges, mutation_probability=0.1, increment=0.01):
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
            # if we are at the maximum, we don't go over it
            if new_feature > ranges[index][1]:
                new_feature = ranges[index][1]
            elif new_feature < ranges[index][0]:
                new_feature = ranges[index][0]
        new_gene.append(ranges[index][2](new_feature))
    return new_gene

def mutate_random(gene, ranges, mutation_probability=0.05):
    new_gene = []
    for index, feature in enumerate(gene):
        new_feature = feature
        # with probability we mutate
        if np.random.random_sample() >= (1 - mutation_probability):
            new_feature = np.random.uniform(ranges[index][0], ranges[index][1])
        new_gene.append(ranges[index][2](new_feature))
    return new_gene

def evaluate(gene):
    evaluation = -99999999
    client, server = run_torcs.run_on_ea_tracks('scr_server', steering_values=gene[:5], max_speed=int(gene[5]), timeout=5)
    times = []
    distances = []
    for track_index, track in enumerate(client):
        distance = run_torcs.get_distance_covered(track)
        if distance == -1:
            print("error, server=",server[track_index])
            print("error, client=",client[track_index])
        else:
            distances.append(distance)
        time_run = run_torcs.get_total_time_covered(track)
        if time_run != -1:
            times.append(time_run)

    # if all runs failed, we just try again.
    if len(times) == 0:
        return evaluate(gene)

    print(distances)
    print(times)
    evaluation = sum(list(map((lambda x, y: x/y), distances, times)))/len(client)

    return evaluation

def select(population, evaluations, count):
    surviving_parents = []
    surv_fitness = []
    for x in range(count):
        index = np.argmax(np.array(evaluations))
        # we make a copy of the gene
        surviving_parents.append(population[index][:])
        surv_fitness.append(evaluations[index])
        del population[index]
        del evaluations[index]
    return surviving_parents, surv_fitness

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

def run_eval_only(population):
    evaluation = [evaluate(gene) for gene in population]
    for gene_index, gene in enumerate(population):
        print_gene(gene, gene_index, evaluation[gene_index])


def main(population_size, ranges, max_generations=100, survivor_count=5, file_name=DATA_FILE):
    population = []
    # randomly initialized population
    for index in range(population_size):
        population.append(get_random_gene(ranges))
    generation = 1

    print("max_generations={}, population_size={}, survivor_count={}".format(max_generations, population_size, survivor_count))
    with open(file_name, "+w") as f:
        f.writelines("generation,max_fitness\n")

        while not terminate(max_generations, generation):
            print_generation(generation)
            # evaluate population
            evaluation = [evaluate(gene) for gene in population]

            # write best gene value to file
            f.writelines("{},{} \n".format(generation, max(evaluation)))

            for gene_index, gene in enumerate(population):
                print_gene(gene, gene_index, evaluation[gene_index])

            # select changes the population and evaluation object
            survivors, surv_fitness = select(population=population, evaluations=evaluation, count=survivor_count)
            print_survivors(survivors=survivors, evaluations=surv_fitness)

            population = []
            for survivor in survivors:
                population.append(survivor)
                for child_i in range(0, int(population_size/survivor_count) - 1):
                    changed = False
                    while not changed:
                        new_gene = mutate_small(gene=survivor, ranges=ranges, mutation_probability=0.1, increment=0.01)
                        new_gene = mutate_random(gene=new_gene, ranges=ranges, mutation_probability=0.05)
                        if new_gene != survivor:
                            changed = True
                    population.append(new_gene)
            generation += 1

def plot(filename, plot_name):
    with open(filename) as fl:
        xs = []
        ys = []
        next(fl)
        for line in fl:
            x, y = line.strip().split(",")
            xs.append(int(x))
            ys.append(float(y))
    plt.plot(xs, ys)
    #plt.show()
    plt.savefig(plot_name)

if __name__ == "__main__":
    # run_eval_only([[0.21, 1.56, 0.68, 0.53, 1.25, 120], \
    #                [0.75, 0.75, 0, 0, 1.5, 120], \
    #                [0.19, 1.56, 0.68, 0.53, 1.25, 120], \
    #                [0.23, 1.56, 0.68, 0.53, 1.25, 120], \
    #                [0.23, 1.56, 0.68, 0.53, 1.25, 110], \
    #                [0.60, 0.80, 0.1, 0.2, 1.5, 120]])
    plot(DATA_FILE, PLOT_NAME)
    #main(population_size=20, ranges=ranges, max_generations=100, survivor_count=4)
