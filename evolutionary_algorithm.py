# Creating the evolutionary algorithm module

# Importing the tools from DEAP
from deap import base, creator, tools

# Defining the fitness function as a weighted sum of accuracy and complexity
def fitness_function(individual):
    # Converting the individual (a list of integers) to a neural network architecture
    architecture = individual_to_architecture(individual)
    
    # Evaluating the accuracy of the architecture using the neural architecture search module
    accuracy = neural_architecture_search(architecture)
    
    # Evaluating the complexity of the architecture as the number of parameters
    complexity = count_parameters(architecture)
    
    # Returning the fitness as a weighted sum of accuracy and complexity
    return 0.8 * accuracy - 0.2 * complexity,

# Defining the mutation operator as a random change in one element of the individual
def mutation_operator(individual):
    # Choosing a random index in the individual
    index = np.random.randint(len(individual))
    
    # Choosing a random value for the element at that index
    value = np.random.randint(1, 10)
    
    # Replacing the element at that index with the new value
    individual[index] = value
    
    # Returning the mutated individual
    return individual,

# Defining the crossover operator as a one-point crossover between two individuals
def crossover_operator(individual1, individual2):
    # Choosing a random point in the individuals
    point = np.random.randint(1, len(individual1))
    
    # Swapping the elements after that point between the individuals
    individual1[point:], individual2[point:] = individual2[point:], individual1[point:]
    
    # Returning the crossed-over individuals
    return individual1, individual2

# Defining the selection operator as a tournament selection with a given size and probability
def selection_operator(population, size, probability):
    # Creating an empty list for the selected individuals
    selected = []
    
    # Repeating until the size is reached
    while len(selected) < size:
        # Choosing two random individuals from the population
        individual1, individual2 = np.random.choice(population, 2)
        
        # Comparing their fitness values
        if fitness_function(individual1) > fitness_function(individual2):
            # Choosing the first individual with a given probability
            if np.random.random() < probability:
                selected.append(individual1)
            else:
                selected.append(individual2)
        else:
            # Choosing the second individual with a given probability
            if np.random.random() < probability:
                selected.append(individual2)
            else:
                selected.append(individual1)
    
    # Returning the selected individuals
    return selected

# Defining a function to convert an individual (a list of integers) to a neural network architecture (a MetaSequential model)
def individual_to_architecture(individual):
    # Creating an empty list for the layers of the architecture
    layers = []
    
    # Iterating over the elements of the individual
    for element in individual:
        # Choosing a layer type based on the element value
        if element == 1:
            # Adding a MetaConv2d layer with random parameters
            layers.append(MetaConv2d(np.random.randint(1, 256), np.random.randint(1, 256), kernel_size=np.random.randint(1, 5)))
        elif element == 2:
            # Adding a MetaLinear layer with random parameters
            layers.append(MetaLinear(np.random.randint(1, 256), np.random.randint(1, 256)))
        elif element == 3:
            # Adding a nn.ReLU layer
            layers.append(nn.ReLU())
        elif element == 4:
            # Adding a nn.MaxPool2d layer with random parameters
            layers.append(nn.MaxPool2d(np.random.randint(1, 5)))
        elif element == 5:
            # Adding a nn.BatchNorm2d layer with random parameters
            layers.append(nn.BatchNorm2d(np.random.randint(1, 256)))
        elif element == 6:
            # Adding a nn.Dropout layer with random parameters
            layers.append(nn.Dropout(np.random.uniform(0, 1)))
        elif element == 7:
            # Adding a nn.Flatten layer
            layers.append(nn.Flatten())
        elif element == 8:
            # Adding a nn.Softmax layer with random parameters
            layers.append(nn.Softmax(dim=np.random.randint(0, 2)))
        elif element == 9:
            # Adding a nn.Sigmoid layer
            layers.append(nn.Sigmoid())
    
    # Creating a MetaSequential model from the layers list
    architecture = MetaSequential(*layers)
    
    # Returning the architecture
    return architecture

# Defining the creator for the individuals and the population
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximizing the fitness function
creator.create("Individual", list, fitness=creator.FitnessMax) # Individuals are lists of integers
creator.create("Population", list) # Population is a list of individuals

# Defining the toolbox for the evolutionary algorithm
toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual, np.random.randint, 1, 10, 10) # Initializing individuals with 10 random integers between 1 and 10
toolbox.register("population", tools.initRepeat, creator.Population, toolbox.individual) # Initializing population with individuals
toolbox.register("evaluate", fitness_function) # Evaluating individuals using the fitness function
toolbox.register("mutate", mutation_operator) # Mutating individuals using the mutation operator
toolbox.register("mate", crossover_operator) # Crossing-over individuals using the crossover operator
toolbox.register("select", selection_operator) # Selecting individuals using the selection operator

# Defining the parameters for the evolutionary algorithm
population_size = 100 # The size of the population
generations = 50 # The number of generations
mutation_rate = 0.2 # The probability of mutation
crossover_rate = 0.8 # The probability of crossover
selection_size = 50 # The size of the selection pool
selection_probability = 0.7 # The probability of selection

# Creating an initial population using the toolbox
population = toolbox.population(population_size)

# Creating a statistics object to record the evolution process
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("mean", np.mean)
stats.register("std", np.std)

# Creating a logbook object to store the statistics
logbook = tools.Logbook()
logbook.header = ["generation", "max", "mean", "std"]

# Starting the evolution process
for generation in range(generations):
    # Evaluating the fitness of each individual in the population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Recording the statistics for this generation
    record = stats.compile(population)
    logbook.record(generation=generation, **record)
    print(logbook.stream) # Printing the statistics to the console
    
    # Selecting the next generation individuals from the current population
    offspring = toolbox.select(population, selection_size, selection_probability)
    
    # Cloning the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    
    # Applying crossover and mutation to the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # Applying crossover with a given probability
        if np.random.random() < crossover_rate:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for mutant in offspring:
        # Applying mutation with a given probability
        if np.random.random() < mutation_rate:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluating the fitness of the new individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Replacing the current population with the new population
    population[:] = offspring

# Printing the best individual and its fitness at the end of the evolution process
best_ind = tools.selBest(population, 1)[0]
print(f"Best individual: {best_ind}")
print(f"Best fitness: {best_ind.fitness.values[0]}")
