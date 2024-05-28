#Import necessary libraries
import wntr  #Water Network Tool for Resilience
import networkx as nx  #For network visualization
import numpy as np  #For numerical operations
import matplotlib.pyplot as plt  #For plotting
import matplotlib.gridspec as gridspec  #For creating grid layouts in plots
import random  #For random sampling
import pickle  #For serializing and deserializing Python objects

#Load the results from the file with pickle
with open('simulation_results.pkl', 'rb') as f:
    results = pickle.load(f)

#Extract time series data for demand and quality from simulation results
demand = results.node['demand']  #Pandas DataFrame with demand (flow rate) at each node over time
quality = results.node['quality']  #Pandas DataFrame with quality (contaminant concentration) at each node over time


#Constants
#Define the minimum concentration threshold for detection
#C_min is set to 1 mg/L
C_min = 1 
#Define the time step interval of the simulation (in seconds)
#This means the simulation updates the conditions at each node every 10 seconds.
DELTA_T = 10  

#Calculate VCW for each node
#VCW = Volume of Contaminated Water
#For each node, the code iterates over every time step
#If the contaminant concentration at a particular time 
#step and node is greater than or equal to C_min, 
#the code adds to the node's VCW the product of the demand 
#at that node and the time step interval (DELTA_T).
VCW = {}
for node in demand.columns:
    node_VCW = 0
    for time_step in demand.index:
        if quality.at[time_step, node] >= C_min:
            node_VCW += demand.at[time_step, node] * DELTA_T
    #After all time steps have been processed for a node,
    #the total contaminated volume (node_VCW) is stored in VCW
    #with the node name as the key        
    VCW[node] = node_VCW


#Implementing Evolutionary Algorithm (EA) for sensor placement
#DEAP = Distributed Evolutionary Algorithms in Python
from deap import base, creator, tools, algorithms


#Parameters for the EA
POP_SIZE = 1000  #The size of the population
NUM_GEN = 50  #The number of generations
NUM_SENSORS = 10  #The number of sensors to place
MUTATION_RATE = 0.2  #The probability of mutating an individual
CROSSOVER_RATE = 0.7  #The probability of crossing over two individuals

#The fitness function evaluates how good a solution is, guiding the 
#evolutionary algorithm in selecting and producing better solutions over generations.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  #We aim to maximize the VCW, hence a positive weight
#Individual is a list of nodes (representing sensor placements)
#With a given fitness value that tells us how good that sensor configuration is
creator.create("Individual", list, fitness=creator.FitnessMax)

#Stores various functions that will be used in the evolutionary algorithm 
toolbox = base.Toolbox()

#Creating a list of all node names in the network, which are the keys in VCW found above
network_nodes = list(VCW.keys())  #List of network nodes

#Initialization function that samples without replacement
#This function initializes an individual for the population. Each individual (a potential solution) 
#is represented as a list of nodes from the network where sensors will be placed.
#Each individual will contain NUM_SENSORS nodes, ensuring a unique set of sensor placements.
def init_individual():
    return creator.Individual(random.sample(network_nodes, NUM_SENSORS)) 

#Register the initialization function with the toolbox
toolbox.register("individual", init_individual)
#Register a population generator with the toolbox
#This creates a list of individuals by repeatedly calling toolbox.individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Mutation operator to ensure unique sensor locations
#Introduces variability into 
#the offspring by randomly changing one of the selected nodes in an individual.
#This is to explore new parts of the solution space that might lead to a better fitness value.
def mut_unique(individual):
    if random.random() < MUTATION_RATE:
        #Select a random element in the individual to mutate
        idx_to_mutate = random.randrange(len(individual))
        #Select a new, unique location for the sensor
        potential_replacements = list(set(network_nodes) - set(individual))
        #Replace the selected index with a new node from the potential replacements
        individual[idx_to_mutate] = random.choice(potential_replacements)
    #Return the mutated individual
    return individual,

#Crossover function that ensures uniqueness
#This function swaps elements between two parent individuals to create new offspring,
#ensuring no duplicate nodes within each individual by calling fix_duplicates.
def crossover_unique(ind1, ind2):
    if random.random() < CROSSOVER_RATE:
        #Perform a uniform crossover
        for i in range(len(ind1)):
            if random.random() > 0.5:
                #Swap genes
                ind1[i], ind2[i] = ind2[i], ind1[i]
        #Fix duplicate nodes within an individual
        def fix_duplicates(ind):
            used_nodes = set(ind)
            available_nodes = list(set(network_nodes) - used_nodes)
            random.shuffle(available_nodes)
            for i in range(len(ind)):
                if ind.count(ind[i]) > 1:
                    ind[i] = available_nodes.pop()
        #Fix duplicates in both offspring            
        fix_duplicates(ind1)
        fix_duplicates(ind2)
    #Return the crossover result    
    return ind1, ind2

#Registration with DEAPs toolbox
#Register the initialization function with the toolbox
toolbox.register("individual", init_individual)
#This creates a list of individuals by repeatedly calling toolbox.individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#Register the crossover operator with the toolbox
toolbox.register("mate", crossover_unique)
#Register the mutation operator with the toolbox
toolbox.register("mutate", mut_unique)
#selects the best individuals from a random subset of the population
toolbox.register("select", tools.selTournament, tournsize=3)
#The evaluation function calculates the fitness
#of an individual as the sum of VCW for the selected nodes
toolbox.register("evaluate", lambda ind: (sum(VCW[node] for node in ind),))

#Creating initial population
#Generating POP_SIZE individuals using the registered population generator
population = toolbox.population(n=POP_SIZE)

#Apply EA to the population
for gen in range(NUM_GEN):
    #Generate offspring through crossover and mutation
    #varAnd generates the offspring population using both crossover (cxpb) and mutation (mutpb)
    offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=1.0)
    #Evaluate the fitness of each offspring
    #Applying the evaluation function to each individual in the offspring
    fits = toolbox.map(toolbox.evaluate, offspring)
    #Assigning the calculated fitness values to the corresponding individuals
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    #Select the next generation population from the offspring
    #toolbox.select selects the best individuals to form the new population
    population = toolbox.select(offspring, k=POP_SIZE)

#Selecting the best individual(s) from the population based on their fitness values.
best_ind = tools.selBest(population, 1)[0]

#Load the network model from the EPANET input file
inp_file = "C:/Users/anton/OneDrive/Documents/EPANET Projects/EPANET_scripts/L-TOWN (1).inp"  # Ensure the file path matches the location of your .inp file
wn = wntr.network.WaterNetworkModel(inp_file)

#Contamination levels from the simulation results
contamination_levels = quality

G = wn.to_graph()
#This dictionary to store node positions based on their coordinates
pos = {node_name: (node.coordinates[0], node.coordinates[1]) for node_name, node in wn.nodes()}

#Extracting the water quality data for the last time step from the simulation results
node_quality = results.node['quality'].iloc[-1,:]  

#Normalize the node_quality values for color mapping, Hence 0-1
max_quality = node_quality.max()
min_quality = node_quality.min()
norm_node_quality = (node_quality - min_quality) / (max_quality - min_quality)

#Define node colors, using a specific color for sensor nodes
default_node_color = 'gray'  #Gray for non-sensor nodes
sensor_node_color = 'red'    #Red color for sensor nodes
node_colors = [sensor_node_color if node in best_ind else plt.cm.jet(norm_node_quality[node]) 
               if node in norm_node_quality else default_node_color for node in G.nodes()]

#Larger size for sensor nodes
sensor_sizes = [100 if node in best_ind else 10 for node in G.nodes()]


fig = plt.figure(figsize=(12, 8))
plt.axis("off")
plt.title('Sensor Placement for Contamination at Reservoir 1 and 2')

#Grid specification for placing subplots
gs = gridspec.GridSpec(8, 12)  # Split into 12 columns

#Assign axes
ax_network = fig.add_subplot(gs[0:8, 0:10])  #
ax_cbar = fig.add_subplot(gs[1:7, 11:12])

#Draw the network graph
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=sensor_sizes, cmap=plt.cm.jet, ax=ax_network)

#Create and add a color bar to represent contamination levels
norm = plt.Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm),
    cax=ax_cbar, orientation='vertical', label='Normalized Contamination'
)
cbar.ax.set_aspect(20)
cbar.ax.set_frame_on(True)  #Add a frame to the color bar

#Hide borders for plot
for spine in ax_network.spines.values():
    spine.set_visible(False)

#Display
plt.show()