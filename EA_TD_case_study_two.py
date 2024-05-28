#Import necessary libraries
import wntr  #Water Network Tool for Resilience
import networkx as nx  #For network visualization
import numpy as np  #For numerical operations
import matplotlib.pyplot as plt  #For plotting
import matplotlib.gridspec as gridspec  #For creating grid layouts in plots
import random  #For random sampling
import pickle  #For serializing and deserializing Python objects

from deap import base, creator, tools, algorithms


#Load the results object from the file with pickle
with open('case_study_two_simulation.pkl', 'rb') as f:
    results = pickle.load(f)

#Load quality data
quality = results.node['quality']  # DataFrame with quality at each node over time
#C_min is set to 1 mg/L
C_min = 1  #Minimum concentration threshold for detection

#Define the new evaluation function
#checks each sensor's node to determine the first 
#time step at which contamination exceeding a predefined threshold (C_min) is detected.
def evaluate_individual(individual):
    #Initialize a list to store the first detection times for each node
    first_detection_times = []
    #Loop through each node in the individual's sensor placement
    for node in individual:
        #Find the time steps where the contamination at the node exceeds the threshold C_min
        detection_times = quality[node][quality[node] >= C_min].index
        #Check if there are any detection times
        if not detection_times.empty:
            #If there are, add the first detection time to the list
            first_detection_times.append(detection_times[0])
        else:
            #If no contamination is detected, assign infinity to indicate no detection
            first_detection_times.append(np.inf)  
    #Determine the minimum detection time from the list of first detection times        
    if first_detection_times:
        min_detection_time = min(first_detection_times)
    else:
        min_detection_time = np.inf #If the list is empty, assign infinity
    #Returning the minimum detection time
    return (min_detection_time,)

#Setup DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  #We aim to minimize TD
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

#List of potential sensor nodes (from the quality data columns)
network_nodes = list(quality.columns)  

#Probability of applying crossover
CROSSOVER_RATE = 0.1
#Custom crossover function that ensures uniqueness

def crossover_unique(ind1, ind2):
    if random.random() < CROSSOVER_RATE:
        #perform a uniform crossover
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

#Initialization function that samples without replacement
#This function initializes an individual for the population. Each individual (a potential solution) 
#is represented as a list of nodes from the network where sensors will be placed.
#Each individual will contain 10 sensors
def init_individual():
    return creator.Individual(random.sample(network_nodes, 10))  

#Register the initialization function with the toolbox
toolbox.register("individual", init_individual)
#Register a population generator with the toolbox
#This creates a list of individuals by repeatedly calling toolbox.individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#The evaluation function calculates the fitness
#of an individual as the sum of VCW for the selected nodes
toolbox.register("evaluate", evaluate_individual)
#Register the crossover operator with the toolbox
toolbox.register("mate", crossover_unique)
#Register the mutation operator with the toolbox
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
#selects the best individuals from a random subset of the population
toolbox.register("select", tools.selTournament, tournsize=3)

#Create the population
population = toolbox.population(n=1000)  

#Apply EA to the population
for gen in range(100):  
    #Generate offspring through crossover and mutation
    #varAnd generates the offspring population using both crossover (cxpb) and mutation (mutpb)
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    #Evaluate the fitness of each offspring
    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind)
    #Select the next generation population from the offspring
    #toolbox.select selects the best individuals to form the new population
    population = toolbox.select(offspring, k=100)

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

# Define node colors, using a specific color for sensor nodes
default_node_color = 'gray'  #Gray for non-sensor nodes
sensor_node_color = 'red'    #Red color for sensor nodes
node_colors = [sensor_node_color if node in best_ind else plt.cm.jet(norm_node_quality[node]) 
               if node in norm_node_quality else default_node_color for node in G.nodes()]

#Larger size for sensor nodes
sensor_sizes = [100 if node in best_ind else 10 for node in G.nodes()]

fig = plt.figure(figsize=(12, 8))
plt.axis("off")
plt.title('Case Study 2, TD')
gs = gridspec.GridSpec(8, 12)  #Split into 12 columns

# Assign axes
ax_network = fig.add_subplot(gs[0:8, 0:10])  
ax_cbar = fig.add_subplot(gs[1:7, 11:12])

#Draw the network graph
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=sensor_sizes, cmap=plt.cm.jet, ax=ax_network)

#Create and add a color bar to represent contamination levels
norm = plt.Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm),
    cax=ax_cbar, orientation='vertical', label='Normalized Contamination'
)
cbar.ax.set_aspect(20)  # Adjust the aspect ratio
cbar.ax.set_frame_on(True)  # Add a frame to the color bar

#Hide borders for plot
for spine in ax_network.spines.values():
    spine.set_visible(False)

#Display
plt.show()



