import numpy as np
import json
import sys
import csv

class Parameters:
	def __init__(self, parameter_dictionary):
		self.max_adaptive_period = parameter_dictionary["MAXIMUM_ADAPTIVE_PERIOD"]
		self.seed = parameter_dictionary["SEED"]
		self.output_file = parameter_dictionary["OUTPUT_FILENAME"]
		self.colony_size = parameter_dictionary["SIZE_OF_COLONY"]
		self.game_size = parameter_dictionary["NUMBER_OF_WORKERS_PER_GAME_GROUP"]
		self.k_reward_foraging = parameter_dictionary["LINEAR_SLOPE_OF_REWARD_FROM_FORAGING"]
		self.alpha_foraging_contribution = parameter_dictionary["PROPORTION_OF_CONTRIBUTION_FROM_FORAGING"]
		self.k_cost_fanning = parameter_dictionary["LINEAR_SLOPE_OF_COST_OF_FANNING"]
		self.q1_cost_foraging = parameter_dictionary["QUADRATIC_SQUARED_COEFFICIENT_OF_COST_OF_FORAGING"]
		self.q2_cost_foraging = parameter_dictionary["QUADRATIC_LINEAR_COEFFICIENT_OF_COST_OF_FORAGING"]
		self.selection_intensity = parameter_dictionary["INTENSITY_OF_SELECTION"]
		self.mutation_rate = parameter_dictionary["RATE_OF_MUTATION"]
		self.mutation_perturbation = parameter_dictionary["PERTURBATION_OF_MUTATION"]
		self.k_reinforcement_learning = parameter_dictionary["NUMBER_OF_GAME_ROUNDS_BEFORE_STRATEGY_REINFORCEMENT"]
        


class Agent:
	def __init__(self, initask):
		self.currentTask = initask
		self.recruitable = False
		self.traitValue = 0.5
    
	def updateTask(self, newTask):
		self.currentTask = newTask

	def updaterecruitable(self, re):
		self.recruitable = re

	def updateTraits(self, newTraits):
		self.traitValue  = newTraits

	def getTask(self):
		return self.currentTask


	def getRecruitable(self):
		return self.recruitable

	def getTrait(self):
		return self.traitValue


class Simulation:
	def __init__(self, parameters):
		self.parameters = parameters
		np.random.seed(self.parameters.seed)

	@staticmethod
	def linear_perception(k, x):
		return k * x
	@staticmethod
	def quadratic_perception(a, b, x):
		return a * (x ** 2) + b * x

	def benefit_quadratic_fanning_linear_foraging(self, traits_per_game):
		number_of_workers = len(traits_per_game)
		return self.quadratic_perception(-4.0 / (number_of_workers ** 2), 4.0 / number_of_workers,
					np.sum(traits_per_game)) * self.linear_perception(self.parameters.k_reward_foraging,
						self.parameters.alpha_foraging_contribution * np.sum(1.0 - traits_per_game)
							) / number_of_workers 

	def cost_linear_fanning_quadratic_foraging(self, traits_per_game):
		return self.linear_perception(self.parameters.k_cost_fanning, traits_per_game) + self.quadratic_perception(self.parameters.q1_cost_foraging, self.parameters.q2_cost_foraging, 1.0 - traits_per_game)

	def evaluate(self, payoffs_all):
		return np.exp(self.parameters.selection_intensity * payoffs_all)

	def payoff(self, benefit_function, cost_function, traits_all):
		number_of_workers = len(traits_all)
		payoffs_all = np.zeros(number_of_workers)
		number_of_games = int(number_of_workers / self.parameters.game_size)
		for j in np.arange(number_of_games):
			payoffs_all[self.parameters.game_size * j:self.parameters.game_size * (j + 1)] = benefit_function(traits_all[self.parameters.game_size * j:self.parameters.game_size * (j + 1)]) - cost_function(traits_all[self.parameters.game_size * j:self.parameters.game_size * (j + 1)])
		return payoffs_all
	

	
	def mutate_gaussian(self, traits_all):
		number_of_workers = len(traits_all)
		number_of_mutants = np.random.binomial(number_of_workers, self.parameters.mutation_rate)
		mutant_indices = np.random.choice(number_of_workers, number_of_mutants, False)
		for i in mutant_indices:
		    traits_all[i] = np.random.normal(loc=traits_all[i], scale=self.parameters.mutation_perturbation)
		traits_all[traits_all < 0] = 0.0
		traits_all[traits_all > 1] = 1.0
		return traits_all

	def getTraitsList(self, allAgents):
		traitsList = []
		for i in allAgents:
			traitsList.append(i.getTrait())
		return traitsList

	def updateTraitsForAgents(self, newTraitsList,allAgents):	
		for i in range(len(allAgents)):
			newTrait = newTraitsList[i]
			allAgents[i].updateTraits(newTrait)
		return allAgents

	def replicate_recruitment(self, evaluations_all, traits_all):
		re = []
		for evaluate in evaluations_all:
			p_weighted = evaluations_all/np.sum(evaluations_all)
			re = np.append(re, np.random.choice(traits_all, size=1, p=p_weighted))
		return re

	def social_learning(self, initialPopulation, benefit_function, cost_function, record_frequency=200):
		output_file = open(self.parameters.output_file, 'w')
		output_writer = csv.writer(output_file)
		allAgents =  initialPopulation	
		traits_all = np.array(self.getTraitsList(allAgents))
		for i in np.arange(self.parameters.max_adaptive_period):			
			payoffs_record = self.payoff(benefit_function, cost_function, traits_all)          	
			new_traits = self.replicate_recruitment(self.evaluate(payoffs_record), traits_all)			
			new_traits = self.mutate_gaussian(new_traits)
			allAgents = self.updateTraitsForAgents(new_traits,allAgents)
			traits_all = new_traits
			if np.remainder(i, record_frequency) == 0:
				output_line = np.append(i, new_traits)
				output_line = np.append(output_line, np.mean(payoffs_record))
				output_writer.writerow(['{:0.3f}'.format(x) for x in output_line])
		output_file.close()



def run(json_filename, which_process, traits_initial=None):
    parameter_file = open(json_filename)
    parameter_dictionary = json.load(parameter_file)
    parameter_file.close()
    parameters = Parameters(parameter_dictionary)
    simulation = Simulation(parameters)
    print("in")
    if traits_initial is None:
        traits_initial = [Agent("T")]*simulation.parameters.colony_size
    if which_process == "social_learning":
        simulation.social_learning(traits_initial, 
        					simulation.benefit_quadratic_fanning_linear_foraging,
                                   		simulation.cost_linear_fanning_quadratic_foraging)


if __name__ == '__main__':
    assert len(sys.argv) == 3, "Please specify the filename of parameters and the name of model"
    run(sys.argv[1], sys.argv[2])