import numpy as np
import random
import copy
from tqdm import tqdm
import config
import random

np.random.seed(45)
random.seed(45)

class NodeState:
	def __init__(self, num_actions, env):
		self.regret_sum = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.strategy_sum = np.zeros(num_actions)
		self.num_actions = num_actions
		self.number = 1
		# game information
		self.static_env = env
		self.env = copy.deepcopy(self.static_env)
	

		
class NodeInfoSet:
	def __init__(self, num_actions, history):
		self.regret_sum = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.strategy_sum = np.zeros(num_actions)
		self.num_actions = num_actions
		self.number = 1
		self.history = history

	
	def get_strategy(self):
		normalizing_sum = 0
		for a in range(self.num_actions):
			if self.regret_sum[a] > 0:
				self.strategy[a] = self.regret_sum[a]
			else:
				self.strategy[a] = 0
			normalizing_sum += self.strategy[a]

		for a in range(self.num_actions):
			if normalizing_sum > 0:
				self.strategy[a] /= normalizing_sum
			else:
				self.strategy[a] = 1.0/self.num_actions

		return self.strategy

	def get_average_strategy(self):
		avg_strategy = np.zeros(self.num_actions)
		normalizing_sum = 0
		for action in range(self.num_actions):
			normalizing_sum += self.strategy_sum[action]
		for action in range(self.num_actions):
			if normalizing_sum > 0:
				avg_strategy[action] = self.strategy_sum[action] / normalizing_sum
			else:
				avg_strategy[action] = 1.0 / self.num_actions
		
		return avg_strategy

import sys
sys.path.append("..")
from GameModel.TempleOfHorror import TempleOfHorror


class TempleCFR():
	def __init__(self, iterations, nodes):
		self.iterations = iterations
		self.nodes_state = nodes
		self.nodes = dict()
		self.env = TempleOfHorror()
		self.env_aux = copy.deepcopy(self.env)
		self.acting_player = None
		self.agents_order = [0,1,2,3]


	def cfr_iterations_external(self):
		utilities = list()
		utility = np.zeros(self.env.N)
		for t in tqdm(range(1, self.iterations + 1)):
			observation = self.env.reset() 
			random.shuffle(self.agents_order)
			self.random_order = self.agents_order
			self.acting_player = self.random_order[0]
			self.env.agent_key = self.random_order[0]


			for learning_player in [0,1,2,3]: # Players
				#print("Player plays:", player)
				probability_players = {agent:1.0 for agent in self.env.agents} 
				self.env_aux = copy.deepcopy(self.env)
				infoSet = self.env.create_observation(learning_player, observation)
				


				hand = config.compute_hand_from_labels(self.env_aux.player_hands[f'agent_{learning_player}'])

				utility[learning_player] += self.external_cfr_message(f"ActingP:{learning_player},{self.env_aux.player_role[f'agent_{learning_player}'][:-2]},H:{hand}GameInits->(P:{self.acting_player}", str(infoSet),  learning_player,  self.acting_player, t, probability_players, self.acting_player)
				#print(player, utility[player])
			utilities.append(utility.copy()/t)

		print('Average game value 0: {}'.format(utility[0]/(self.iterations)))
		print('Average game value 1: {}'.format(utility[1]/(self.iterations)))
		print('Average game value 2: {}'.format(utility[2]/(self.iterations)))
		#for i in sorted(self.nodes_state):
			#print(i, self.nodes_state[i].get_average_strategy())
		return utilities
				
	  

	def external_cfr(self, history, infoSet, learning_player, acting_player, t, probability_players):

		"""
		Perform the CFR algorithm for a given information set.
		
		Parameters:
		infoSet (str): The current information set.
		learning_player (int): The player learning the strategy.
		acting_player (int): The player currently taking action.
		t (int): The current iteration.
		probability_players (dict): Probability distribution of players.
		
		Returns:
		float: The utility value for the current state.
		"""
	


		# Build infoset
		if infoSet not in self.nodes_state:
			self.nodes_state[infoSet] = NodeState(len(self.env.action_spaces[f"agent_{acting_player}"]), copy.deepcopy(self.env_aux))
		else:
			self.nodes_state[infoSet].number += 1	
			# Reset to original infoset env to run analysis again
			self.nodes_state[infoSet].env = copy.deepcopy(self.nodes_state[infoSet].static_env)

		if history not in self.nodes:
			self.nodes[history] = NodeInfoSet(len(self.env.action_spaces[f"agent_{acting_player}"]), history)
		else:
			self.nodes[history].number += 1	


		done, winner = self.nodes_state[infoSet].env.referee()

		# History is in a terminal state then calculate payments
		if done:		
			
			# To change
			fire_message_0 = self.nodes_state[infoSet].env.message_history[0]> 0
			fire_message_1 = self.nodes_state[infoSet].env.message_history[2]> 0
			fire_message_2 = self.nodes_state[infoSet].env.message_history[4]> 0
			nor_fire_hand_0 = 2 not in self.nodes_state[infoSet].env.enc_player_hands[f"agent_0"]
			nor_fire_hand_1 = 2 not in self.nodes_state[infoSet].env.enc_player_hands[f"agent_1"]
			nor_fire_hand_2 = 2 not in self.nodes_state[infoSet].env.enc_player_hands[f"agent_2"]
			lying_fire_0 = fire_message_0*nor_fire_hand_0
			lying_fire_1 = fire_message_1*nor_fire_hand_1
			lying_fire_2 = fire_message_2*nor_fire_hand_2

			gold_message_0 = self.nodes_state[infoSet].env.message_history[1] > 0
			gold_message_1 = self.nodes_state[infoSet].env.message_history[3] > 1
			gold_message_2 = self.nodes_state[infoSet].env.message_history[5] > 1
			nor_gold_hand_0 = 3 not in self.nodes_state[infoSet].env.enc_player_hands[f"agent_0"]
			nor_gold_hand_1 = 3 not in self.nodes_state[infoSet].env.enc_player_hands[f"agent_1"]
			nor_gold_hand_2 = 3 not in self.nodes_state[infoSet].env.enc_player_hands[f"agent_2"]
			lying_gold_0 = gold_message_0*nor_gold_hand_0
			lying_gold_1 = gold_message_1*nor_gold_hand_1
			lying_gold_2 = gold_message_2*nor_gold_hand_2			

			lying_payoff = 1*lying_fire_0 + .5*lying_fire_1 + .5*lying_fire_2 + 1*lying_gold_0 + .5*lying_gold_1 + .5*lying_gold_2

			if self.nodes_state[infoSet].env.enc_player_role[f"agent_{acting_player}"] == 1: ## If player is a Defender

				if self.nodes_state[infoSet].env.enc_player_role[f"agent_{acting_player}"] == winner: ## If player won

					if self.nodes_state[infoSet].env.enc_player_role[f"agent_{learning_player}"] == 1: 
						return 2 + lying_payoff
					else:
						return - 2 - lying_payoff
				else:

					if self.nodes_state[infoSet].env.enc_player_role[f"agent_{learning_player}"] == 1: 
						return  - 2 - lying_payoff
					else:
						return  2 + lying_payoff
					
			else: ## If player is an Attacker 
				if self.nodes_state[infoSet].env.enc_player_role[f"agent_{acting_player}"] == winner:

					if self.nodes_state[infoSet].env.enc_player_role[f"agent_{learning_player}"] == 1: 
						return -2 - lying_payoff
					else:
						return 2 + lying_payoff

				else:

					if self.nodes_state[infoSet].env.enc_player_role[f"agent_{learning_player}"] == 1: 
						return 2 + lying_payoff
					else:
						return -2 - lying_payoff
					
 
				
		## Learning procedure
		if acting_player == learning_player:
			action_space_length = len(self.nodes_state[infoSet].env.action_spaces[f"agent_{acting_player}"])
			utility = np.zeros(action_space_length) 
			node_utility = 0
			strategy = self.nodes[history].get_strategy()

			for index, action in enumerate(self.nodes_state[infoSet].env.action_spaces[f"agent_{acting_player}"]):
				# Reset to original infoset env to run analysis again
				self.nodes_state[infoSet].env = copy.deepcopy(self.nodes_state[infoSet].static_env)
				
				# Step ahead
				next_acting_player = action
				done, next_observation_spaces, _, card, _ = self.nodes_state[infoSet].env.step(action)
				nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)
				card = card.replace("_","")

				probability_players[f"agent_{acting_player}"] *= strategy[index]

				# Update global env after state change
				self.env_aux = copy.deepcopy(self.nodes_state[infoSet].env)

				# new message stage
				if self.nodes_state[infoSet].env.message_provided:

					utility[index] = self.external_cfr(history+f",A:{action},C:{card})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players)
				else:
					agent_key = next_acting_player
					next_acting_player = self.random_order[self.nodes_state[infoSet].env.provide_message]
					hand = config.compute_hand_from_labels(self.nodes_state[infoSet].env.player_hands[f'agent_{learning_player}'])
					nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)
					utility[index] = self.external_cfr_message(history+f",A:{action},C:{card})*NEW-TURN*HAND:{hand}*->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players, agent_key)
				


				
				node_utility += strategy[index] * utility[index]



			if learning_player==0:
				p_other = probability_players["agent_1"]*probability_players["agent_2"]*probability_players["agent_3"]
			elif learning_player==1:
				p_other = probability_players["agent_0"]*probability_players["agent_2"]*probability_players["agent_3"]
			elif learning_player==2:
				p_other = probability_players["agent_0"]*probability_players["agent_1"]*probability_players["agent_3"]	
			else:
				p_other = probability_players["agent_0"]*probability_players["agent_1"]*probability_players["agent_2"]			

			# Update regrets
			for action in range(action_space_length):
				regret = utility[action] - node_utility
				self.nodes[history].regret_sum[action] += max(p_other*regret,0)
				#print(regret)

			for index_action in range(action_space_length):
				self.nodes[history].strategy_sum[index_action] += strategy[index_action]*t

			

			return node_utility

		else:  # Here is where self play is done

			action_space_length = len(self.nodes_state[infoSet].env.action_spaces[f"agent_{acting_player}"])
			prefix_hist_learn = f"P:{learning_player},H:{self.env_aux.enc_player_hands[f'agent_{learning_player}']}->(P:{self.acting_player}"
			prefix_hist_acting = f"P:{acting_player},H:{self.env_aux.enc_player_hands[f'agent_{acting_player}']}->(P:{self.acting_player}"
			acting_history = history
			acting_history.replace(prefix_hist_learn,prefix_hist_acting)
			strategy = self.nodes[acting_history].get_strategy()
			utility = 0

			# Step ahead
			try:
				action = np.random.choice(self.nodes_state[infoSet].env.action_spaces[f"agent_{acting_player}"], p=strategy)
			except:
				action = self.nodes_state[infoSet].env.action_spaces[f"agent_{acting_player}"][0]

			action_probability = strategy[self.nodes_state[infoSet].env.action_spaces[f"agent_{acting_player}"].index(action)]
			next_acting_player = action
			done, next_observation_spaces, _, card, _ = self.nodes_state[infoSet].env.step(action)
			card = card.replace("_","")
			nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)

			probability_players[f"agent_{acting_player}"] *= action_probability

			# Update global env after state change
			self.env_aux = copy.deepcopy(self.nodes_state[infoSet].env)

			# recursion
			# recursion
			if self.nodes_state[infoSet].env.message_provided:

				utility = self.external_cfr(history+f",A:{action},C:{card})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players)
			else:
				## Another message needs to be provided
				agent_key = next_acting_player
				next_acting_player = self.random_order[self.nodes_state[infoSet].env.provide_message]
				hand = config.compute_hand_from_labels(self.nodes_state[infoSet].env.player_hands[f'agent_{learning_player}'])
				nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)
				utility = self.external_cfr_message(history+f",A:{action})*NEW-TURN*HAND:{hand}*->(P:{next_acting_player}",str(nextInfoSet), learning_player, next_acting_player,t, probability_players, agent_key)				






			return utility
		


	def external_cfr_message(self,history, infoSet, learning_player, acting_player, t, probability_players, agent_key):
		"""
			Decisions for message space
		"""
		#print('THIS IS ITERATION', t)


		message_action_space = self.env_aux.message_space


		# Build infoset by state
		if infoSet not in self.nodes_state:
			self.nodes_state[infoSet] = NodeState(len(message_action_space), copy.deepcopy(self.env_aux))
		else:
			self.nodes_state[infoSet].number += 1	
			# Reset to original infoset env to run analysis again
			self.nodes_state[infoSet].env = copy.deepcopy(self.nodes_state[infoSet].static_env)

		# Build infoset by history
		if history not in self.nodes:
			self.nodes[history] = NodeInfoSet(len(message_action_space), history)
		else:
			self.nodes[history].number += 1	



		if acting_player == learning_player:

			action_space_length = len(message_action_space)
			utility = np.zeros(action_space_length) 
			node_utility = 0
			strategy = self.nodes[history].get_strategy()

			for index, action in enumerate(message_action_space):
				# Reset to original infoset env to run analysis again
				self.nodes_state[infoSet].env = copy.deepcopy(self.nodes_state[infoSet].static_env)
				
				# Step ahead
				next_observation_spaces = self.nodes_state[infoSet].env.step_message(action)


				probability_players[f"agent_{learning_player}"] *= strategy[index]

				# Update global env after state change
				self.env_aux = copy.deepcopy(self.nodes_state[infoSet].env)

				if self.nodes_state[infoSet].env.message_provided:
					# If all messages were provided
					next_acting_player = agent_key
					nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)

					utility[index] = self.external_cfr(history+f",A:{action})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players)
				else:
					# Another message needs to be provided
					next_acting_player = self.random_order[self.nodes_state[infoSet].env.provide_message]
					nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)
					utility[index] = self.external_cfr_message(history+f",A:{action})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players, agent_key)
				
				node_utility += strategy[index] * utility[index]



			if learning_player==0:
				p_other = probability_players["agent_1"]*probability_players["agent_2"]
			elif learning_player==1:
				p_other = probability_players["agent_0"]*probability_players["agent_2"]
			else:
				p_other = probability_players["agent_0"]*probability_players["agent_1"]				

			for action in range(action_space_length):
				regret = utility[action] - node_utility
				self.nodes[history].regret_sum[action] += max(p_other*regret,0)

			for index_action in range(action_space_length):
				self.nodes[history].strategy_sum[index_action] += strategy[index_action]*t
				#print(regret)


			return node_utility

		else:  # Here is where self play is done

			action_space_length = len(message_action_space)

			prefix_hist_learn = f"P:{learning_player},H:{self.env_aux.enc_player_hands[f'agent_{learning_player}']}->(P:{self.acting_player}"
			prefix_hist_acting = f"P:{acting_player},H:{self.env_aux.enc_player_hands[f'agent_{acting_player}']}->(P:{self.acting_player}"
			acting_history = history
			acting_history.replace(prefix_hist_learn,prefix_hist_acting)
			strategy = self.nodes[acting_history].get_strategy()
			utility = 0

			# Step ahead

			action = np.random.choice(range(len(message_action_space)), p=strategy)
			action_probability = strategy[action]

			next_observation_spaces = self.nodes_state[infoSet].env.step_message(message_action_space[action])



			probability_players[f"agent_{acting_player}"] *= action_probability

			# Update global env after state change
			self.env_aux = copy.deepcopy(self.nodes_state[infoSet].env)

			# recursion
			if self.nodes_state[infoSet].env.message_provided:

				## If all messages were provided
				next_acting_player = agent_key 
				nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)
				utility = self.external_cfr(history+f",A:{message_action_space[action]})->(P:{next_acting_player}",str(nextInfoSet), learning_player, next_acting_player,t, probability_players)
			else:
				## Another message needs to be provided
				next_acting_player = self.random_order[self.nodes_state[infoSet].env.provide_message]
				nextInfoSet = self.nodes_state[infoSet].env.create_observation(next_acting_player, next_observation_spaces)
				utility = self.external_cfr_message(history+f",A:{message_action_space[action]})->(P:{next_acting_player}",str(nextInfoSet), learning_player, next_acting_player,t, probability_players, agent_key)				



			return utility
