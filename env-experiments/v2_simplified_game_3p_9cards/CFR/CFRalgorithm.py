import numpy as np
import copy
from tqdm import tqdm
import config
import pickle
import os




class NodeState:
	def __init__(self, num_actions):
		self.regret_sum = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.strategy_sum = np.zeros(num_actions)
		self.num_actions = num_actions
		self.number = 1
	

	
	
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
	def __init__(self, iterations, nodes, nodes_state):
		self.iterations = iterations
		self.nodes_state = nodes_state
		self.nodes = nodes
		self.env = TempleOfHorror()
		self.acting_player = None
		self.agents_order = [0,1,2]
		self.iteration = 0


	def save_checkpoint(self, checkpoint_dir='checkpoints'):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		checkpoint_path = os.path.join(checkpoint_dir, f'cfr_checkpoint_{self.iteration}.pkl')
		with open(checkpoint_path, 'wb') as f:
			pickle.dump(self, f)
		print(f"Checkpoint saved to {checkpoint_path}")

	@staticmethod
	def load_checkpoint(checkpoint_path):
		with open(checkpoint_path, 'rb') as f:
			return pickle.load(f)

	def cfr_iterations_external(self, attackers_are_truthful=True):
		average_utilities = np.zeros((self.iterations, self.env.N))
		cumulative_utility = np.zeros(self.env.N)
		start_iteration = self.iteration + 1

		for t in tqdm(range(start_iteration, self.iterations + 1)):
			self.iteration = t
			observation = self.env.reset() 
			self.random_order = self.agents_order.copy()
			self.acting_player = self.random_order[0]


			for learning_player in [0,1,2]: # Players
				self.acting_player = 0 

				probability_players = {agent: 1.0 for agent in self.env.agents} 
				env_instance = copy.deepcopy(self.env)
				infoSet = self.env.create_observation(learning_player, observation)
				hand = config.compute_hand_from_labels(env_instance.static_player_hands[f'agent_{learning_player}'])

				cumulative_utility[learning_player] += self.external_cfr_message(f"P:{learning_player},R:{env_instance.player_role[f'agent_{learning_player}'][:-2]},C:{hand}GameInits->(P:{self.acting_player}", str(infoSet),  learning_player,  self.acting_player, t, probability_players, env_instance, attackers_are_truthful=attackers_are_truthful)
				#print(player, utility[player])                       

			average_utilities[t-1][:] = cumulative_utility.copy()/t

			if t % 5000 == 0:
				self.save_checkpoint()


		print('Average game value 0: {}'.format(cumulative_utility[0]/(self.iterations)))
		print('Average game value 1: {}'.format(cumulative_utility[1]/(self.iterations)))
		print('Average game value 2: {}'.format(cumulative_utility[2]/(self.iterations)))

		return average_utilities
		
	  

	def external_cfr(self, history, infoSet, learning_player, acting_player, t, probability_players, env):

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
		#print('THIS IS ITERATION', t)

		# Build infoset
		if infoSet not in self.nodes_state:
			self.nodes_state[infoSet] = NodeState(len(env.action_spaces[f"agent_{acting_player}"]))
		else:
			self.nodes_state[infoSet].number += 1	



		done, winner = env.referee()

		# History is in a terminal state then calculate payments
		if done:		

			message_history = env.message_history
			
			# Extract fire and gold messages for each agent
			fire_messages = np.array([
				message_history[0] > 0,  # agent 0
				message_history[2] > 0,  # agent 1
				message_history[4] > 0   # agent 2
			])
			
			gold_messages = np.array([
				message_history[1] > 0,  # agent 0
				message_history[3] > 0,  # agent 1 
				message_history[5] > 0   # agent 2 
			])
			
			# Check if agents actually have the cards they claimed
			has_fire = np.array([
				2 in env.enc_player_hands[f"agent_{i}"]
				for i in range(3)
			])
			
			has_gold = np.array([
				3 in env.enc_player_hands[f"agent_{i}"]
				for i in range(3)
			])
			
			# Calculate lying: claimed to have card but doesn't actually have it
			lying_about_fire = fire_messages & ~has_fire
			lying_about_gold = gold_messages & ~has_gold
			
			# Combined lying vector: 1 if lied about either fire or gold, 0 otherwise
			lying_vector = (lying_about_fire | lying_about_gold).astype(int)		

			lying_factor = 1

			lying_payoff = lying_vector*lying_factor

			if env.enc_player_role[f"agent_{acting_player}"] == 1: ## If player is a Defender

				if env.enc_player_role[f"agent_{acting_player}"] == winner: ## If player won

					if env.enc_player_role[f"agent_{learning_player}"] == 1: ## Learning Defenders win
						return 2 + lying_payoff[learning_player]
					else:
						return - 2 - lying_payoff[learning_player]
				else:

					if env.enc_player_role[f"agent_{learning_player}"] == 1: ## Learning Defenders lose
						return  - 2 - lying_payoff[learning_player]
					else:
						return  2 + lying_payoff[learning_player]
					
			else: ## If player is an Attacker 
				if env.enc_player_role[f"agent_{acting_player}"] == winner:

					if env.enc_player_role[f"agent_{learning_player}"] == 1: ## Learning Defenders lose
						return -2 - lying_payoff[learning_player]
					else:
						return 2 + lying_payoff[learning_player]

				else:

					if env.enc_player_role[f"agent_{learning_player}"] == 1: ## Learning Defenders win
						return 2 + lying_payoff[learning_player]
					else:
						return -2 - lying_payoff[learning_player]
					
 
				
		## Learning procedure
		if acting_player == learning_player:
			if history not in self.nodes:
				self.nodes[history] = NodeInfoSet(len(env.action_spaces[f"agent_{acting_player}"]), history)
			else:
				self.nodes[history].number += 1	
				
			action_space_length = len(env.action_spaces[f"agent_{acting_player}"])
			utility = np.zeros(action_space_length) 
			node_utility = 0
			strategy = self.nodes[history].get_strategy()

			for index, action in enumerate(env.action_spaces[f"agent_{acting_player}"]):
				# Reset to original infoset env to run analysis again
				next_env = copy.deepcopy(env)
				
				# Step ahead
				next_acting_player = action
				done, next_observation_spaces, _, card, _ = next_env.step(action)
				nextInfoSet = next_env.create_observation(next_acting_player, next_observation_spaces)


				probability_players[f"agent_{acting_player}"] *= strategy[index]


				utility[index] = self.external_cfr(history+f",A:{action},C:{card})->(P:{next_acting_player}",str(nextInfoSet), learning_player, next_acting_player,t, probability_players, next_env)
				
				
				node_utility += strategy[index] * utility[index]



			if learning_player==0:
				p_other = probability_players["agent_1"]*probability_players["agent_2"]
			elif learning_player==1:
				p_other = probability_players["agent_0"]*probability_players["agent_2"]
			else:
				p_other = probability_players["agent_0"]*probability_players["agent_1"]				

			# Update regrets
			for action in range(action_space_length):
				regret = utility[action] - node_utility
				self.nodes[history].regret_sum[action] += max(p_other*regret,0)


			for index_action in range(action_space_length):
				self.nodes[history].strategy_sum[index_action] += probability_players[f"agent_{acting_player}"]*strategy[index_action]*t


			return node_utility

		else:  # Here is where self play is done

			action_space_length = len(env.action_spaces[f"agent_{acting_player}"])

			hand_learn = config.compute_hand_from_labels(env.static_player_hands[f'agent_{learning_player}'])
			prefix_hist_learn = f"P:{learning_player},R:{env.player_role[f'agent_{learning_player}'][:-2]},C:{hand_learn}"
			hand_acting = config.compute_hand_from_labels(env.static_player_hands[f'agent_{acting_player}'])
			prefix_hist_acting = f"P:{acting_player},R:{env.player_role[f'agent_{acting_player}'][:-2]},C:{hand_acting}"
			acting_history = history
			acting_history = acting_history.replace(prefix_hist_learn, prefix_hist_acting)


			# Build infoset by history
			if acting_history not in self.nodes:
				self.nodes[acting_history] = NodeInfoSet(action_space_length, acting_history)
			else:
				self.nodes[acting_history].number += 1	

			strategy = self.nodes[acting_history].get_strategy()
			utility = 0

			# Step ahead
			try:
				action = np.random.choice(env.action_spaces[f"agent_{acting_player}"], p=strategy)
			except:
				action = env.action_spaces[f"agent_{acting_player}"][0]

			action_probability = strategy[env.action_spaces[f"agent_{acting_player}"].index(action)]
			next_acting_player = action
			next_env = copy.deepcopy(env)
			done, next_observation_spaces, _, card, _ = next_env.step(action)

			nextInfoSet = next_env.create_observation(next_acting_player, next_observation_spaces)

			probability_players[f"agent_{acting_player}"] *= action_probability

			# recursion

			utility = self.external_cfr(history+f",A:{action},C:{card})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players, next_env)




			return utility
		


	def external_cfr_message(self,history, infoSet, learning_player, acting_player, t, probability_players, env, attackers_are_truthful=True):
		"""
			Decisions for message space
		"""
		#print('THIS IS ITERATION', t)

		# If Attackers are truthful
		if "attacker" in env.player_role[f'agent_{acting_player}'] and attackers_are_truthful:
			# number of fire and gold in acting player hand
			number_fire = env.enc_player_hands[f'agent_{acting_player}'].count(2)
			number_gold = env.enc_player_hands[f'agent_{acting_player}'].count(3)
			# number of fire and gold in learning player hand
			attacker_message_space = [(number_fire,number_gold)]
			message_action_space = attacker_message_space
		else:
			message_action_space = env.message_space

		# Build infoset by state
		if infoSet not in self.nodes_state:
			self.nodes_state[infoSet] = NodeState(len(message_action_space))
		else:
			self.nodes_state[infoSet].number += 1	





		if acting_player == learning_player:
					# Build infoset by history
			if history not in self.nodes:
				self.nodes[history] = NodeInfoSet(len(message_action_space), history)
			else:
				self.nodes[history].number += 1	

			action_space_length = len(message_action_space)
			utility = np.zeros(action_space_length) 
			node_utility = 0
			strategy = self.nodes[history].get_strategy()

			for index, action in enumerate(message_action_space):
				# Reset to original infoset env to run analysis again
				next_env = copy.deepcopy(env)
				
				# Step ahead
				next_observation_spaces = next_env.step_message(action)


				probability_players[f"agent_{learning_player}"] *= strategy[index]

				if next_env.message_provided:

					next_acting_player = 0 
					nextInfoSet = next_env.create_observation(next_acting_player, next_observation_spaces)

					utility[index] = self.external_cfr(history+f",A:{action})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players, next_env)
				else:
					next_acting_player = self.random_order[next_env.provide_message]

					nextInfoSet = next_env.create_observation(next_acting_player, next_observation_spaces)
					utility[index] = self.external_cfr_message(history+f",A:{action})->(P:{next_acting_player}", str(nextInfoSet), learning_player, next_acting_player,t, probability_players, next_env, attackers_are_truthful=attackers_are_truthful)
				
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
				self.nodes[history].strategy_sum[index_action] += probability_players[f"agent_{acting_player}"]*strategy[index_action]*t
				#print(regret)


			return node_utility

		else:  # Here is where self play is done

			action_space_length = len(message_action_space)


			hand_learn = config.compute_hand_from_labels(env.static_player_hands[f'agent_{learning_player}'])
			prefix_hist_learn = f"P:{learning_player},R:{env.player_role[f'agent_{learning_player}'][:-2]},C:{hand_learn}"
			hand_acting = config.compute_hand_from_labels(env.static_player_hands[f'agent_{acting_player}'])
			prefix_hist_acting = f"P:{acting_player},R:{env.player_role[f'agent_{acting_player}'][:-2]},C:{hand_acting}"
			acting_history = history

			acting_history = acting_history.replace(prefix_hist_learn, prefix_hist_acting)



			# Build infoset by history
			if acting_history not in self.nodes:
				self.nodes[acting_history] = NodeInfoSet(action_space_length, acting_history)
			else:
				self.nodes[acting_history].number += 1	
	
			strategy = self.nodes[acting_history].get_strategy()
			utility = 0

			# Step ahead

			action = np.random.choice(range(len(message_action_space)), p=strategy)
			action_probability = strategy[action]

			next_env = copy.deepcopy(env)
			next_observation_spaces = next_env.step_message(message_action_space[action])



			probability_players[f"agent_{acting_player}"] *= action_probability

			# recursion
			if next_env.message_provided:
				## If all messages were provided
				next_acting_player = 0 
				nextInfoSet = next_env.create_observation(next_acting_player, next_observation_spaces)
				utility = self.external_cfr(history+f",A:{message_action_space[action]})->(P:{next_acting_player}",str(nextInfoSet), learning_player, next_acting_player,t, probability_players, next_env)
			else:
				## Another message needs to be provided
				next_acting_player = self.random_order[next_env.provide_message]
				nextInfoSet = next_env.create_observation(next_acting_player, next_observation_spaces)
				utility = self.external_cfr_message(history+f",A:{message_action_space[action]})->(P:{next_acting_player}",str(nextInfoSet), learning_player, next_acting_player,t, probability_players, next_env, attackers_are_truthful=attackers_are_truthful)				



			return utility
