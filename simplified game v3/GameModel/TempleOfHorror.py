### Temple of Horror Game Model
### Env MARL Definition:
### 1. Agents: 4 from 2 defender and 3 attackers
### 2. Actions: 
###    - choose a player that open a card (4 cards per agent)
###    - Beginning: Provide a message (number of gold and fire cards)
### 3. Observations:
###    - Public message
###    - table information
###    - score
###    - player hand
###    - player number
###    - number of rounds
### 4. Rewards:
###    - 100 for defender if he wins
###    - 100 for attacker if he wins
###    - -100 for defender if he loses
###    - -100 for attacker if he loses
### 5. End of game:
###    - 4 gold cards, OR
###    - 2 fire cards
### 6. Deck:
###    - 10 empty cards
###    - 4 gold cards
###    - 2 fire cards



import random
import numpy as np
import torch

class TempleOfHorror():
    def __init__(self):
        super().__init__()

        self.N = 4

        self.empty_cards = 10
        self.gold_cards = 4
        self.fire_cards = 2        
        self.attacker = 3
        self.defender = 2
        self.done = None
        self.score = None
        self.provide_message = 0
        self.message_provided = None

        # Define the agents

        self.agents = [f"agent_{i}" for i in range(self.N)]

        #Numer of cards in the deck
        self.deck_roles = None
        self.deck_cards = None
        
        # Sample from deck       
        self.player_hands = None
        self.enc_player_hands = None

        # Sample roles
        self.player_role = None
        self.player_number = None
        self.enc_player_role = None

        # Action Space. Index position in observation vector
        self.action_spaces = None

        # Observations Spaces for all the agents
        self.public_observation = None

        # Observations Messages
        self.message_space = None
        self.message_history = None

        self.history = ""
        self.round = None
        self._initial_cards = 4
        self.number_cards = None
        self.turns = 0
        self.agent_key = None


    def init_message_space(self):
        """
        Initialize the message space for each agent.
        The message space is a list of two integers representing all combinations of number of fire, and gold cards respectively.
        """
        for num_gold in range(0, 4):
            for num_fire in range(0, 2):
                if num_gold + num_fire <= self.number_cards:
                    self.message_space.append((num_fire, num_gold))



    def init_action_space(self):
       ### Fix action space
        for agent in self.agents:
            for e, _agent in enumerate(self.agents):
                if agent != _agent:
                    self.action_spaces[agent].append(e)

    def init_public_observation(self):
        """
        Initialize the public observation space for each agent.
        The public observation space is a dictionary where each agent's key maps to a list of three integers.
        The three integers represent the number of empty, fire, and gold cards respectively.
        """
        self.public_observation = {agent:[0,0,0] for agent in self.agents}
               

    def sample_roles(self):

        aux = self.deck_roles.copy()
        #Sample Roles
        for agent in self.agents:
            role = random.sample(aux , 1)[0]
            self.player_role[agent] = role
            aux.remove(role)

        self.player_number = {self.player_role[agent_no]: int(agent_no.split("_")[1]) for agent_no in self.player_role.keys()}

        self.enc_player_role = self.encode_roles()

    def sample_deck(self):
        #Sample from deck 
        aux = self.deck_cards.copy()
        for agent in self.agents:
            sampled_items = random.sample(aux, self.number_cards)
            self.player_hands[agent] = sampled_items
            aux = [card for card in aux if card not in sampled_items]
        
        self.enc_player_hands = self.encode_hand(self.player_hands)     

    def encode_hand(self, hands):
        """
            Encodes players' hand. Encoding rule is: Gold: 3, Fire:2, Empty:1, Unknown:0.
            Output:
                encoded_hand (dict): encodings by agent. 
        """
        encoded_hand = {}
        encoding = {"gold": 3, "fire": 2, "empty": 1 }
        for agent in self.agents:
            encoded_hand[agent] = [ encoding[card.split("_")[0]] for card in hands[agent]]

        return encoded_hand
        
    def encode_roles(self):
        """
            Encodes players' roles. Encoding rule is: defender: 1, attacker: 0. 
            Output:
                encoded_role (dict): encodings by agent. 
        """
        encoded_roles = {}
        encoding = {"attacker": 0, "defender": 1}
        for agent in self.agents:
            encoded_roles[agent] = encoding[self.player_role[agent].split("_")[0] ]

        return encoded_roles
    
    
    def reward(self,agent, winner):
        """
        Reward function for each agent.
        """

        if self.enc_player_role[agent] == winner:
            return 5
        else:
            return -5

            

    def create_observation(self, player_number, world_state):
        """
        Inputs:
            - player_number (str): Player number
            - world_state (dict): State of the game.
        Outputs:
            For each player state includes:
            - Public message
            - Private information (number of cards)
            - Score
            - Player hand
            - Player number
        """
        final = []
        final += world_state["public_message"].copy()
        final += world_state["table_information"][f"agent_0"].copy()
        final += world_state["table_information"][f"agent_1"].copy()
        final += world_state["table_information"][f"agent_2"].copy()
        final.append(world_state["score"]["empty"]) 
        final.append(world_state["score"]["fire"]) 
        final.append(world_state["score"]["gold"])
        hand = self.enc_player_hands[f"agent_{player_number}"]
        final.append(hand.count(1))
        final.append(hand.count(2))
        final.append(hand.count(3))
        final.append(int(player_number))
        final.append(int(self.round))

 
        return torch.tensor(final)
            
    def referee(self):
        """
        Method that validates who won.
        """

        if self.score["gold"] == 4:

            #print("*******************")
            #print("** ATTACKERS WIN **")
            #print("*******************")
            return True, 0
        elif  self.score["fire"] == 2:
            #print("*******************")
            #print("** DEFENDERS WIN **")
            #print("*******************")
            return True, 1
        else:
            return False, None
        
        
        
    def reset(self):
        # Define the agents

        self.round = 0
        self.number_cards = self._initial_cards

        #Numer of cards in the deck
        self.deck_roles = [f"attacker_{i+1}" for i in range(self.attacker)] + \
                  [f"defender_{i+1}" for i in range(self.defender)]

        self.deck_cards = [f"gold_{i+1}" for i in range(self.gold_cards)] + \
                        [f"empty_{i+1}" for i in range(self.empty_cards)] + \
                        [f"fire_{i+1}" for i in range(self.fire_cards)]
        
        # Sample from deck       
        self.player_hands = {agent:None for agent in self.agents}


        # Sample roles
        self.player_role = {agent:None for agent in self.agents}
        self.player_number = None

        self.score = {"gold": 0, "fire": 0, "empty": 0}
        self.action_spaces = {agent:[] for agent in self.agents}
        self.done = False
        self.message_space = []

        self.sample_roles()

        self.sample_deck()

        self.init_action_space()

        self.init_message_space()

        self.init_public_observation()

        self.history = ""
        self.provide_message = 0

        self.message_provided = False
        # Random Message Creation 
        self.message_history = [-1]*(2*self.N)
        self.turns = 0





        return {"table_information": self.public_observation,
                "public_message": self.message_history, 
                "score": self.score}
  
    def reset_turn(self):
        # Define the agents

        self.round += 1
        self.number_cards -= 1 

        
        # Sample from deck       
        self.player_hands = {agent:None for agent in self.agents}


        self.action_spaces = {agent:[] for agent in self.agents}
        self.done = False
        self.message_space = []



        self.sample_deck()

        self.init_action_space()

        self.init_message_space()

        self.init_public_observation()


        self.provide_message = 0
        self.message_provided = False
        # Random Message Creation 
        self.message_history = [-1]*8






        return {"table_information": self.public_observation,
                "public_message": self.message_history, 
                "score": self.score}
    
    def step_message(self, action_message):
        """
            Possible messages are number of gold and fire cards. 
        """
        self.history += str(action_message)
        self.message_history[self.provide_message*2] = action_message[0]
        self.message_history[self.provide_message*2+1] = action_message[1]
        self.provide_message += 1

        if self.provide_message == self.N:
            self.message_provided = True
            self.provide_message = 0

        state = {"table_information": self.public_observation,
                "public_message": self.message_history, 
                "score": self.score}

        return state      


    def step(self, action):
        self.history += str(action)

        def open_card(agent_number):
            """
            Open new card. Updates observables and action space for each agent. 
            Input:
                number (int): Card index.
            Output:
                new_card (int): encoded card
            """

            card = np.random.choice(self.player_hands[f"agent_{agent_number}"])
            self.player_hands[f"agent_{agent_number}"].remove(card)
            if "gold" in card:
                self.enc_player_hands[f"agent_{agent_number}"].remove(3)
            elif "fire" in card:
                self.enc_player_hands[f"agent_{agent_number}"].remove(2)
            else:
                self.enc_player_hands[f"agent_{agent_number}"].remove(1)
            #print(card)
            # Erase card from Deck
            self.deck_cards.remove(card)

            # Increase score
            self.score[card.split("_")[0]] += 1
       
            # Update Observations  
            if "gold" in card:
                self.public_observation[f"agent_{agent_number}"][2] += 1
            elif "fire" in card:
                self.public_observation[f"agent_{agent_number}"][1] += 1  
            else:
                self.public_observation[f"agent_{agent_number}"][0] += 1                          
    
            # Update Possible actions. Remove Agent option if he has no cards left
            if sum(self.public_observation[f"agent_{agent_number}"]) == self.number_cards:
                for agent in self.agents: 
                    if agent != f"agent_{agent_number}":
                        self.action_spaces[agent].remove(agent_number)
            return card

        # Open Card (Update Action Space and Observation State)
        card = open_card(action)
        self.turns += 1
        # See if self.done. 
        self.done, winner = self.referee()
        # Compute Reward For all agents
    

        if self.done:
            rewards = [self.reward(agent, winner) for agent in self.agents]

            state = {"table_information": self.public_observation,
                "public_message": self.message_history, 
                "score": self.score}
            
            return self.done, state, rewards, card[:-2], winner
        else:
            rewards = []
            for agent in self.player_role.keys():
                if "fire" in card[:-2]  and self.player_role[agent][:-2] == "defender": 
                    rewards.append(1)
                elif "gold" in card and self.player_role[agent][:-2] == "attacker":
                    rewards.append(1)
                elif "empty" in card:
                    rewards.append(0)
                else:
                    rewards.append(-1)

            if self.turns == self.N:

                self.turns = 0
                self.reset_turn()

            state = {"table_information": self.public_observation,
                    "public_message": self.message_history, 
                    "score": self.score}
        

            return self.done, state, rewards, card[:-2], winner