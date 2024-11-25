import random
import numpy as np
import torch

class TempleOfHorror():
    def __init__(self, N ):
        super().__init__()
        assert N >= 3
        assert N <= 6
        self.N = N

        self.empty_cards = 8 + (self.N - 3)*4
        self.gold_cards = 5 + (self.N - 3) 
        self.fire_cards = 2          
        self.aventurers = 2 + int(self.N/3) if self.N > 3  else 2
        self.hunters = 2
        self.done = None
        self.number_plays = None
        self.score = None
        self.turns = None

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
        self.enc_player_role = None

        # Action Space. Index position in observation vector
        self.action_spaces = None

        # Observations Spaces for all the agents
        self.observation_spaces = None

        # Observations Messages
        self.message_space = None



    def set_action_space(self):
       ### Fix action space
        for agent in self.agents:
            for e, _agent in enumerate(self.agents):
                if agent != _agent:
                    self.action_spaces[agent].append(e)


    def set_observation_space(self):
        

        # First space are closed cards. Second space are open cards
        self.observation_spaces = {agent:[0,0,0] for agent in self.agents}
               


    def sample_roles(self):
        aux = self.deck_roles.copy()
        #Sample Roles
        for agent in self.agents:
            role = random.sample(aux , 1)[0]
            self.player_role[agent] = role
            aux.remove(role)
        
        self.enc_player_role = self.encode_roles()

    def sample_deck(self):
        #Sample from deck 
        aux = self.deck_cards.copy()
        for agent in self.agents:
            sampled_items = random.sample(aux, 5-self.turns)
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
            Encodes players' roles. Encoding rule is: Hunter: 1, Adventurer: 0. 
            Output:
                encoded_role (dict): encodings by agent. 
        """
        encoded_roles = {}
        encoding = {"adventurer": 0, "hunter": 1}
        for agent in self.agents:
            encoded_roles[agent] = encoding[self.player_role[agent].split("_")[0] ]

        return encoded_roles
    
    
    def reward(self,agent, winner):
        """
        Reward function for each agent.
        """

        if self.enc_player_role[agent] == winner:
                return 100
        else:
                return -100

            

    def create_state(self, player_number, state):


        final = state["public_message"].copy()
        final.append(int(state["turn"]))
        final.append(state["private_role"][f"agent_{player_number}"])
        final += state["private_information"][f"agent_0"].copy()
        final += state["private_information"][f"agent_1"].copy()
        final += state["private_information"][f"agent_2"].copy()
        final.append(state["private_role"][f"agent_{player_number}"])
        final.append(state["score"]["gold"])
        final.append(state["score"]["fire"])   
        final.append(state["score"]["empty"]) 
        hand = self.enc_player_hands[f"agent_{player_number}"]
        final.append(hand.count(3))
        final.append(hand.count(2))
        final.append(hand.count(1))


 
        return torch.Tensor(final)
            
    def referee(self):
        """
        Method that validates who won.
        """

        if self.score["gold"] == 4:
            return True, 0
        elif  self.score["fire"] == 2:
            return True, 1
        else:
            return False, None
        
    def random_message(self, prob_lie):
        """
            Randomizes a message intially
        """
        player_hands = {}
        #Sample from deck 
        aux = self.deck_cards.copy()
        for agent in self.agents:
            sampled_items = random.sample(aux, 5-self.turns)
            player_hands[agent] = sampled_items
            aux = [card for card in aux if card not in sampled_items]

        enc_player_hands_fake = self.encode_hand(player_hands)


        for agent in self.agents:
            sample = random.random()
            if sample < prob_lie:
                hand = enc_player_hands_fake[agent]
            else:

                hand = self.enc_player_hands[agent]

            gold = hand.count(3)
            fire = hand.count(2)  
            empty = hand.count(1)  

            message = [empty, fire, gold] 
            self.message_space += message             


        
    def reset(self):
        # Define the agents


        #Numer of cards in the deck
        self.deck_roles = [f"adventurer_{i+1}" for i in range(self.aventurers)] + \
                  [f"hunter_{i+1}" for i in range(self.hunters)]

        self.deck_cards = [f"gold_{i+1}" for i in range(self.gold_cards)] + \
                        [f"empty_{i+1}" for i in range(self.empty_cards)] + \
                        [f"fire_{i+1}" for i in range(self.fire_cards)]
        
        # Sample from deck       
        self.player_hands = {agent:None for agent in self.agents}


        # Sample roles
        self.player_role = {agent:None for agent in self.agents}

        self.turns = 0
        self.number_plays = 0
        self.score = {"gold": 0, "fire": 0, "empty": 0}
        self.action_spaces = {agent:[] for agent in self.agents}
        self.done = False

        self.sample_roles()

        self.sample_deck()

        self.set_action_space()

        self.set_observation_space()



        # Random Message Creation 
        self.message_space = []
        self.random_message( .01)


        return {"private_information": self.observation_spaces,
                "public_message": self.message_space, 
                "private_role": self.enc_player_role,
                "turn": self.turns,
                "score": self.score}
  
    def reset_turn_end(self):
        self.turns += 1  
        self.number_plays = 0

        # Sample from deck       
        self.player_hands = {agent:None for agent in self.agents}

        self.action_spaces = {agent:[] for agent in self.agents}

        self.sample_deck()

        self.set_action_space()

        self.set_observation_space()


        # Random Message Creation 
        self.message_space = []
        self.random_message( .50)


        return {"private_information": self.observation_spaces,
                "public_message": self.message_space, 
                "private_role": self.enc_player_role,
                "turn": self.turns,
                "score": self.score}


    def step(self, action):
        self.number_plays += 1
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
            #print(card)
            # Erase card from Deck
            self.deck_cards.remove(card)

            # Increase score
            self.score[card.split("_")[0]] += 1
       
            # Update Observations  
            if "gold" in card:
                self.observation_spaces[f"agent_{agent_number}"][0] += 1
            elif "fire" in card:
                self.observation_spaces[f"agent_{agent_number}"][1] += 1  
            else:
                self.observation_spaces[f"agent_{agent_number}"][2] += 1                          
    
            # Update Possible actions. Remove Agent option if he has no cards left
            if sum(self.observation_spaces[f"agent_{agent_number}"]) == 5-self.turns:
                for agent in self.agents: 
                    if agent != f"agent_{agent_number}":
                        self.action_spaces[agent].remove(agent_number)


        # Open Card (Update Action Space and Observation State)
        open_card(action)
        # See if self.done. 
        self.done, winner = self.referee()
        # Compute Reward For all agents
    


        if self.done:
            rewards = [self.reward(agent, winner) for agent in self.agents]
        else:
            rewards = None


        if self.done:
            if winner == 1:
                #print("Hunter won", self.score)
                #print(self.observation_spaces)
                None
            else:
                #print("Adventurer won", self.score)              
                #print(self.observation_spaces)
                None


        



        state = {"private_information": self.observation_spaces,
                "public_message": self.message_space, 
                "private_role": self.enc_player_role,
                "turn": self.turns,
                "score": self.score}
        
        if not self.done and self.number_plays == self.N:
            self.reset_turn_end()
            if self.turns == 3:
                self.done = True
                winner = 1

                if self.done:
                    rewards = [self.reward(agent, winner) for agent in self.agents]
                else:
                    rewards = None


            if self.done:
                if winner == 1:
                    #print("Hunter won", self.turns)
                    None
                else:
                    #print("Adventurer won", self.turns)                
                    #print(self.observation_spaces)
                    None

        
        return self.done, state, rewards