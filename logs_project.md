

21Apr2025: Started to work in the original game to create a MARL algorithm. 
01May2025: Create a new simplified game v02 whee I implement latest game with empty chamber cards. 
An idea that I have is to create an agent that records history of play with an lstm and shares the hidden output to the agents so they know where they are positioned. This in addition of an LSTM for each actor and critic. 
Maybe I can implement this agent later on. I implement the first MARL solution in MARL v1. 
I was able to create the env by now. I have to train the agents now. 
11May2025: simplified game v2 is biased for defenders to win 68% of the time assuming everyone acts randomly. But if I assume Attackers win in the end condition then winning rate drops to 52%. This is usually what will happen if they play well. 
Added end game winner in case 2 Gold 1 Fire 4 Empty. 
I added -1 as a symbol that a message hasn't been sent by each agent. 
13May2025
Decided to code a CFR for 4 card game 4 players and rounds. 
