RL Debugging
Hyperparameters
- [ ] Increase Gamma
- [ ] Decrease Epsilon slower 
- [ ] Learning rate under 1e-4
- [ ] Large Batch sizes for noise reduction
- [ ] Increase buffer size
Algorithm
- [ ] Reduce Clipping
- [ ] Detect anomaly (Davit Whatsapp)
- [ ] Check for Dying RELU -> lower learning rate
- [ ] Vanishing exploding gradients -> Reduce number of layer
- [ ] Vanishing or exploding activations -> layer batch normalisation
- [x] Shrinking learning rate with epsilon

Measurements to Output for debugging
- [ ] Loss value
- [ ] action space entropy?
- [ ] KL-Divergence
- [x] Episode value -> How long does the agent live 

https://medium.com/swlh/deep-rl-debugging-and-diagnostics-5c9a17e78653

Repos to steal state_to_features method for testing:
- [ ] https://github.com/nickstr15/bomberman/blob/master/agent_code/maverick/ManagerFeatures.py
- [ ] 
- [ ] 
Tensorflow implementations online
- [ ] https://github.com/Borzyszkowski/RL-Bomberman-Gradient/blob/master/pommerman/agents/dqn_agent.py
