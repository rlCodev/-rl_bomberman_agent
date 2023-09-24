# Train q-learning coin collector agent:
`python main.py play --no-gui --agents ql_agent --train 1 --scenario coin-heaven`
# Run Round with q-learning coin collector agent:
`python main.py play --agents random_agent ql_agent --no-gui`

python main.py play --agents dqn_hyper_agent random_agent rule_based_agent coin_collector_agent --scenario coin-heaven --train 1 --no-gui --n-rounds 1000