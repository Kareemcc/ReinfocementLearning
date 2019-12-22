batch_size = 64
gamma = 0.999 # discount factor in bellman equation
eps_start = 1 # epsilon Greedy
eps_end = 0.01
eps_decay = 0.001
target_update = 10 # frequency of updating target network weights
memory_size = 100_000
lr = 0.001 # Learning rate
num_episodes = 1000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_action_available(), device)
memory = ReplayMemory(memory_size)


policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.num_action_available()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width(), em.num_action_available()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


episode_durations = []
total_rewards = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = Qvalues.get_current(policy_net, states, actions)
            next_q_values = Qvalues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()
