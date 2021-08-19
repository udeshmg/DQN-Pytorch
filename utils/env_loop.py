from itertools import count
import torch

class EnvLoop():
    def __init__(self, agent, env, target_update_period, device='cuda', tensorboard=None):
        self.agent = agent
        self.env = env
        self.target_update_period = target_update_period
        self.device = device
        self.writer = tensorboard

    def loop(self, num_episodes):
        episode_durations = []
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            episode_return = 0
            for t in count():
                # Select and perform an action
                action = self.agent.select_action(state.unsqueeze(0))
                next_state, reward_original, done, info = self.env .step(action.item())
                episode_return += reward_original
                reward = torch.tensor([reward_original], device=self.device)

                # Observe new state
                if done:
                    next_state = None

                # Store the transition in memory
                # covert state and next state to tensors
                if next_state is not None:
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                # Remember
                self.agent.remember(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.agent.learn()
                if done:
                    episode_durations.append(t + 1)
                    if self.writer is not None:
                        self.writer.add_scalar("Env/reward", episode_return, i_episode)
                    print("Episode: {}, {:.2f}, {:.2f}".format(i_episode, episode_return, reward_original))
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update_period == 0:
                self.agent.update_target()

