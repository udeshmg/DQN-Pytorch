import os.path

from networks import FeedForward
from agents import DQN
from external_env.vehicle_controller.vehicle_env_mp import Vehicle_env_mp
from utils import EnvLoop, Monitor_save_step_data, paths
from torch.utils.tensorboard import SummaryWriter

import argparse
import json
import torch

def createNextFileWriter(log_dir, suffix):
    id = paths.find_next_path_id(log_dir, suffix) + 1
    train_log_dir = log_dir + suffix + "_"+ str(id)
    print("Logs at: ", train_log_dir)
    return train_log_dir

def createNextFileName(tensorboard_log_dir, suffix):
    id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
    return  tensorboard_log_dir + suffix + "_"+ str(id)

def run(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = createNextFileWriter("./logs/", "DQN")
    writer = SummaryWriter(log_dir=path)

    with open(os.path.join(path, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)


    # setup network
    network = FeedForward(num_layers=3, input_dim=3, embed_dim=128, outputs=opts.num_actions,
                          device=device).to(device)
    # setup agent
    agent = DQN(num_actions=opts.num_actions, network=network, gamma=0.98, batch_size=256, lr=opts.lr,
                n_step=opts.n_steps, replay_mem_size=50000, eps_start=1, eps_end=0.02, eps_decay=0.99,
                device=device, tensorboard=writer)
    # setup env
    env = Vehicle_env_mp(id=2, num_actions=opts.num_actions, multi_objective=opts.multi_objective,
                         lexicographic=opts.lexicographic, front_vehicle=opts.front_vehicle, use_smarts=False)
    env = Monitor_save_step_data(env, step_data_file=os.path.join(path, "episode_data.csv"))

    loop = EnvLoop(agent=agent, env=env, target_update_period=1, device=device, tensorboard=writer)
    loop.loop(2000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent to connect with SMARTS")

    parser.add_argument('--controller', default='RL', help="Available controllers (RL, Gurobi, Heuristic)")
    parser.add_argument('--multi_objective', action='store_true', help="if true use MD-DQN")
    parser.add_argument('--lexicographic', action='store_true', help="if true use TL-DQN")
    parser.add_argument('--log_dir', default='../../logs/', help='Directory to write TensorBoard information to')
    parser.add_argument('--output_dir', default='../../outputs/', help='Directory to write output models to')
    parser.add_argument('--pretrained', help='pretrained file location')
    parser.add_argument('--eval_only', help='Evaluation only, no training')
    parser.add_argument('--discounts', nargs='+', default=[1, 1, 1],
                        help="Discount factors for each objective")
    parser.add_argument('--n_steps', default=5, type=int, help='N-Step Q-learning value')
    parser.add_argument('--lr', default=1e-3, help='Learning Rate for Adam')

    parser.add_argument('--num_steps', default=500000, help='Number of steps to train for.')
    parser.add_argument('--test_steps', default=4000, help='Number of steps to test for.')

    ## Similation setting
    parser.add_argument('--front_vehicle', action='store_true', help='Use front vehicle information')
    parser.add_argument('--step_size', default=0.2, help='Use front vehicle information')
    parser.add_argument('--num_actions', default=3, type=int, help='The number of actions the agent can take')

    opts = parser.parse_args()

    run(opts)
