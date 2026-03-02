import argparse
from experiments.common.setup_experiment import setup_experiment, flush_logs, get_value_logger
from core.constraints import BaseConstraint, BoxConstraint
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
from core.flow.constrained_distribution import ConstrainedDistribution
import torch as th
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def compute_mean_std(data):
    mean = th.mean(data,axis=0)
    std = th.std(data,axis=0)
    data = (data - mean) / (std+1e-8)
    return data, mean, std

def main():
    """
    Train flow forwad using generated samples from a file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, nargs='?', default=5000)
    parser.add_argument("--eval_freq", type=int, nargs='?', default=1)
    parser.add_argument("--lr", type=float, nargs='?', default=1e-5)
    parser.add_argument("--batch_size", type=int, nargs='?', default=5000)
    parser.add_argument("--hidden_size", type=int, nargs='?', default=256)
    parser.add_argument("--transform_count", type=int, nargs='?', default=6)
    parser.add_argument("--mollifier_sigma", type=float, nargs='?', default=0.0001)
    parser.add_argument("--gradient_clip_value", type=float, nargs='?', default=0.1)
    parser.add_argument("--take_log_again", action="store_true")

    parser.add_argument("--seed", type=int, nargs='?', default=1)
    parser.add_argument("--device", type=str, nargs='?', default='cuda:2')
    parser.add_argument("--data_folder", type=str, nargs='?', default='')
    args = parser.parse_args()

    log_dir = args.data_folder + 'flow_model/state/'
    logger = get_value_logger(log_dir)
    seed_everything(args.seed)

    state = np.load(args.data_folder + "state.npy")
    dim = state.shape[1]

    # Define the flow model
    flow = RealNvp(dim, args.transform_count, conditional_param_count=0, hidden_size=args.hidden_size).to(args.device)
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=args.lr)

    # Define the mollified uniform distribution
    box_l = th.full((dim, ), -1).double()
    box_h = th.full((dim, ), 1).double()
    uniform_constraint = BoxConstraint(dim, box_l, box_h).to(args.device)
    mollified_uniform_distribution = ConstrainedDistribution(uniform_constraint, args.mollifier_sigma)

    # Load dataset
    data_orign = th.from_numpy(state).double().to(args.device)
    data, mean, std = compute_mean_std(data_orign)
    np.savez(log_dir + "src_info.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())

    train_sample_count = int(data.shape[0]*0.95)
    test_sample_count = data.shape[0] - train_sample_count

    train_data = data[:train_sample_count]
    bound = th.stack([data_orign[:train_sample_count].max(dim=0).values, data_orign[:train_sample_count].min(dim=0).values]).cpu().numpy()
    test_data = data[train_sample_count:]

    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("Staring experiment")
    start_time = time.time()
    for epoch in range(args.epochs):
        losses = []
        # Update flow for each batch
        for batch, in data_loader:
            loss = update_flow_batch(flow, mollified_uniform_distribution, batch, optimizer, gradient_clip_value=args.gradient_clip_value, take_log_again=args.take_log_again)
            losses.append(loss)

        if (epoch+1)%args.eval_freq == 0:
            # Evaluate
            with th.no_grad():
                # Calculate accuracy (z -(g)-> x)
                z_act = th.rand((len(test_data), dim)).double().to(args.device)*2-1 
                z = th.cat([z_act, test_data[:, dim:]], dim=1)

                generated_samples = flow.g(z)[0]*std + mean

                valid_count = 0
                numpy_generated_samples = generated_samples.cpu().detach().numpy()
                for i in range(test_sample_count):
                    if((numpy_generated_samples[i] <= bound[0]).all() and (numpy_generated_samples[i] >= bound[1]).all()):
                        valid_count += 1
                accuracy = valid_count/test_sample_count
                
                # Calculate recall (x -(f)-> z)
                mapped_z = flow.f(test_data)[0][:, :dim]
                validity_z = th.all(mapped_z >= -1, dim=1) & th.all(mapped_z <= 1, dim=1)
                valid_z_count = validity_z.int().sum().item()
                recall = valid_z_count/len(validity_z)
            

            elapsed_time = time.time() - start_time
            logger.record("train/time_elapsed", elapsed_time)
            logger.record("train/mean_loss", np.mean(losses))
            logger.record("train/accuracy", accuracy)
            logger.record("train/recall", recall)
            logger.record("train/epoch", epoch+1)
            flush_logs()
            logger.dump(epoch)

    flow.save_module(log_dir + "flow.pt")

if __name__ == "__main__":
    main()

    