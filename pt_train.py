from pt_net import LeelaZeroNet
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import multiprocess_generator, ARRAY_SHAPES
import torch
from torch import nn
from tqdm import tqdm
from collections import Counter

# TODO Test layers with equivalent weights to make sure we can match the TF net's outputs


class LeelaDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path):
        self.gen = multiprocess_generator(dataset_path)

    def __iter__(self):
        return self.gen


if __name__ == '__main__':
    parser = ArgumentParser()
    # These parameters control the net and the training process
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--num_residual_blocks', type=int, default=10)
    parser.add_argument('--se_ratio', type=int, default=8)
    parser.add_argument('--l2_reg', type=float, default=0.0005)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.6)
    # These parameters control the data pipeline
    parser.add_argument('--dataset_path', type=Path, required=True)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # These parameters control the loss calculation. They should not be changed unless you
    # know what you're doing, as the loss values you get will not be comparable with other
    # people's unless they are kept at the defaults.
    parser.add_argument('--policy_loss_weight', type=float, default=1.0)
    parser.add_argument('--value_loss_weight', type=float, default=1.6)
    parser.add_argument('--moves_left_loss_weight', type=float, default=0.5)
    parser.add_argument('--q_ratio', type=float, default=0.2)
    args = parser.parse_args()
    with torch.no_grad():
        model = LeelaZeroNet(num_filters=args.num_filters,
                             num_residual_blocks=args.num_residual_blocks,
                             se_ratio=args.se_ratio,
                             policy_loss_weight=args.policy_loss_weight,
                             value_loss_weight=args.value_loss_weight,
                             moves_left_loss_weight=args.moves_left_loss_weight,
                             q_ratio=args.q_ratio)
        model = model.cuda()
        weight_decay_params = []
        non_weight_decay_params = []
        for param in model.named_parameters():
            if param[0].endswith('weight') and ('conv_block' in param[0] or 'residual_block' in param[0]):
                weight_decay_params.append(param[1])
            else:
                non_weight_decay_params.append(param[1])
        opt = torch.optim.AdamW(non_weight_decay_params, lr=args.learning_rate, weight_decay=0.)
        opt.add_param_group({'params': weight_decay_params, "weight_decay": args.l2_reg})
        all_params = list(model.parameters())
        # opt = torch.optim.Adam(all_params, lr=args.learning_rate)

        dataset = LeelaDataset(args.dataset_path)
        dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, collate_fn=lambda x: (torch.tensor(x[0][0]),
                                                                                                 torch.tensor(x[0][1]),
                                                                                                 torch.tensor(x[0][2]),
                                                                                                 torch.tensor(x[0][3]),
                                                                                                 torch.tensor(x[0][4])))

    loss_totals = Counter()
    total_steps = 0
    with tqdm(total=8192) as bar:
        model.zero_grad()
        for batch in dataloader:
            batch = [tensor.cuda() for tensor in batch]
            outputs = model(*batch)
            outputs.loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=args.max_grad_norm)
            opt.step()
            for key, val in outputs._asdict().items():
                if key.endswith('loss'):
                    loss_totals[key] = float(val.detach().cpu())
            total_steps += 1
            displayed_loss = {key.removesuffix('_loss'): val for key, val in loss_totals.items()}
            bar.set_postfix(displayed_loss)
            bar.update(1)
