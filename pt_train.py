from pt_net import LeelaZeroNet
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import multiprocess_generator
import torch
from torch import nn
from tqdm import tqdm
from collections import Counter
from queue import Queue
from threading import Thread
from time import sleep
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

class LeelaPrefetchBuffer:
    def __init__(self, chunk_dir, batch_size, num_workers, skip_factor, shuffle_buffer_size):
        self.queue = Queue(maxsize=8)

        def data_prefetcher(queue):
            gen = multiprocess_generator(chunk_dir=chunk_dir, batch_size=batch_size,
                                         num_workers=num_workers, skip_factor=skip_factor,
                                         shuffle_buffer_size=shuffle_buffer_size)
            for batch in gen:
                output = [torch.tensor(array, requires_grad=False).pin_memory() for array in batch]
                queue.put(output)

        self.prefetcher = Thread(target=data_prefetcher, args=(self.queue,), daemon=True)
        self.prefetcher.start()
        while self.queue.empty():
            sleep(1)  # Wait for the data generator to start returning data before continuing

    def __iter__(self):
        while True:
            yield self.queue.get()


def main():
    parser = ArgumentParser()
    # These parameters control the net and the training process
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--num_residual_blocks', type=int, default=10)
    parser.add_argument('--se_ratio', type=int, default=8)
    parser.add_argument('--l2_reg', type=float, default=0.0005)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.6)
    parser.add_argument('--mixed_precision', action='store_true')
    # These parameters control the data pipeline
    parser.add_argument('--dataset_path', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle_buffer_size', type=int, default=2 ** 17)
    parser.add_argument('--skip_factor', type=int, default=32)
    parser.add_argument('--save_dir', type=Path)
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
        # model = torch.jit.script(model)
        model = model.cuda().train()
        weight_decay_params = []
        non_weight_decay_params = []
        for param in model.named_parameters():
            if param[0].endswith('weight') and ('conv_block' in param[0] or 'residual_block' in param[0]):
                weight_decay_params.append(param[1])
            else:
                non_weight_decay_params.append(param[1])
        opt = torch.optim.AdamW(non_weight_decay_params, lr=args.learning_rate, weight_decay=0.)
        opt.add_param_group({'params': weight_decay_params, "weight_decay": args.l2_reg})

        if args.save_dir is not None and (args.save_dir / "model.pt").is_file():
            saved_data = torch.load(args.save_dir / "model.pt")
            model.load_state_dict(saved_data["model_state_dict"])
            opt.load_state_dict(saved_data["optimizer_state_dict"])
            start_epoch = saved_data["epoch"]
        else:
            start_epoch = 0

    prefetcher = LeelaPrefetchBuffer(chunk_dir=args.dataset_path, batch_size=args.batch_size,
                                     skip_factor=args.skip_factor, num_workers=args.num_workers,
                                     shuffle_buffer_size=args.shuffle_buffer_size)

    if args.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(start_epoch + 1, 999):
        prefetch_iterator = iter(prefetcher)
        loss_totals = Counter()
        total_steps = 0
        with tqdm(total=8192, desc=f"Epoch {epoch}", dynamic_ncols=True) as bar:
            for i in range(8192):
                batch = next(prefetch_iterator)
                batch = [tensor.cuda(non_blocking=True) for tensor in batch]
                opt.zero_grad(set_to_none=True)
                if args.mixed_precision:
                    with autocast():
                        outputs = model(*batch)
                    scaler.scale(outputs.loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, error_if_nonfinite=False)
                    scaler.step(opt)
                    scaler.update()
                else:
                    outputs = model(*batch)
                    outputs.loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, error_if_nonfinite=True)
                    opt.step()
                for key, val in outputs._asdict().items():
                    if key.endswith('loss'):
                        loss_totals[key] += float(val.detach().cpu())
                total_steps += 1
                displayed_loss = {key.removesuffix('_loss'): val / total_steps for key, val in loss_totals.items()}
                bar.set_postfix(displayed_loss)
                bar.update(1)
        if args.save_dir is not None:
            args.save_dir.mkdir(exist_ok=True, parents=True)
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        }, args.save_dir / "model.pt")


if __name__ == '__main__':
    main()
