from pt_net import LeelaZeroNet
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import multiprocess_generator
import torch
import pytorch_lightning as pl
from threading import Thread
from queue import Queue


torch.backends.cudnn.benchmark = True


def queued_generator(queue, **kwargs):
    generator = multiprocess_generator(**kwargs)
    for batch in generator:
        batch = [torch.from_numpy(tensor).pin_memory() for tensor in batch]
        queue.put(batch)


class LeelaDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, **kwargs
    ):
        self.queue = Queue(maxsize=4)
        kwargs['queue'] = self.queue
        self.thread = Thread(target=queued_generator, kwargs=kwargs, daemon=True)
        self.thread.start()

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise RuntimeError("This dataset does multiprocessing internally, and should only have a single torch worker!")
        return self

    def __next__(self):
        return self.queue.get(block=True)


def main():
    parser = ArgumentParser()
    # These parameters control the net and the training process
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--num_residual_blocks", type=int, default=10)
    parser.add_argument("--se_ratio", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.6)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "ranger21", "adan"])
    # These parameters control the data pipeline
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer_size", type=int, default=2 ** 19)
    parser.add_argument("--skip_factor", type=int, default=32)
    parser.add_argument("--save_dir", type=Path, required=True)
    # These parameters control the loss calculation. They should not be changed unless you
    # know what you're doing, as the loss values you get will not be comparable with other
    # people's unless they are kept at the defaults.
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--value_loss_weight", type=float, default=1.6)
    parser.add_argument("--moves_left_loss_weight", type=float, default=0.5)
    parser.add_argument("--q_ratio", type=float, default=0.2)
    args = parser.parse_args()
    # TODO Still slower than TF and I'm not sure why
    with torch.no_grad():
        model = None
        if args.save_dir.is_dir():
            try:
                model = LeelaZeroNet.load_from_checkpoint(args.save_dir)
            except:
                model = None
        if model is None:
            model = LeelaZeroNet(
                num_filters=args.num_filters,
                num_residual_blocks=args.num_residual_blocks,
                se_ratio=args.se_ratio,
                policy_loss_weight=args.policy_loss_weight,
                value_loss_weight=args.value_loss_weight,
                moves_left_loss_weight=args.moves_left_loss_weight,
                q_ratio=args.q_ratio,
                optimizer=args.optimizer,
                learning_rate=args.learning_rate
            )

        dataset = LeelaDataset(
            chunk_dir=args.dataset_path,
            batch_size=args.batch_size,
            skip_factor=args.skip_factor,
            num_workers=args.num_workers,
            shuffle_buffer_size=args.shuffle_buffer_size,
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, pin_memory=True)

    precision = 16 if args.mixed_precision else 32
    trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=precision, limit_train_batches=8192, max_epochs=100,
                         default_root_dir=args.save_dir)
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
