from tf_net import LeelaZeroNet
from tf_data_pipeline import get_dataset
import tensorflow as tf
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--num_residual_blocks', type=int, default=10)
    parser.add_argument('--se_ratio', type=int, default=8)
    parser.add_argument('--l2_reg', type=float, default=0.0005)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dataset_path', type=Path)
    args = parser.parse_args()
    model = LeelaZeroNet(num_filters=args.num_filters, num_residual_blocks=args.num_residual_blocks,
                         se_ratio=args.se_ratio, l2_reg=args.l2_reg)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))
    dataset = get_dataset(args.dataset_path)
    model.fit(dataset, epochs=999, steps_per_epoch=8192)
