from tf_net import LeelaZeroNet
import tensorflow as tf
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import multiprocess_generator, ARRAY_SHAPES, make_callable

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
    tf.config.optimizer.set_jit(True)
    # tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = LeelaZeroNet(num_filters=args.num_filters,
                         num_residual_blocks=args.num_residual_blocks,
                         se_ratio=args.se_ratio,
                         l2_reg=args.l2_reg,
                         policy_loss_weight=args.policy_loss_weight,
                         value_loss_weight=args.value_loss_weight,
                         moves_left_loss_weight=args.moves_left_loss_weight,
                         q_ratio=args.q_ratio)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=args.max_grad_norm))
    output_signature = tuple([tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in ARRAY_SHAPES])
    callable_gen = make_callable(args.dataset_path)
    dataset = tf.data.Dataset.from_generator(callable_gen,
                                             output_signature=output_signature).prefetch(10)
    model.fit(dataset, epochs=999, steps_per_epoch=8192)
