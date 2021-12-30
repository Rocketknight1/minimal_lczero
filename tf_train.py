from tf_net import LeelaZeroNet
import tensorflow as tf
from argparse import ArgumentParser
from pathlib import Path
from new_data_pipeline import ARRAY_SHAPES_WITHOUT_BATCH, make_callable

def get_schedule_function(starting_lr, reduce_lr_every_n_epochs, reduce_lr_factor, min_learning_rate):
    def scheduler(epoch, lr):
        num_reductions = int(epoch // reduce_lr_every_n_epochs)
        reduction_factor = reduce_lr_factor ** num_reductions
        return max(min_learning_rate, starting_lr / reduction_factor)
    return scheduler

if __name__ == '__main__':
    parser = ArgumentParser()
    # These parameters control the net and the training process
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--num_residual_blocks', type=int, default=10)
    parser.add_argument('--se_ratio', type=int, default=8)
    parser.add_argument('--constrain_norms', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.6)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--reduce_lr_every_n_epochs', type=int)
    parser.add_argument('--reduce_lr_factor', type=int, default=3)
    parser.add_argument('--min_learning_rate', type=float, default=5e-6)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--tensorboard_dir', type=Path)
    # These parameters control the data pipeline
    parser.add_argument('--dataset_path', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle_buffer_size', type=int, default=2 ** 19)
    parser.add_argument('--skip_factor', type=int, default=32)
    # These parameters control the loss calculation. They should not be changed unless you
    # know what you're doing, as the loss values you get will not be comparable with other
    # people's unless they are kept at the defaults.
    parser.add_argument('--policy_loss_weight', type=float, default=1.0)
    parser.add_argument('--value_loss_weight', type=float, default=1.6)
    parser.add_argument('--moves_left_loss_weight', type=float, default=0.5)
    parser.add_argument('--q_ratio', type=float, default=0.2)
    args = parser.parse_args()
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)
    model = LeelaZeroNet(num_filters=args.num_filters,
                         num_residual_blocks=args.num_residual_blocks,
                         se_ratio=args.se_ratio,
                         constrain_norms=args.constrain_norms,
                         policy_loss_weight=args.policy_loss_weight,
                         value_loss_weight=args.value_loss_weight,
                         moves_left_loss_weight=args.moves_left_loss_weight,
                         q_ratio=args.q_ratio)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=args.max_grad_norm)
    if args.mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    callbacks = []
    if args.reduce_lr_every_n_epochs is not None:
        scheduler = get_schedule_function(args.learning_rate, args.reduce_lr_every_n_epochs,
                                          args.reduce_lr_factor, args.min_learning_rate)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1))
    if args.save_dir is not None:
        args.save_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_path = args.save_dir / 'checkpoint'
        callbacks.append(tf.keras.callbacks.experimental.BackupAndRestore(checkpoint_path))
    if args.tensorboard_dir is not None:
        args.tensorboard_dir.mkdir(exist_ok=True, parents=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_dir, update_freq='batch',
                                                        histogram_freq=1))
    model.compile(optimizer=optimizer)
    array_shapes = [tuple([args.batch_size] + list(shape)) for shape in ARRAY_SHAPES_WITHOUT_BATCH]
    output_signature = tuple([tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in array_shapes])
    callable_gen = make_callable(chunk_dir=args.dataset_path, batch_size=args.batch_size, skip_factor=args.skip_factor,
                                 num_workers=args.num_workers, shuffle_buffer_size=args.shuffle_buffer_size)
    dataset = tf.data.Dataset.from_generator(callable_gen,
                                             output_signature=output_signature).prefetch(10)
    model.fit(dataset, epochs=999, steps_per_epoch=8192, callbacks=callbacks)
