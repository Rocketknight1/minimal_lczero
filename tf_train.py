from tf_net import LeelaZeroNet
from tf_data_pipeline import get_dataset
import tensorflow as tf


if __name__ == '__main__':
    model = LeelaZeroNet(num_filters=128, num_residual_blocks=10, se_ratio=8, l2_reg=0.0005)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    dataset = get_dataset("/media/matt/Data/leela_training_data/chad_new_rescored/")
    model.fit(dataset, epochs=999, steps_per_epoch=8192)
