import tensorflow as tf
import lc0_az_policy_map


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, se_ratio, name):
        super().__init__()
        self.se_ratio = se_ratio
        self.pooler = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')
        self.squeeze = None
        self.excite = None
        self.name_str = name

    def build(self, input_shape):
        channels = input_shape[-3]
        assert channels % self.se_ratio == 0
        squeeze_dim = int(channels // self.se_ratio)
        excite_dim = 2 * channels
        self.squeeze = self.add_weight(name=self.name_str + '/squeeze', shape=(channels, squeeze_dim),
                                       initializer='glorot_normal', trainable=True)
        self.excite = self.add_weight(name=self.name_str + '/excite', shape=(squeeze_dim, excite_dim),
                                      initializer='glorot_normal', trainable=True)

    def call(self, inputs):
        pooled = self.pooler(inputs)
        squeezed = tf.nn.relu(pooled @ self.squeeze)
        excited = squeezed @ self.excite
        excited = tf.expand_dims(tf.expand_dims(excited, -1), -1)  # Add two extra dims for broadcasting
        gammas, betas = tf.split(excited, 2, axis=1)
        gammas = tf.nn.sigmoid(gammas)
        return gammas * inputs + betas


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, output_channels, l2_reg, name, bn_scale):
        super().__init__()
        if l2_reg is None or l2_reg == 0:
            regularizer = None
        else:
            regularizer = tf.keras.regularizers.L2(l2_reg)
        self.conv_layer = tf.keras.layers.Conv2D(output_channels, filter_size, use_bias=False,
                                                 padding='same',
                                                 kernel_initializer='glorot_normal',
                                                 kernel_regularizer=regularizer,
                                                 data_format='channels_first',
                                                 name=name + '/conv2d'
                                                 )
        self.batchnorm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            axis=1,
            center=True,
            scale=bn_scale,
            name=name + '/batchnorm')

    def call(self, inputs):
        out = self.conv_layer(inputs)
        out = self.batchnorm(out)
        return tf.keras.activations.relu(out)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, se_ratio, l2_reg, name):
        super().__init__()
        # We always retain L2 regularization in the residual block because it's necessary when combined with
        # batchnorms, see https://blog.janestreet.com/l2-regularization-and-batch-norm/
        if l2_reg is None or l2_reg == 0:
            regularizer = None
        else:
            regularizer = tf.keras.regularizers.L2(l2_reg)
        self.conv1 = tf.keras.layers.Conv2D(channels,
                                            3,
                                            use_bias=False,
                                            padding='same',
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=regularizer,
                                            data_format='channels_first',
                                            name=name + '/1/conv2d')
        self.batch_norm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            axis=1,
            center=True,
            scale=False,
            name=name + '/batchnorm')
        self.conv2 = tf.keras.layers.Conv2D(channels,
                                            3,
                                            use_bias=False,
                                            padding='same',
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=regularizer,
                                            data_format='channels_first',
                                            name=name + '/2/conv2d')
        self.squeeze_excite = SqueezeExcitation(se_ratio, name=name + '/se')

    def call(self, inputs):
        out1 = self.conv1(inputs)
        out1 = tf.nn.relu(self.batch_norm(out1))
        out2 = self.conv2(out1)
        out2 = self.squeeze_excite(out2)
        return tf.nn.relu(inputs + out2)


class ConvolutionalPolicyHead(tf.keras.layers.Layer):
    def __init__(self, num_filters, l2_reg):
        super().__init__()
        self.conv_block = ConvBlock(filter_size=3, output_channels=num_filters,
                                    l2_reg=l2_reg, name='policy1', bn_scale=True)
        # No l2_reg on the final convolution, because it's not going to be followed by a batchnorm
        self.conv = tf.keras.layers.Conv2D(
            80,
            3,
            use_bias=True,
            padding='same',
            kernel_initializer='glorot_normal',
            data_format='channels_first',
            name='policy')
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        flow = self.conv_block(inputs)
        flow = self.conv(flow)
        h_conv_pol_flat = tf.reshape(flow, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class DensePolicyHead(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='glorot_normal',
                                         name='policy/dense1')
        # No l2_reg on the final layer, because it's not going to be followed by a batchnorm
        self.fc_final = tf.keras.layers.Dense(1858,
                                              kernel_initializer='glorot_normal',
                                              name='policy/dense')

    def call(self, inputs):
        if tf.rank(inputs) > 2:
            # Flatten input before proceeding
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        out = self.fc1(inputs)
        return self.fc_final(out)


class ConvolutionalValueOrMovesLeftHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_filters, hidden_dim, l2_reg, relu):
        super().__init__()
        self.num_filters = num_filters
        self.conv_block = ConvBlock(filter_size=1, output_channels=num_filters,
                                    l2_reg=l2_reg, name='value/conv', bn_scale=True)
        # No l2_reg on the final layers, because they're not going to be followed by a batchnorm
        self.fc1 = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            kernel_initializer='glorot_normal',
            name='value/dense1')
        self.fc_out = tf.keras.layers.Dense(output_dim, use_bias=True, activation='relu' if relu else None,
                                            kernel_initializer='glorot_normal', name='value/dense2')

    def call(self, inputs):
        flow = self.conv_block(inputs)
        flow = tf.reshape(flow, [-1, self.num_filters * 8 * 8])
        flow = self.fc1(flow)
        return self.fc_out(flow)


class DenseValueOrMovesLeftHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, hidden_dim, relu):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='glorot_normal',
                                         activation='relu', name='value/dense1')
        self.fc_out = tf.keras.layers.Dense(output_dim,
                                            kernel_initializer='glorot_normal',
                                            name='value/dense',
                                            activation='relu' if relu else None)

    def call(self, inputs):
        if tf.rank(inputs) > 2:
            # Flatten input before proceeding
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        flow = self.fc1(inputs)
        return self.fc_out(flow)
