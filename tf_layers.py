import tensorflow as tf
import lc0_az_policy_map


class L2WeightDecay(tf.keras.constraints.Constraint):
    def __init__(self, decay_rate):
        super().__init__()
        self.decay_factor = 1 - decay_rate

    def __call__(self, w):
        return w * self.decay_factor


class NormConstraint(tf.keras.constraints.Constraint):
    def __init__(self, initialization_type=None):
        self.initialization_type = initialization_type

    def __call__(self, w):
        fan_in = tf.cast(tf.reduce_prod(w.shape[:-1]), tf.float32)
        fan_out = tf.cast(w.shape[-1], tf.float32)
        n_dims = fan_in * fan_out
        # The expected norms with _uniform and _normal versions of each initializer
        # are equivalent but have some bonus maths for clarity anyway
        if self.initialization_type == 'glorot_uniform':
            limit = tf.sqrt(6 / (fan_in + fan_out))
            desired_norm = tf.sqrt(n_dims / 3) * limit
        elif self.initialization_type == 'he_uniform':
            limit = tf.sqrt(6 / fan_in)
            desired_norm = tf.sqrt(n_dims / 3) * limit
        elif self.initialization_type == 'glorot_normal':
            scale = tf.sqrt(2 / (fan_in + fan_out))
            desired_norm = scale * tf.sqrt(n_dims)
        elif self.initializatio_tpe == 'he_normal':
            scale = tf.sqrt(2 / fan_in)
            desired_norm = scale * tf.sqrt(n_dims)
        else:
            raise ValueError("Unknown initialization type!")
        return tf.clip_by_norm(w, desired_norm)

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

    def call(self, inputs, training=None, mask=None):
        pooled = self.pooler(inputs)
        squeezed = tf.nn.relu(pooled @ self.squeeze)
        excited = squeezed @ self.excite
        excited = tf.expand_dims(tf.expand_dims(excited, -1), -1)  # Add two extra dims for broadcasting
        gammas, betas = tf.split(excited, 2, axis=1)
        gammas = tf.nn.sigmoid(gammas)
        return gammas * inputs + betas


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, output_channels, constrain_norms, name, bn_scale):
        super().__init__()
        if constrain_norms:
            constraint = NormConstraint("glorot_normal")
        else:
            constraint = None
        self.conv_layer = tf.keras.layers.Conv2D(output_channels, filter_size, use_bias=False,
                                                 padding='same',
                                                 kernel_initializer='glorot_normal',
                                                 kernel_constraint=constraint,
                                                 data_format='channels_first',
                                                 name=name + '/conv2d'
                                                 )
        self.batchnorm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            axis=1,
            center=True,
            scale=bn_scale,
            name=name + '/batchnorm',
            dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        out = self.conv_layer(inputs)
        out = self.batchnorm(out, training=training)
        return tf.keras.activations.relu(out)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, se_ratio, constrain_norms, name):
        super().__init__()
        # We always retain norm constraints in the residual block because it's necessary when combined with
        # batchnorms, see https://blog.janestreet.com/l2-regularization-and-batch-norm/
        if constrain_norms:
            constraint = NormConstraint("glorot_normal")
        else:
            constraint = None
        self.conv1 = tf.keras.layers.Conv2D(channels,
                                            3,
                                            use_bias=False,
                                            padding='same',
                                            kernel_initializer='glorot_normal',
                                            kernel_constraint=constraint,
                                            data_format='channels_first',
                                            name=name + '/1/conv2d')
        self.batch_norm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            axis=1,
            center=True,
            scale=False,
            name=name + '/batchnorm',
            dtype=tf.float32)
        self.conv2 = tf.keras.layers.Conv2D(channels,
                                            3,
                                            use_bias=False,
                                            padding='same',
                                            kernel_initializer='glorot_normal',
                                            kernel_constraint=constraint,
                                            data_format='channels_first',
                                            name=name + '/2/conv2d')
        self.squeeze_excite = SqueezeExcitation(se_ratio, name=name + '/se')

    def call(self, inputs, training=None, mask=None):
        out1 = self.conv1(inputs)
        out1 = tf.nn.relu(self.batch_norm(out1))
        out2 = self.conv2(out1)
        out2 = self.squeeze_excite(out2)
        return tf.nn.relu(inputs + out2)


class ConvolutionalPolicyHead(tf.keras.layers.Layer):
    def __init__(self, num_filters, constrain_norms):
        super().__init__()
        self.conv_block = ConvBlock(filter_size=3, output_channels=num_filters,
                                    constrain_norms=constrain_norms, name='policy1', bn_scale=True)
        # No constraint on the final convolution, because it's not going to be followed by a batchnorm
        self.conv = tf.keras.layers.Conv2D(
            80,
            3,
            use_bias=True,
            padding='same',
            kernel_initializer='glorot_normal',
            data_format='channels_first',
            name='policy')
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs, training=None, mask=None):
        flow = self.conv_block(inputs)
        flow = self.conv(flow)
        h_conv_pol_flat = tf.reshape(flow, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class DensePolicyHead(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='glorot_normal',
                                         name='policy/dense1', activation='relu')
        # No constraint on the final layer, because it's not going to be followed by a batchnorm
        self.fc_final = tf.keras.layers.Dense(1858,
                                              kernel_initializer='glorot_normal',
                                              name='policy/dense')

    def call(self, inputs, training=None, mask=None):
        if tf.rank(inputs) > 2:
            # Flatten input before proceeding
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        out = self.fc1(inputs)
        return self.fc_final(out)


class ConvolutionalValueOrMovesLeftHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_filters, hidden_dim, constrain_norms, relu):
        super().__init__()
        self.num_filters = num_filters
        self.conv_block = ConvBlock(filter_size=1, output_channels=num_filters,
                                    constrain_norms=constrain_norms, name='value/conv', bn_scale=True)
        # No constraint on the final layers, because they're not going to be followed by a batchnorm
        self.fc1 = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            kernel_initializer='glorot_normal',
            name='value/dense1')
        self.fc_out = tf.keras.layers.Dense(output_dim, use_bias=True, activation='relu' if relu else None,
                                            kernel_initializer='glorot_normal', name='value/dense2')

    def call(self, inputs, training=None, mask=None):
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

    def call(self, inputs, training=None, mask=None):
        if tf.rank(inputs) > 2:
            # Flatten input before proceeding
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        flow = self.fc1(inputs)
        return self.fc_out(flow)


class CoatnetSelfAttention(tf.keras.layers.Layer):
    # TODO Not done yet!
    #      Among other things, missing the self-attention logit scale!
    def __init__(self, dim):
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.qkv_weights = self.add_weight(name="qkv_weights", shape=(dim, dim*3),
                                           initializer='glorot_normal', trainable=True)
        self.relative_attention_bias = self.add_weight(name="relative_attention_bias", shape=(15 * 15,),
                                                       initializer="glorot_normal", trainable=True)
        width_offsets = tf.expand_dims(tf.range(8), 0) - tf.expand_dims(tf.range(8), 1)
        width_offsets += 7
        height_offsets = tf.transpose(width_offsets)
        self.relative_attention_indices = width_offsets + (15 * height_offsets)
        breakpoint()  # Double-check those indices
        print()

    def call(self, inputs):

        normalized_input = self.layernorm(inputs)
        qkv = normalized_input @ self.qkv_weights
        query, key, value = tf.split(qkv, 3, axis=-1)
        self_attention_logits = tf.einsum("bi, bj -> bij", query, key)
        relative_attention_bias = tf.gather(self.relative_attention_bias, self.relative_attention_indices)
        self_attention_logits += relative_attention_bias
        self_attention_weights = tf.nn.softmax(self_attention_logits)
        breakpoint()
        print()  # Double-check weights shape here
        self_attention_output = value @ self_attention_weights
        return inputs + self_attention_output
