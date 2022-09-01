import tensorflow as tf
from keras.optimizers.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export
from keras import backend_config


@keras_export('keras.optimizers.Adan')
class TFAdan(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.02,
                 beta_2=0.08,
                 beta_3=0.01,
                 epsilon=1e-7,
                 weight_decay=0.,
                 name='Adan',
                 **kwargs):

        super(TFAdan, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('beta_3', beta_3)
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        for var in var_list:
            self.add_slot(var, 'n')
        for var in var_list:
            self.add_slot(var, 'prev_grad')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(TFAdan, self)._prepare_local(var_device, var_dtype, apply_state)

        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_3_t = tf.identity(self._get_hyper('beta_3', var_dtype))
        lr = tf.identity(self._get_hyper('learning_rate', var_dtype))
        weight_decay = tf.identity(self._get_hyper('weight_decay', var_dtype))
        local_step = tf.cast(self.iterations + 1, var_dtype)

        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=tf.convert_to_tensor(
                    self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                beta_3_t=beta_3_t,
                weight_decay=weight_decay,
                local_step=local_step))

    @tf.function(jit_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        n = self.get_slot(var, 'n')
        prev_grad = self.get_slot(var, 'prev_grad')

        if coefficients['local_step'] > 1:
            # m.mul_(1 - beta1).add_(grad, alpha=beta1)
            m = m.assign(m * (1 - coefficients["beta_1_t"]) + grad * coefficients["beta_1_t"])

            grad_diff = grad - prev_grad  # Unchanged from PyTorch

            # v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
            v = v.assign(v * (1 - coefficients["beta_2_t"]) + grad_diff * coefficients["beta_2_t"])

            # next_n = (grad + (1 - beta2) * grad_diff) ** 2
            next_n = tf.square(grad + (1 - coefficients["beta_2_t"]) * grad_diff)

            # n.mul_(1 - beta3).add_(next_n, alpha=beta3)
            n = n.assign(n * (1 - coefficients["beta_3_t"]) + next_n * coefficients["beta_3_t"])

        # Bias correction terms

        correct_m, correct_v, correct_n = map(lambda n: 1 / (1 - (1 - n) ** coefficients['local_step']), (coefficients["beta_1_t"], coefficients["beta_2_t"], coefficients["beta_3_t"]))

        # gradient step

        # weighted_step_size = lr / (n * correct_n).sqrt().add_(eps)
        weighted_step_size = coefficients["lr"] / (tf.sqrt(n * correct_n) + coefficients["epsilon"])

        # denom = 1 + weight_decay * lr
        denom = 1 + coefficients["weight_decay"] * coefficients["lr"]

        # data.addcmul_(weighted_step_size, (m * correct_m + (1 - beta2) * v * correct_v), value = -1.).div_(denom)
        var = var.assign_sub(weighted_step_size * (m * correct_m + (1 - coefficients["beta_2_t"]) * v * correct_v))
        var = var.assign(var / denom)

        prev_grad.assign(grad)

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        raise NotImplementedError

    def get_config(self):
        config = super(TFAdan, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'beta_3': self._serialize_hyperparameter('beta_3'),
            'epsilon': self.epsilon,
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
        })
        return