import tensorflow.keras.backend as K
from p4g.p4g_conv import p4gconv2d_util
from tensorflow.keras.layers import InputSpec, Conv2D, Conv2DTranspose
from tensorflow.keras.utils import get_custom_objects
from keras_gcnn.layers.convolutional import gconv2d


class P4GConv2D(Conv2D):
    def __init__(self, filters, kernel_size, h_input, strides=(1, 1), padding='valid', data_format=None,
                 dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, transpose=False, **kwargs):
        """
        :param filters:
        :param kernel_size:
        :param h_input:
        :param h_output:
        :param h_input: one of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        :param h_output: one of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
              The choice of h_output of one layer should equal h_input of the next layer.
        :param strides:
        :param padding:
        :param data_format:
        :param dilation_rate:
        :param activation:
        :param use_bias:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param kwargs:
        """
        if use_bias:
            raise NotImplementedError('Does not support bias yet')  # TODO: support bias

        if not isinstance(kernel_size, int) and not kernel_size[0] == kernel_size[1]:
            raise ValueError('Requires square kernel')

        self.h_input = h_input
        self.h_output = 'D4'
        self.transpose = transpose

        super(P4GConv2D, self).__init__(filters, kernel_size, strides=strides, padding=padding, data_format=data_format,
                                      dilation_rate=dilation_rate, activation=activation,
                                      use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint, **kwargs)

    def compute_output_shape(self, input_shape):
        if self.transpose:
            shape = Conv2DTranspose.compute_output_shape(self, input_shape)
        else:
            shape = super(P4GConv2D, self).compute_output_shape(input_shape)
        nto = shape[3] * 8

        return (shape[0], shape[1], shape[2], nto)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            raise NotImplementedError('Channels first is not implemented for GConvs yet.')
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        orig_input_dim = input_dim
        if self.h_input == 'C4':
            input_dim //= 4
        elif self.h_input == 'D4':
            input_dim //= 8

        self.gconv_indices, self.gconv_shape_info, w_shape = p4gconv2d_util(h_input=self.h_input,
                                                                          in_channels=input_dim,
                                                                          out_channels=self.filters,
                                                                          ksize=self.kernel_size[0])

        self._kernel = self.add_weight(shape=w_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            raise NotImplementedError()
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: orig_input_dim})
        self.built = True

    def call(self, inputs):
        outputs = gconv2d(
            inputs,
            self._kernel,
            self.gconv_indices,
            self.gconv_shape_info,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            transpose=self.transpose,
            output_shape=self.compute_output_shape(inputs.shape))

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = super(P4GConv2D, self).get_config()
        config['h_input'] = self.h_input
        config['h_output'] = self.h_output
        return config




get_custom_objects().update({'P4GConv2D': P4GConv2D})

