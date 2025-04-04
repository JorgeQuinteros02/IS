from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import tensorflow as tf
import keras

class SplitConv2d(keras.Layer):
    h_input = None
    h_output = None
    input_stabilizer_size = None
    output_stabilizer_size = None
    def __init__(self,
                 out_channels:int,
                 kernel_size:int,
                 stride=1,
                 pad=0,
                 *,
                 activity_regularizer=None,
                 trainable=True,
                 dtype=tf.float32,
                 autocast=True,
                 name=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer,
                         trainable=trainable,
                         dtype=dtype,
                         autocast=autocast,
                         name=name,
                         **kwargs)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.stride = stride if hasattr(stride, "__getitem__") else (stride, stride)

    def build(self, input_shape):
        self.in_channels = input_shape[-1] // self.input_stabilizer_size
        
        inds, gconv_shape_info, w_shape = gconv2d_util(self.h_input,
                                                self.h_output,
                                                self.in_channels,
                                                self.out_channels,
                                                self.kernel_size)
        

        
        self.w = self.add_weight(
            shape=w_shape,
            name="W",
            dtype=self.dtype,
            initializer="random_normal"
        )

        self.b = self.add_weight(
            shape=(self.out_channels,),
            name='b',
            dtype=self.dtype
        )
        
 

    def call(self, inputs):
        inds, gconv_shape_info, w_shape = gconv2d_util(self.h_input,
                                        self.h_output,
                                        self.in_channels,
                                        self.out_channels,
                                        self.kernel_size)


        y = gconv2d(inputs,
                       self.w,
                       [1,1,1,1],
                       "VALID",
                       inds,
                       gconv_shape_info,
                       use_cudnn_on_gpu=True,
                       data_format='NHWC',)
        # batch_size, _,  ny_out, nx_out = y.shape
        # y = tf.reshape(y, (batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out))

        # bb = tf.reshape(self.b, (1, self.out_channels, 1, 1, 1))
        # b_shape = tf.broadcast_static_shape(y, bb)
        # y = tf.broadcast_to(y, b_shape)
        # b = tf.broadcast_to(bb, b_shape)

        # y = y + b

        # n, nc, ng, nx, ny = y.data.shape
        # y = tf.reshape(y, (n, nc * ng, nx, ny))
        # print(y.shape)
        return y



class P4ConvZ2(SplitConv2d):
    h_input = 'Z2'
    h_output = 'C4'
    input_stabilizer_size = 1
    output_stabilizer_size = 4


class P4ConvP4(SplitConv2d):
    h_input = 'C4'
    h_output = 'C4'
    input_stabilizer_size = 4
    output_stabilizer_size = 4
