from keras.models import Model
import keras.backend as K
from keras import regularizers
from keras.engine.topology import Layer
import numpy as np
from keras.layers import Dropout, merge, Input

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
            norm1 = K.tf.subtract(x1, mu1)
            norm2 = K.tf.subtract(x2, mu2)
            s1s2 = K.tf.multiply(s1, s2)
            #print(x1,mu1)
            #return s1s2
            z = K.tf.square(K.tf.div(norm1, s1)) + K.tf.square(K.tf.div(norm2, s2)) - \
                2 * K.tf.div(K.tf.multiply(rho, K.tf.multiply(norm1, norm2)), s1s2)
            #return z
            negRho = 1 - K.tf.square(rho)
            result = K.tf.exp(K.tf.div(-z, 2 * negRho))
            denom = 2 * np.pi * K.tf.multiply(s1s2, K.tf.sqrt(negRho))
            result = K.tf.div(result, denom)
            return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data):
            
            result0 = tf_2d_normal( x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            #return result0
            epsilon = 1e-20
            result1 = K.tf.multiply(result0, z_pi)
            result1 = K.tf.reduce_sum(result1, 1, keep_dims=True)
            # at the beginning, some errors are exactly zero.
            result1 = -K.tf.log(K.tf.maximum(result1, 1e-20))
            #return result1
            result2 = K.tf.multiply(z_eos, eos_data) + \
                K.tf.multiply(1 - z_eos, 1 - eos_data)
            
            result2 = -K.tf.log(result2)
            #return result2
            result = result1 + result2
            #return result
            return K.tf.reduce_sum(result)
        
class Mod(Layer):
    
    def __init__(self, kernel_regularizer=None, **kwargs):
        #self.v_dim = v_dim
        super(Mod, self).__init__(**kwargs)

    def call(self, data, mask=None):
        loss = get_lossfunc(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9])
        return loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)
    

#x1_data, x2_data, eos_data = np.zeros(2),np.zeros(2),np.zeros(2)
#z_eos = np.array([0.4914, 0.4914])
#z_pi = np.array([[0.3745, 0.3426, 0.2829],[0.3745, 0.3426, 0.2829]])
#z_mu1 = np.array([[0.0226, -0.1327,  0.0370],[0.0226, -0.1327,  0.0370]])
#z_mu2 = np.array([[0.2141, 0.1454, 0.2249],[0.2141, 0.1454, 0.2249]])
#z_sigma1 = np.array([[1.0479, 0.8967, 0.9361],[1.0479, 0.8967, 0.9361]])
#z_sigma2= np.array([[1.2994, 0.9598, 0.8391],[1.2994, 0.9598, 0.8391]])
#z_corr = np.array([[-0.3572, -0.0127, -0.0807],[-0.3572, -0.0127, -0.0807]])
        
#x1_data, x2_data, eos_data = np.zeros(2),np.zeros(2),np.zeros(2)
x1_data = np.array([0.70, 1.00])
x2_data = np.array([0.75, 0.1500])
eos_data = np.array([0.0, 0.0])

'''
z_eos = np.array([0.5616, 0.9616])
z_pi = np.array([[0.3256, 0.3545, 0.3199],[0.1256, -0.3545, 0.3199]])
z_mu1 = np.array([[-0.2747,  -0.0437, -0.1852],[-0.2747,  0.0437, -0.1852]])
z_mu2 = np.array([[-0.1291, -0.1344, -0.2144],[-0.1291, 0.1344, -0.2144]])
z_sigma1 = np.array([[0.0822, 1.2335, 0.9681],[0.9822, 1.2335, 0.5681]])
z_sigma2= np.array([[0.8423, 1.1518, 1.5683],[0.3423, 1.1518, 1.2683]])
z_corr = np.array([[-0.0108,  0.1526,  -0.2197],[-0.0108,  -0.1526,  0.2197]])
'''

z_eos = np.array([0.5317, 0.5309])
z_pi = np.array([[0.3464, 0.2960, 0.3576],
         [0.3494, 0.2955, 0.3551]])
z_mu1 = np.array([[ 0.0840, -0.1579, -0.1997],
         [ 0.0917, -0.1582, -0.1874]])
z_mu2 = np.array([[-0.1352, -0.1710, -0.2975],
         [-0.1304, -0.1704, -0.3077]])
z_sigma1 = np.array([[1.0845, 1.0115, 0.9486],
         [1.0979, 1.0160, 0.9539]])
z_sigma2= np.array([[1.0803, 0.9188, 0.8369],
         [1.0840, 0.9162, 0.8346]])
z_corr = np.array([[0.0546, 0.1053, 0.0143],
         [0.0485, 0.1062, 0.0176]])



inp_data0 = Input(shape=(None,))
inp_data1 = Input(shape=(None,))
inp_data2 = Input(shape=(None,))
inp_data3 = Input(shape=(None,))
inp_data4 = Input(shape=(None,))
inp_data5 = Input(shape=(None,))
inp_data6 = Input(shape=(None,))
inp_data7 = Input(shape=(None,))
inp_data8 = Input(shape=(None,))
inp_data9 = Input(shape=(None,))

loss = Mod()([inp_data0,inp_data1,inp_data2,inp_data3,inp_data4,inp_data5,inp_data6,inp_data7,inp_data8,inp_data9])
m = Model([inp_data0,inp_data1,inp_data2,inp_data3,inp_data4,inp_data5,inp_data6,inp_data7,inp_data8,inp_data9],loss)

print(m.predict([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data]))
#print(m.predict([z_pi[0:1], z_mu1[0:1], z_mu2[0:1], z_sigma1[0:1], z_sigma2[0:1], z_corr[0:1], z_eos[0:1], x1_data[0:1], x2_data[0:1], eos_data[0:1]]))