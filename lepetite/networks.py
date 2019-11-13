'''
Translated to TF2 from
VGG11/13/16/19 in Pytorch (https://github.com/anubhavashok/N2N/blob/544f5dd6c9c023c81b9c7b8ff5c8ccc1c895c66d/model/vgg.py)
'''
import tensorflow as tf
from tensorflow.keras import layers,Model

class VGGLike(Model):
    def __init__(self, action = [1 for i in range(10)], 
        #(layer type,kernel/pool size, stride, filters)
        parent_states=[(1,9,1,64),(2,2,2,0), 
                      (1,7,1,128),(2,2,2,0),
                      (1,5,1,256),(2,2,2,0),
                      (1,3,1,512),(2,2,2,0),
                      (1,3,1,1024),(2,2,2,0)]):
        super(VGGLike, self).__init__()
        self.action = action
        self.parent_states = parent_states
        self.child_states = []
        for i,a in enumerate(action):
          if a == 1:
            self.child_states.append(parent_states[i])
          else:
            self.child_states.append((0,0,0,0))
        self.features = self._make_layers()
        self.head = tf.keras.Sequential()
        self.head.add(layers.Dense(10))

    def _make_layers(self):
      m = tf.keras.Sequential()
      for s in self.child_states:
        if s[0]==2:
          m.add(layers.MaxPool2D(pool_size=s[1], 
                                 strides=s[2], 
                                 padding='same'))
        elif s[0]==1:
          m.add(layers.Conv2D(filters=s[3], 
                              kernel_size=s[1], 
                              strides=s[2],
                              padding='same'))
          m.add(layers.ReLU())
        else:
          pass
      m.add(layers.AvgPool2D(pool_size=1, strides=1, padding='same'))
      m.add(layers.Flatten())
      return m
  
    def call(self, x):
        x = self.features(x)
        x = self.head(x)
        return x