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

class LayerRemover(Model):
    def __init__(self, input_dim=(10, 4), hidden_dim=30, output_dim=2,
                 emb_input_dim = 3, emb_output_dim = 5):
        super(LayerRemover, self).__init__()
        #layer embedding
        self.layer_emb = layers.Embedding(emb_input_dim,emb_output_dim)
        input_dim = (input_dim[0],input_dim[1]+emb_input_dim)

        #feature extraction
        self.features = tf.keras.Sequential()
        self.features.add(layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True), input_shape=input_dim))

        #actor
        self.actor = tf.keras.Sequential()
        # self.actor.add(layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True), input_shape=input_dim))
        self.actor.add(layers.Dense(output_dim))

        #critic
        self.critic = tf.keras.Sequential()
        # self.critic.add(layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True), input_shape=input_dim))
        self.critic.add(layers.Flatten())
        self.critic.add(layers.Dense(1))

    def call(self, x):
        emb = self.layer_emb(x[:,:,0])
        x = tf.concat([emb,x[:,:,1:]],-1)
        return self.actor(x), self.critic(x)