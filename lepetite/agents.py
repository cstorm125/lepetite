'''
PPO base translated to TF2 from https://github.com/seungeunrho/minimalRL
'''

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp

class PPOCategorical(Model):
    def __init__(self, input_dim = 4, hidden_dims = [64], output_dim = 2, lr = 5e-4,
                gamma=1, lamb=0.95, eps_clip=0.1):
        super(PPOCategorical, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.lamb = lamb
        self.eps_clip = eps_clip
        
        self.loss_lst = []
        self.trajectories = []
        self.features  = tf.keras.Sequential([layers.Dense(h, activation='relu') for h in self.hidden_dims])
        self.actor = layers.Dense(self.output_dim, activation='softmax')
        self.critic  = layers.Dense(1, activation=None)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
    def act(self, x):
        x = self.features(x)
        prob = self.actor(x)
        return prob
    
    def critique(self, x):
        x = self.features(x)
        v = self.critic(x)
        return v
      
    def add(self, *args):
        t = {'state':args[0],'action':args[1],'reward':args[2],
             'next_state':args[3],'prob':args[4],'done':args[5]}
        self.trajectories.append(t)
        
    def make_batch(self):
        states = tf.convert_to_tensor([t['state'] for t in self.trajectories],dtype=tf.float32)
        actions = tf.convert_to_tensor([[t['action']] for t in self.trajectories],dtype=tf.int32)
        rewards = tf.convert_to_tensor([[t['reward']] for t in self.trajectories],dtype=tf.float32)
        next_states = tf.convert_to_tensor([t['next_state'] for t in self.trajectories],dtype=tf.float32)
        probs = tf.convert_to_tensor([[t['prob']] for t in self.trajectories],dtype=tf.float32)
        dones = tf.convert_to_tensor([[float(t['done'])] for t in self.trajectories],dtype=tf.float32)
        self.trajectories = []
        return states,actions,rewards,next_states,probs,dones

    def compute_gae(self,states, next_states, rewards, dones):
        q_targets = rewards + self.gamma * self.critique(next_states) * (1-dones)
        deltas = q_targets - self.critique(states)
        deltas = deltas.numpy()
        advantage_lst = []
        advantage = 0.0
        for delta_t in deltas[::-1]:
            advantage = (self.gamma * self.lamb * advantage) + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = tf.convert_to_tensor(advantage_lst, dtype=tf.float32)
        return advantages
        
    def train(self, update_times=3):
        states,actions,rewards,next_states,probs,dones  = self.make_batch()
        for i in range(update_times):
            #q targets to optimize critic
            q_targets = rewards + self.gamma * self.critique(next_states) * (1-dones)
            #advantages to optimize actor
            advantages = self.compute_gae(states,next_states,rewards,dones)

            #train
            with tf.GradientTape() as tape:
                preds = self.act(states)
                action_idxs = tf.stack([tf.range(tf.shape(actions)[0]),actions[:,0]],axis=-1)
                preds_a = tf.gather_nd(preds,action_idxs)[:,None]
                ratio = tf.math.exp(tf.math.log(preds_a) - tf.math.log(probs)) 
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = tf.math.minimum(surr1,surr2)
                critic_loss_fn = tf.keras.losses.MeanSquaredError()
                critic_loss = critic_loss_fn(q_targets, self.critique(states))
                loss = -actor_loss + critic_loss
                loss = tf.reduce_mean(loss)
                gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients,self.trainable_weights))
            self.loss_lst.append(loss.numpy())


class PPOBernoulli(Model):
    def __init__(self, input_dim=(10, 4), hidden_dim=30, output_dim=1,
                lr = 5e-4, gamma=1, lamb=0.95, eps_clip=0.1):
        super(PPOBernoulli, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.lamb = lamb
        self.eps_clip = eps_clip

        #feature extraction
        self.features = tf.keras.Sequential()
        self.features.add(layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True), input_shape=input_dim))

        #actor
        self.actor = tf.keras.Sequential()
        self.actor.add(layers.Dense(output_dim,activation='sigmoid'))

        #critic
        self.critic = tf.keras.Sequential()
        self.critic.add(layers.Flatten())
        self.critic.add(layers.Dense(1))
        
        self.loss_lst = []
        self.trajectories = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def act(self,x):
        x = self.features(x)
        prob = self.actor(x)
        return tf.squeeze(prob,-1)

    def critique(self, x):
        x = self.features(x)
        v = self.critic(x)
        return v

    def add(self, *args):
        t = {'state':args[0],'action':args[1],'reward':args[2],
             'next_state':args[3],'prob':args[4],'done':args[5]}
        self.trajectories.append(t)

    def make_batch(self):
        states = tf.convert_to_tensor([t['state'] for t in self.trajectories],dtype=tf.float32)
        actions = tf.convert_to_tensor([t['action'] for t in self.trajectories],dtype=tf.int32)
        rewards = tf.convert_to_tensor([[t['reward']] for t in self.trajectories],dtype=tf.float32)
        next_states = tf.convert_to_tensor([t['next_state'] for t in self.trajectories],dtype=tf.float32)
        probs = tf.convert_to_tensor([t['prob'] for t in self.trajectories],dtype=tf.float32)
        dones = tf.convert_to_tensor([[float(t['done'])] for t in self.trajectories],dtype=tf.float32)
        self.trajectories = []
        return states,actions,rewards,next_states,probs,dones

    def compute_gae(self,states, next_states, rewards, dones):
        q_targets = rewards + self.gamma * self.critique(next_states) * (1-dones)
        deltas = q_targets - self.critique(states)
        deltas = deltas.numpy()
        advantage_lst = []
        advantage = 0.0
        for delta_t in deltas[::-1]:
            advantage = (self.gamma * self.lamb * advantage) + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = tf.convert_to_tensor(advantage_lst, dtype=tf.float32)
        return advantages

    def train(self, update_times=3):
        states,actions,rewards,next_states,probs,dones  = self.make_batch()
        for i in range(update_times):
            #q targets to optimize critic
            q_targets = rewards + self.gamma * self.critique(next_states) * (1-dones)
            #advantages to optimize actor
            advantages = self.compute_gae(states,next_states,rewards,dones)

            #train
            with tf.GradientTape() as tape:
                preds = self.act(states)
                preds_a = tf.cast(actions,tf.float32) * preds + (1-tf.cast(actions,tf.float32)) * (1-preds)
                ratio = tf.reduce_sum(tf.math.exp(tf.math.log(preds_a) - tf.math.log(probs)),1)[:,None]
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = tf.reduce_mean(tf.math.minimum(surr1,surr2))
                critic_loss_fn = tf.keras.losses.MeanSquaredError()
                critic_loss = critic_loss_fn(q_targets, self.critique(states))
                loss = -actor_loss + critic_loss
                gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients,self.trainable_weights))
            self.loss_lst.append(loss.numpy())