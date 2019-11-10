import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

class LayerRemover(object):
    def __init__(self, parent_class, trains, valids, tests, bs = 500):
        self.parent_class = parent_class
        self.trains = trains
        self.valids = valids
        self.tests = tests
        self.bs = bs
        self.reset()
        #parent model
        self.parent_model = self.parent_class()
        x,y = next(iter(self.train_ds))
        self.parent_model(x)

        #get the benchmarks from parent model
        self.acc_test_parent, self.nb_params_parent = self.train_parent()

    def get_state(self):
        return tf.convert_to_tensor(self.child_model.child_states)[None,:]

    def reset(self):
        #records for debugging
        self.epoch_train_losses = []
        self.epoch_valid_losses = []

        #datasets
        self.train_ds = tf.data.Dataset.from_tensor_slices(self.trains)\
            .shuffle(buffer_size=self.trains[1].shape[0]).batch(self.bs)
        self.valid_ds = tf.data.Dataset.from_tensor_slices(self.valids)\
            .batch(self.valids[1].shape[0])
        self.test_ds = tf.data.Dataset.from_tensor_slices(self.tests)\
            .batch(self.tests[1].shape[0])

        #loss functions
        self.cce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.mse_fn = tf.keras.losses.MeanSquaredError()
    
        #child model
        self.child_model = self.parent_class()
        x,y = next(iter(self.train_ds))
        self.child_model(x)

        return self.get_state()
    
    def step(self, action):
        state = self.get_state()
        acc_test, nb_params = self.train_child(action)
        reward = self.get_reward(acc_test, nb_params)
        next_state = self.get_state()
        return state, action, reward, next_state

    def train_child(self, action, nb_epoch=3, lamb=0.9):
        self.child_model = self.parent_class(action)
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, amsgrad=True)
        for e in tqdm(range(nb_epoch)):
            for i,(x,y) in enumerate(self.train_ds):
                epoch_train_loss = []
                preds_parent = self.parent_model(x) 
                with tf.GradientTape() as tape:
                    preds = self.child_model(x)
                    cce_loss = self.cce_fn(y, preds)
                    mse_loss = self.mse_fn(preds_parent, preds)
                    train_loss = lamb * mse_loss + (1-lamb) * cce_loss
                    epoch_train_loss.append(train_loss)
                    #record gradients
                    gradients = tape.gradient(train_loss, self.child_model.trainable_weights)
                #update 
                optimizer.apply_gradients(zip(gradients, self.child_model.trainable_weights))
            self.epoch_train_losses.append(np.mean(epoch_train_loss))
            #validation loop
            for i,(x,y) in enumerate(self.valid_ds):
                epoch_valid_loss = []
                preds = self.child_model(x)
                valid_loss = self.cce_fn(y,preds)
                epoch_valid_loss.append(valid_loss)
            self.epoch_valid_losses.append(np.mean(epoch_valid_loss))
            
            #log
            print(f'epoch {e} - Train Loss: {np.mean(epoch_train_loss)};\
            Valid Loss: {np.mean(epoch_valid_loss)};\
            Valid Acc: {(tf.argmax(preds,1)==tf.cast(y,tf.int64)).numpy().mean()}')
        print(f'Training done in {datetime.now() - start_time}')
        #test
        x_test, y_test = next(iter(self.test_ds))
        preds_test = self.child_model(x_test)
        acc_test = (tf.argmax(preds_test,1)==tf.cast(y_test,tf.int64)).numpy().mean()
        #record
        return acc_test, self.child_model.count_params()

    def train_parent(self, nb_epoch=3):
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, amsgrad=True)
        for e in tqdm(range(nb_epoch)):
            for i,(x,y) in enumerate(self.train_ds):
                epoch_train_loss = []
                with tf.GradientTape() as tape:
                    preds = self.parent_model(x) #prediction
                    train_loss = self.cce_fn(y,preds) #record loss
                    epoch_train_loss.append(train_loss)
                    #record gradients
                    gradients = tape.gradient(train_loss, self.parent_model.trainable_weights)
                #update 
                optimizer.apply_gradients(zip(gradients, self.parent_model.trainable_weights))
            self.epoch_train_losses.append(np.mean(epoch_train_loss))
            
            #validation loop
            for i,(x,y) in enumerate(self.valid_ds):
                epoch_valid_loss = []
                preds = self.parent_model(x)
                valid_loss = self.cce_fn(y,preds)
                epoch_valid_loss.append(valid_loss)
            self.epoch_valid_losses.append(np.mean(epoch_valid_loss))
            
            #log
            print(f'epoch {e} - Train Loss: {np.mean(epoch_train_loss)};\
            Valid Loss: {np.mean(epoch_valid_loss)};\
            Valid Acc: {(tf.argmax(preds,1)==tf.cast(y,tf.int64)).numpy().mean()}')
        print(f'Training done in {datetime.now() - start_time}')
        #test
        x_test, y_test = next(iter(self.test_ds))
        preds_test = self.parent_model(x_test)
        acc_test = (tf.argmax(preds_test,1)==tf.cast(y_test,tf.int64)).numpy().mean()
        #record
        return acc_test, self.parent_model.count_params()

    def get_reward(self, acc_test, nb_params):
        c = 1 - (nb_params/self.nb_params_parent)
        R_c = c*(2-c)
        R_a = acc_test / self.acc_test_parent
        return R_c * R_a