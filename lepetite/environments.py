import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

class LayerRemoverEnv(object):
	def __init__(self, parent_class, trains, valids, tests, 
	             batch_input_shape = (500,28,28,1), max_t=5):
		self.parent_class = parent_class
		self.trains = trains
		self.valids = valids
		self.tests = tests
		self.batch_input_shape = batch_input_shape
		self.reset()
  
		#episode
		self.max_t = max_t

		#parent model
		self.parent_model = self.parent_class()
		self.parent_model.build(self.batch_input_shape)

		#get the benchmarks from parent model
		self.acc_test_parent, self.nb_params_parent = self.train_parent()

	def get_state(self):
		return tf.convert_to_tensor(self.child_model.child_states, dtype=tf.float32)

	def reset(self):
		#reset timestep
		self.t = 0

		#records for debugging
		self.epoch_train_losses = []
		self.epoch_valid_losses = []

		#datasets
		self.train_ds = tf.data.Dataset.from_tensor_slices(self.trains)\
			.shuffle(buffer_size=self.trains[1].shape[0]).batch(self.batch_input_shape[0])
		self.valid_ds = tf.data.Dataset.from_tensor_slices(self.valids)\
			.batch(self.batch_input_shape[0])
		self.test_ds = tf.data.Dataset.from_tensor_slices(self.tests)\
			.batch(self.batch_input_shape[0])

		#loss functions
		self.cce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.mse_fn = tf.keras.losses.MeanSquaredError()
	
		#child model
		self.child_model = self.parent_class()
		self.child_model.build(self.batch_input_shape)

		return self.get_state()
	
	def step(self, action):
		state = self.get_state()
		#create child model
		#check if it has more params than parent
		self.child_model = self.parent_class(action)
		self.child_model.build(self.batch_input_shape)
		if self.child_model.count_params() > self.parent_model.count_params():
			reward = -1
			nb_params, acc_test = self.child_model.count_params(), 1e-3
		else: 
			acc_test, nb_params = self.train_child(action)
			reward = self.get_reward(acc_test, nb_params)
		next_state = self.get_state()
		self.t+=1
		if self.t==self.max_t:
			done = 1.
		else:
			done = 0.
		info = {'compression ratio': self.nb_params_parent/nb_params, 
		        'accuracy ratio':acc_test/self.acc_test_parent}
		return next_state, reward, done, info

	def train_child(self, action, nb_epoch=1, lamb=0.9, verbose=False):
		optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, amsgrad=True)
		start_time = datetime.now()
		for e in range(nb_epoch):
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
			if verbose:
				print(f'epoch {e} - Train Loss: {np.mean(epoch_train_loss)};\
				Valid Loss: {np.mean(epoch_valid_loss)};\
				Valid Acc: {(tf.argmax(preds,1)==tf.cast(y,tf.int64)).numpy().mean()}')
		if verbose: print(f'Training done in {datetime.now() - start_time}')
		#test
		x_test, y_test = next(iter(self.test_ds))
		preds_test = self.child_model(x_test)
		acc_test = (tf.argmax(preds_test,1)==tf.cast(y_test,tf.int64)).numpy().mean()
		#record
		return acc_test, self.child_model.count_params()

	def train_parent(self, nb_epoch=1, verbose=True):
		optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, amsgrad=True)
		start_time = datetime.now()
		for e in range(nb_epoch):
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
			valid_preds = []
			valid_ys = []
			for i,(x,y) in enumerate(self.valid_ds):
				epoch_valid_loss = []
				preds = self.parent_model(x)
				valid_loss = self.cce_fn(y,preds)
				valid_preds.append(preds)
				valid_ys.append(y)
				epoch_valid_loss.append(valid_loss)
			self.epoch_valid_losses.append(np.mean(epoch_valid_loss))

			#concat valid preds and ys
			valid_preds = tf.concat(valid_preds,0)
			valid_ys = tf.concat(valid_ys,0)
			#log
			if verbose:
				print(f'epoch {e} - Train Loss: {np.mean(epoch_train_loss)};\
				Valid Loss: {np.mean(epoch_valid_loss)};\
				Valid Acc: {(tf.argmax(valid_preds,1)==tf.cast(valid_ys,tf.int64)).numpy().mean()}')
		if verbose: print(f'Training done in {datetime.now() - start_time}')

		#test loop
		test_preds = []
		test_ys = []
		for i,(x,y) in enumerate(self.test_ds):
			preds = self.parent_model(x)
			test_preds.append(preds)
			test_ys.append(y)
		#concat test preds and ys
		test_preds = tf.concat(test_preds,0)
		test_ys = tf.concat(test_ys,0)
		acc_test = (tf.argmax(test_preds,1)==tf.cast(test_ys,tf.int64)).numpy().mean()
  
		return acc_test, self.parent_model.count_params()

	def get_reward(self, acc_test, nb_params):
		c = 1 - (nb_params/self.nb_params_parent)
		R_c = c*(2-c)
		R_a = acc_test / self.acc_test_parent
		return R_c * R_a
