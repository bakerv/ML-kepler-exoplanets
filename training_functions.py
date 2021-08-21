import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf
import tensorflow_addons as tfa

class CustomHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Choice(
            'batch_size',
            values = [17])
        super(CustomHyperband, self).run_trial(trial, *args, **kwargs)   
    
def model_tuner(hyper_parameters):
    # tuning parameters
    look_ahead_sync = hyperparameters.Choice('sync',
                                             values=[3,6,9])
    look_ahead_step_size = hyperparameters.Choice('stepsize',
                                                  values = [0.1,0.3,
                                                            0.5,0.9])
    rectified_adam = tfa.optimizers.RectifiedAdam(
        learning_rate = test_learning_rates)
    ranger = tfa.optimizers.Lookahead(rectified_adam,
                                      sync_period = look_ahead_sync,
                                      slow_step_size = look_ahead_step_size)
    mish = tfa.activations.mish
    dense_layer_1 = hyper_parameters.Choice('dense_1',
                                            values = [64, 128])
    drop_out_rate = hyper_parameters.Choice('dropout',
                                            values= 0.4,0.5,0.6)
    dense_layer_2 = hyper_parameters.Choice('dense_2',
                                            values = [32, 64, 128, 
                                                      256, 320])
    test_learning_rates = hyper_parameters.Choice('learning_rate',
                                                  values = [1e-1, 1e-2, 4e-2,
                                                            5e-2, 1e-3, 1e-4])
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = dense_layer_1, activation = mish, input_dim=33),
        tf.keras.layers.Dropout(rate = drop_out_rate)
        tf.keras.layers.Dense(units = dense_layer_2, activation = mish),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(
        optimizer = ranger,
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    return model 

def plot_training(model_history, measure):
    plt.plot(model_history.history[measure])
    plt.plot(model_history.history['val_'+measure])
    plt.xlabel('Epochs')
    plt.ylabel(measure)
    plt.legend(['training '+ measure,'validation '+ measure])
    plt.show()
        
