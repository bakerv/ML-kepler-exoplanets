import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf
import tensorflow_addons as tfa

class CustomHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyper_parameters.Choice(
            'batch_size',
            values = [17, 85, 221, 663, 1326])
        super(CustomHyperband, self).run_trial(trial, *args, **kwargs)   
    
def model_tuner(hyper_parameters):
    # tuning parameters
    dense_layer_1 = hyper_parameters.Choice('dense_1',
                                      values = [32, 64, 128, 
                                                256, 320])
    dense_layer_2 = hyper_parameters.Choice('dense_2',
                                      values = [32, 64, 128, 
                                                256, 320])
    test_learning_rates = hyper_parameters.Choice('learning_rate',
                                                  values = [ 0.001, 0.01,
                                                            0.05, 0.1])
    
    rectified_adam = tfa.optimizers.RectifiedAdam(
        learning_rate = test_learning_rates)
    ranger = tfa.optimizers.Lookahead(rectified_adam,
                                      sync_period = 6,
                                      slow_step_size = 0.5)
    mish = tfa.activations.mish

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = dense_layer1, activation = mish, input_dim=33),
        tf.keras.layers.Dense(units = dense_layer2, activation = mish),
        tf.keras.layers.Dense(1, activation = mish)
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
        
