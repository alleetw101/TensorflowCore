from euroSAT import euroSATModel
import tensorflow

def print_hi(name):
    print(f'Hi, {name}')

if __name__ == '__main__':
    sample = tensorflow.keras.models.load_model('euroSAT/euroSAT_SavedModel20201223')
    sample.summary()

