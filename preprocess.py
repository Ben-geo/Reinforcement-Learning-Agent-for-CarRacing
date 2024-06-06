import cv2
import tensorflow as tf

class PreProcess:

    def process(self,state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = tf.expand_dims(state,-1)
        return state
preProcess = PreProcess()