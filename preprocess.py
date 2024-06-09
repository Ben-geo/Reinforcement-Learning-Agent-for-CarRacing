import cv2
import tensorflow as tf
import numpy as np
class PreProcess:

    def process(self, state ):
        x, y, _ = state.shape
        cropped = state[ 0:int( 0.85*y ) , 0:x ]
        mask = cv2.inRange( cropped,  np.array([100, 100, 100]),  # dark_grey
                                    np.array([150, 150, 150]))  # light_grey
        gray = cv2.cvtColor( state, cv2.COLOR_BGR2GRAY )
        gray = gray/255
        gray[85:100, 0:12] = 0
        return np.expand_dims( gray, axis=2 ) 
preProcess = PreProcess()