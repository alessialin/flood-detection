import keras.backend as K
import tensorflow as tf


class LossFunctions:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    

    def DiceLoss_square(self, smooth=1):
        #create the missing data mask
        mask = tf.math.not_equal(self.y_true, 255)
        #apply the mask
        y_true = tf.boolean_mask(self.y_true, mask)
        y_pred = tf.boolean_mask(self.y_pred, mask)

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true_f * y_pred_f))
        dice_loss_square = 1-(
                (2. * intersection + smooth) / (K.sum(K.square(y_true_f), -1) 
                    + K.sum(K.square(y_pred_f),-1) 
                    + smooth
                    ))
        return dice_loss_square


    def DiceLoss(self, smooth=1):
        y_true_f = K.flatten(self.y_true)
        y_pred_f = K.flatten(self.y_pred)
        intersection = K.sum(self.y_true * self.y_pred)
        dice_loss = 1-((2. * intersection + smooth) 
            / (K.sum(y_true_f) + K.sum(y_pred_f)+ smooth)
                )
        return dice_loss

 
    def tversky(self, alpha=0.75, smooth=1):
        y_true_pos = K.flatten(self.y_true)
        y_pred_pos = K.flatten(self.y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        tversky_coeff = (true_pos + smooth)/(
                true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth
            )
        return tversky_coeff

    
    def tversky_loss(self):
        return 1 - LossFunctions.tversky(self.y_true, self.y_pred)


    def focal_tversky(self, gamma = 0.75):
        pt_1 = LossFunctions.tversky(self.y_true, self.y_pred)
        return K.pow((1-pt_1), gamma)
