import tensorflow as tf
from abstractdecoder.cfg import ModelCFG

class ClassificationMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes=ModelCFG.NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.true_positive = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.false_positive = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.false_negative = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")
        print(type(self.true_positive[0]))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        for i in range(self.num_classes):
            true_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, i), tf.equal(y_true, i)), dtype=tf.float32))
            # true_neg = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred, i), tf.not_equal(y_true, i)), dtype=tf.float32))
            false_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, i), tf.not_equal(y_true, i)), dtype=tf.float32))
            false_neg = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.not_equal(y_pred, i)), dtype=tf.float32))

            tf.keras.backend.update(self.true_positive[i], self.true_positive[i] + true_pos)
            tf.keras.backend.update(self.false_positive[i], self.false_positive[i] + false_pos)
            tf.keras.backend.update(self.false_negative[i], self.false_negative[i] + false_neg)
    
    def result(self):
        precision = self.true_positive / (self.true_positive + self.false_positive + tf.keras.backend.epsilon()) # + epsilon term in case the denominator = 0
        recall = self.true_positive / (self.true_positive + self.false_negative + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall)  

        metrics = {}
        for i in range(self.num_classes):
            metrics[f"class_{i + 1}"] = {
                "precision" : precision[i],
                "recall" : recall[i],
                "f1_score" : f1[i]
            }
        
        return metrics