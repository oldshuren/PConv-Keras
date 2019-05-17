Modified the code to use tf.keras.

The purpose is to get more distributed training support.

In tensoflow 1.13.1, python/keras/engine/base_layer.py has a bug, it needs to be patched to run the code. Just apply tensorflow-1.13.1-base_layer.py.diff
