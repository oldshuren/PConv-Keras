import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from argparse import ArgumentParser
from libs.util import MaskGenerator
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime
import os
import numpy as np

def server_it(args):
    predictor = tf.contrib.predictor.from_saved_model(export_dir=args.model_dir)

    input_img = image.load_img(args.image, target_size=[512,512])
    input_img = image.img_to_array(input_img)
    input_img *= 1./255

    mask_generator = MaskGenerator(512, 512, 3)
    mask = mask_generator.sample(1212)
    masked = deepcopy(input_img)
    masked[mask==0] = 1
    result = predictor({"inputs_img":[masked],
                        "inputs_mask":[mask]}
                       )
    pred_img = result['outputs'][0]
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked)
    axes[1].imshow(pred_img)
    axes[2].imshow(input_img)
    axes[0].set_title('Masked Image')
    axes[1].set_title('Predicted Image')
    axes[2].set_title('Original Image')

    output_path = os.path.join('.', 'output_img_{}.png'.format(pred_time))
    plt.savefig(output_path)
    print("output to {}".format(output_path))

    plt.close()


def parse_args():
    parser = ArgumentParser(description='Server Exported model of PConv inpainting')

    parser.add_argument(
        '-model_dir', '--model_dir',
        type=str, default=None,
        help='export directory',
    )
    parser.add_argument(
        '-image', '--image',
        type=str, default=None,
        help='test image',
    )
    
    return  parser.parse_args()

# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
    server_it(args)
