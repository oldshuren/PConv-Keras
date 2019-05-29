import tensorflow as tf
from argparse import ArgumentParser

from libs.pconv_model import PConvUnet

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference

def export_it(args):
    model = PConvUnet(gpus=args.num_gpus, inference_only=True)
    model.load(args.checkpoint, train_bn=False)
    
    model = model.model
    print('input;{}, output:{}'.format(model.inputs, model.outputs))

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key

    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            args.export_dir,
            inputs={'inputs_img': model.inputs[0], 'inputs_mask': model.inputs[1]},
            outputs={'outputs': model.outputs[0]})


def parse_args():
    parser = ArgumentParser(description='Exporting model of PConv inpainting')

    parser.add_argument(
        '-checkpoint', '--checkpoint',
        type=str, default=None,
        help='Checkpoint to export',
    )
    parser.add_argument(
        '-export_dir', '--export_dir',
        type=str, default=None,
        help='export directory',
    )

    # checkpoint save with multiple gpu must be loaded with multiple gpu
    parser.add_argument(
        '-num_gpus', '--num_gpus',
        type=int, default=1,
        help='Number of GPUs to use'
    )


    return  parser.parse_args()

# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
    export_it(args)
