import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from model import *
import matplotlib.pyplot as plt
from IPython import embed
import freeze


IMAGE_HEIGHT = 96 # Image height
IMAGE_WIDTH = 96 # Image hidth 
CHANNELS = 3 # Number of channels (3 for rgb)
BATCH_SIZE = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE

def apply_image_augmentation(image):

    #mean, var = tf.nn.moments(image, axes=[0,1])

    image = tf.image.random_brightness(
        image,
        max_delta = 40,
        seed=None
    )
    image += 10
    
    image = tf.image.random_hue(
        image,
        max_delta = 0.2, #max_delta must be in the interval [0, 0.5]
        seed=None
    )

    '''
    image = tf.image.random_saturation( #Multiplies saturation channel 
        image,
        lower = 0.99,
        upper = 1.01,

    )
    '''
   
    '''
    image = tf.image.random_contrast(
        image,
        lower = 0.95,
        upper = 1.0,
    )
    '''

    ''' - This causes the input pipeline to break
    #Insert Jpeg noise randomly
    image = tf.image.random_jpeg_quality(
        image,
        min_jpeg_quality = 90, #Scale of [0, 100]
        max_jpeg_quality = 100,
        seed=None
    )
    '''
    '''
    if random.randint(0,2) == 0:
        image = tf.image.flip_left_right(image)
    if random.randint(0,2) == 0:
        image = tf.image.flip_up_down(image)
    '''
    image = tf.clip_by_value(image, 0, 255)
    #image = tf.dtypes.cast(image, tf.uint8)
    #image = tf.math.floor(image)
    return image


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    #image = tf.image.per_image_standardization(image)
    #image = tf.div(tf.subtract(image,tf.reduce_min(image)),tf.subtract(tf.reduce_max(image),tf.reduce_min(image)))
    return image

def preprocess_image_mask(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], method = 1)
    #image = tf.image.per_image_standardization(image)
    #image = tf.div(tf.subtract(image,tf.reduce_min(image)),tf.subtract(tf.reduce_max(image),tf.reduce_min(image)))
    return image

def preprocess_train(path):
    image = tf.read_file(path)
    image = preprocess_image(image)
    image = apply_image_augmentation(image)
    return image

def preprocess_test(path):
    image = tf.read_file(path)
    image = preprocess_image(image)
    return image

def preprocess_train_mask(path):

    image = tf.read_file(path)
    image = preprocess_image_mask(image)
    image = image // 255
    image = tf.dtypes.cast(image, dtype = tf.int64)
    return image

def preprocess_test_mask(path):
    image = tf.read_file(path)
    image = preprocess_image_mask(image)
    image = image // 255
    image = tf.dtypes.cast(image, dtype = tf.int64)
    return image



def main():
    #Each subdirectory will get a numeric label by alphabetical order, starting with 0
    #For each subdirectory (class)
    train_meta_data = []
    test_meta_data = []

    counts = [0,0]


    dataset_path_train = "/data/fm_tools/autofm/temp13/train"
    dataset_path_train_masks = "/data/fm_tools/autofm/temp13/masks/train"
    classes = sorted(os.walk(dataset_path_train).__next__()[1])
    for label, c in enumerate(classes):
        c_dir = os.path.join(dataset_path_train, c)
        walk = os.walk(c_dir).__next__()
        file_list = walk[2]
        for sample in file_list:
            if sample.endswith('png'):  
                one_hot = np.zeros(2)
                one_hot[label] = 1
                mask_path = os.path.join(dataset_path_train_masks, c_dir.split("/")[-1], sample)
                if label == 0:
                    mask_path = '/data/fm_tools/autofm/temp13/clean_mask.png'
                train_meta_data.append([os.path.join(c_dir, sample), one_hot, mask_path])

    dataset_path_test = "/data/fm_tools/autofm/temp13/test"
    dataset_path_test_masks = "/data/fm_tools/autofm/temp13/masks/test"
    classes = sorted(os.walk(dataset_path_test).__next__()[1])
    for label, c in enumerate(classes):
        c_dir = os.path.join(dataset_path_test, c)
        walk = os.walk(c_dir).__next__()
        file_list = walk[2]
        for sample in file_list:
            if sample.endswith('png'):
                one_hot = np.zeros(2)
                one_hot[label] = 1  
                mask_path = os.path.join(dataset_path_test_masks, c_dir.split("/")[-1], sample)
                if label == 0:
                    mask_path = '/data/fm_tools/autofm/temp13/clean_mask.png'
                test_meta_data.append([os.path.join(c_dir, sample), one_hot, mask_path])

    '''
    We are shuffleing here as well as letting tf.dataset shuffle because the tf.dataset shuffle cannot
    do a uniform shuffle on large datasets
    '''
    random.shuffle(train_meta_data) 
    random.shuffle(test_meta_data)

    '''
    Create dataset pipeline. 
    The pipeline takes in a list of filenames and an array of labels (not one-hot).
    We are using a tensroflow reinitializable iterator, which allows us to switch between
    test and train datasets at train time by calling iterator.make_initializer().
    '''

    #Train
    image_ds_train = tf.data.Dataset.from_tensor_slices(np.array(train_meta_data)[:,0])
    image_ds_train = image_ds_train.map(preprocess_train, num_parallel_calls=12)
    mask_ds_train = tf.data.Dataset.from_tensor_slices(np.array(train_meta_data)[:,2])
    mask_ds_train = mask_ds_train.map(preprocess_train_mask, num_parallel_calls=12)
    #image_ds_train = image_ds_train.map(apply_image_augmentation, num_parallel_calls=12)
    label_ds_train = tf.data.Dataset.from_tensor_slices( np.stack(np.array(train_meta_data)[:,1], axis=0))

    #Test
    image_ds_test = tf.data.Dataset.from_tensor_slices(np.array(test_meta_data)[:,0])
    image_ds_test = image_ds_test.map(preprocess_test, num_parallel_calls=12)
    mask_ds_test = tf.data.Dataset.from_tensor_slices(np.array(test_meta_data)[:,2])
    mask_ds_test = mask_ds_test.map(preprocess_test_mask, num_parallel_calls=12)
    label_ds_test = tf.data.Dataset.from_tensor_slices( np.stack(np.array(test_meta_data)[:,1], axis=0))


    dataset_train = tf.data.Dataset.zip((image_ds_train, label_ds_train, mask_ds_train))
    dataset_train = dataset_train.shuffle(buffer_size=1000) #buffer_size determins uniformity of the shuffle
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_train = dataset_train.prefetch(buffer_size=AUTOTUNE)

    dataset_test = tf.data.Dataset.zip((image_ds_test, label_ds_test, mask_ds_test))
    dataset_test = dataset_test.shuffle(buffer_size=1000) #buffer_size determins uniformity of the shuffle
    dataset_test = dataset_test.batch(BATCH_SIZE)
    dataset_test = dataset_test.prefetch(buffer_size=AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_test.output_shapes)
    X_BATCH , Y_BATCH, Z_BATCH = iterator.get_next()
    training_init_op = iterator.make_initializer(dataset_train)
    validation_init_op = iterator.make_initializer(dataset_test)

    X_BATCH = tf.identity(X_BATCH, "images_input")    
    Y_BATCH = tf.identity(Y_BATCH, "labels_input") 
    Z_BATCH = tf.identity(Z_BATCH, "masks_input")     

    network = FCN_Model(X_BATCH, Z_BATCH) #X_BATCH is images Y_BATCH is labels
    
    #network.x = tf.identity(network.x, "kee_rate_input")
    output_op = tf.get_default_graph().get_operation_by_name("Softmax")                                                   
    #_ = tf.identity(output_op.outputs[0], "Softmax_output") 

    NUM_EPOCHS = 2000
    learning_rate = 5e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(network.loss)
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=500)


    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)    

    with sess.graph.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        saver.restore(sess, "model1221")
        print("Training started")
        for epoch in range(NUM_EPOCHS):

            #random.shuffle(train_meta_data) 
            #random.shuffle(test_meta_data)
    
            #Train epoch
            #Set to use train data
            sess.run(training_init_op)

            step = 0
            train_losses = []
            train_accuracies = []
            allLabels = []
            try:
                while True:
                    softmax, images, masks, train_loss, _, pred, accuracy = sess.run([network.softmax, network.original_image, network.mask, network.loss, optimizer, network.predictions, network.accuracy])
                    embed()
                  
                    train_losses.append(train_loss)    
                    train_accuracies.append(accuracy[0])

                    step += 1
            except tf.errors.OutOfRangeError: 
                #This looks like a hack but this is the correct way to tell when an epoch is complete
                pass
            
            print("Epoch" , epoch, "complete. Train accuracy for epoch was", np.mean(train_accuracies), "train loss was", np.mean(train_losses))
            #Test epoch
            #Set to use test_data

            sess.run(validation_init_op)

            test_accuracies = []
            try:
                while True:
                    pred, softmax, images, masks, test_loss, accuracy = sess.run([network.predictions, network.softmax, network.original_image, network.mask, network.loss, network.accuracy])
                    print("Test accuracy =", accuracy[0], end='\r')
                    test_accuracies.append(accuracy[0])


                    
                    cv2.imshow("prediction", np.squeeze(pred[0]).astype(np.uint8)*255)
                    cv2.imshow("softmax", (np.squeeze(softmax[0][:,:,1])*255).astype(np.uint8))
                    cv2.imshow("mask", np.squeeze(masks[0]).astype(np.uint8)*255)
                    cv2.imshow("image", np.squeeze(images[0]).astype(np.uint8)[...,::-1])
                    cv2.waitKey(0)
                    
                    print("Test accuracy =", "%.3f" % accuracy[0], "Test loss =", test_loss, end='\r')

            except tf.errors.OutOfRangeError: 
                pass

            print("Test accuracy after epoch was", np.mean(test_accuracies))
            saver.save(sess, "model1221")
            freeze.create_pb("model1221")

            print("Checkpoint saved and pb created.")   


if __name__== "__main__":
  main()















