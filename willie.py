import tensorflow as tf
import os
import click
import numpy as np
import time
import datetime
import rasterio
from matplotlib import pyplot as plt
from IPython import display

# training_path = 'npzs/winter_train_norm.npz'
# testing_path = 'npzs/winter_test_norm.npz'
# steps = 40000
# test_images_path = 'imagery/climate_winters/test'
# temporary_folder = 'imagery/tmp/new6'
# output_image = '/Users/adriaankeurhorst/Documents/MScThesis/imagery/merged_40k_2018.tif'

# python script/willie.py npzs/winter_train_norm.npz npzs/winter_test_norm.npz 25000 imagery/climate_winters/test imagery/tmp/new5 imagery/merged_25k_2018.tif

def cGAN(training_path, testing_path, steps, test_images_path, temporary_folder, output_image):
  def load_data(training_path, testing_path):
    data = np.load(training_path)
    train_lcm = data['arr_0']
    train_lcm = train_lcm.transpose((0, -1, -2, -3))
    train_sen2 = data['arr_1']
    train_sen2 = train_sen2.transpose((0, -1, -2, -3))

    test = np.load(testing_path)
    test_lcm = test['arr_0']
    test_lcm = test_lcm.transpose((0, -1, -2, -3))
    test_sen2 = test['arr_1']
    test_sen2 = test_sen2.transpose((0, -1, -2, -3))

    # The training set consist of 2552 images
    BUFFER_SIZE = train_sen2.shape[0]
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    # Each image is 256x256 in size
    IMG_WIDTH = train_sen2.shape[1]
    IMG_HEIGHT = train_sen2.shape[2]
    OUTPUT_CHANNELS = train_sen2.shape[-1]

    train_lcm = tf.data.Dataset.from_tensor_slices((train_lcm))
    train_lcm = train_lcm.batch(BATCH_SIZE)
    train_sen2 = tf.data.Dataset.from_tensor_slices((train_sen2))
    train_sen2 = train_sen2.batch(BATCH_SIZE)

    test_lcm = tf.data.Dataset.from_tensor_slices((test_lcm))
    test_lcm = test_lcm.batch(BATCH_SIZE)
    test_sen2 = tf.data.Dataset.from_tensor_slices((test_sen2))
    test_sen2 = test_sen2.batch(BATCH_SIZE)
    return IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS, train_lcm, train_sen2, test_lcm, test_sen2
  print('starting the deception')
  IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS, train_lcm, train_sen2, test_lcm, test_sen2 = load_data(training_path, testing_path)
  print('time to load.')
  def prepare(train_lcm, train_sen2, test_lcm, test_sen2):
    train_dataset = tf.data.Dataset.zip((train_lcm, train_sen2))
    test_dataset = tf.data.Dataset.zip((test_lcm, test_sen2))

    return train_dataset, test_dataset

  train_dataset, test_dataset = prepare(train_lcm, train_sen2, test_lcm, test_sen2)
  print('datasets loaded. pray nothing crashes')
  def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

  def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

  def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 16], dtype='float32')

    down_stack = [
      downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
      downsample(128, 4),  # (batch_size, 64, 64, 128)
      downsample(256, 4),  # (batch_size, 32, 32, 256)
      downsample(512, 4),  # (batch_size, 16, 16, 512)
      downsample(512, 4),  # (batch_size, 8, 8, 512)
      downsample(512, 4),  # (batch_size, 4, 4, 512)
      downsample(512, 4),  # (batch_size, 2, 2, 512)
      downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
      upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
      upsample(512, 4),  # (batch_size, 16, 16, 1024)
      upsample(256, 4),  # (batch_size, 32, 32, 512)
      upsample(128, 4),  # (batch_size, 64, 64, 256)
      upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh', dtype='float32')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
  generator = Generator()
  print('generator configured')
  LAMBDA = 100
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

  def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 16], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 7], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar], axis=-1)  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer, dtype='float32')(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  discriminator = Discriminator()
  print('discriminator configured')
  def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(17, 12))

    display_list = [test_input[0,:,:,0], tar[0,:,:,1:4], prediction[0,:,:,1:4]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    plt.show()

  log_dir="logs/"

  summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  @tf.function
  def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, training=True)

      disc_real_output = discriminator([input_image, target], training=True)
      disc_generated_output = discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
      tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

  def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
      if (step) % 1000 == 0:
        display.clear_output(wait=True)
        if step != 0:
          print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

        start = time.time()

        print(f"Step: {step//1000}k. how many steps have you taken today?")

      train_step(input_image, target, step)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)


      # Save (checkpoint) the model every 5k steps
      if (step + 1) % 5000 == 0:
        # generate_images(generator, example_input, example_target)
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("\n\n\n\nsaved a checkpoint bro\n\n\n\n")

  del test_lcm, train_lcm, train_sen2, test_sen2

  fit(train_dataset, test_dataset, steps=steps)

  # Restoring the latest checkpoint in checkpoint_dir
  # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  def generate_images1(model, test_input):
    prediction = model(test_input, training=True)
    return prediction

  def gtiff(dset, test_imagery_path, out_path, merged_path):
    lst = []
    for example_input, example_target in dset.take(len(dset)):
      lst += [generate_images1(generator, example_input)]
    lst = np.array(lst)
    lst_min = np.min(lst)
    lst_max = np.max(lst)
    print('Loaded list to arrrrrrrr')
    for i in range(len(dset)):
      gen_out = lst[i]
      out_path1 = os.path.join(out_path, f'test{i + 1}.tif')
      test_im = rasterio.open(os.path.join(test_imagery_path, os.listdir(test_imagery_path)[i]))
      eee = gen_out.transpose()
      eee = np.reshape(eee, (7, 256, 256))
      eee1 = ((eee - lst_min) / (lst_max - lst_min)) * 10
      try:
        # Create empty TIF image with dimensions of FIN but with name of FOUT.
        with rasterio.open(
                out_path1,
                'w',
                driver='GTiff',
                height=256,
                width=256,
                count=7,
                dtype='float32',
                crs=test_im.crs,
                transform=test_im.transform
        ) as dst:
          dst.write(eee1)
          print(f"File created: {out_path1}")
      except IOError as e:
        print(f"Couldn't write a file at {out_path1}. Error: {e}")
    print("\n\n\ngotta make a single big image now\n\n\ndo some exercises\n\n")
    cmd = f'gdal_merge.py -ot Float32 -of GTiff -o {merged_path} {out_path}/*.tif'
    os.system(cmd)

  gtiff(test_dataset, test_images_path, temporary_folder, output_image)


@click.command()
@click.argument('training_path', type=click.Path(exists=True))
@click.argument('testing_path', type=click.Path(exists=True))
@click.argument('steps', type=int)
@click.argument('test_images_path', type=click.Path(exists=True))
@click.argument('temporary_folder', type=click.Path(exists=True))
@click.argument('output_image', type=click.Path())

def init(training_path, testing_path, steps, test_images_path, temporary_folder, output_image):
    cGAN(training_path, testing_path, steps, test_images_path, temporary_folder, output_image)

if __name__ == "__main__":
    init()


