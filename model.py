import os
import tensorflow as tf
from flask import redirect

import numpy as np
import PIL.Image
import imageio

def run():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(
          gpus[0],
          [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)

  def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

  def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

  content_image = load_img('static/uploads/content.jpeg')
  style_image = load_img('static/uploads/style.jpeg')


  x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
  x = tf.image.resize(x, (224, 224))
  vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
  prediction_probabilities = vgg(x)
  prediction_probabilities.shape

  predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
  [(class_name, prob) for (number, class_name, prob) in predicted_top_5]

  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

  content_layers = ['block5_conv2'] 

  style_layers = ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1', 
                  'block4_conv1', 
                  'block5_conv1']

  num_content_layers = len(content_layers)
  num_style_layers = len(style_layers)

  def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

  style_extractor = vgg_layers(style_layers)
  style_outputs = style_extractor(style_image*255)

  def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

  class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
      super(StyleContentModel, self).__init__()
      self.vgg =  vgg_layers(style_layers + content_layers)
      self.style_layers = style_layers
      self.content_layers = content_layers
      self.num_style_layers = len(style_layers)
      self.vgg.trainable = False

    def call(self, inputs):
      "Expects float input in [0,1]"
      inputs = inputs*255.0
      preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
      outputs = self.vgg(preprocessed_input)
      style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])

      style_outputs = [gram_matrix(style_output)
                      for style_output in style_outputs]

      content_dict = {content_name:value 
                      for content_name, value 
                      in zip(self.content_layers, content_outputs)}

      style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}
      
      return {'content':content_dict, 'style':style_dict}

  extractor = StyleContentModel(style_layers, content_layers)

  results = extractor(tf.constant(content_image))

  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  image = tf.Variable(content_image)

  def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
  style_weight=1e-1
  content_weight=1e4

  def style_content_loss(outputs):
      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
      style_loss *= style_weight / num_style_layers

      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                              for name in content_outputs.keys()])
      content_loss *= content_weight / num_content_layers
      loss = style_loss + content_loss
      return loss

  @tf.function()
  def train_step(image):
    with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

  epochs = 1000

  for n in range(epochs):
      print('epoch {}'.format(n))
      train_step(image)

  tensor_to_image(image).save('static/result/result.jpeg')
  return redirect('/Result')
