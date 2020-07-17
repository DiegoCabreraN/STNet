# Style Transfer Network

This project contains a **Flask** Application to serve a Neural Style Transfer Model developed during a tutorial on Tensorflow's page

The purpose of this is to create a simplistic way to interact with the model.

In order to run this project you'll need to set up a Tensorflow v2.2.0 environment, if you want to speed up the project be sure to have installed the Cudnn, Cuda, and/or CudaToolkit. I highly recommend to set all of this inside an Anaconda environment, since they have the option to install cudnn and Cudatoolkit directly into anaconda without having to specify PATH variables. And if you are using the GPU, be sure you have more than 4gb available.

You'll also need to install the following pip packages:

    - Tensorflow
    - Flask
    - Numpy
    - Pillow

Finally, In the application you can upload your own Style and Content images, but they are required to be `.jpeg` files.
Have fun testing the app.
