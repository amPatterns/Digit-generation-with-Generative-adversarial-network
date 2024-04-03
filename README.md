# Digit-generation-with-Generative-adversarial-network
The aim of the project was to generate images of the digit 0, by using a generative adversarial network .
After training the  generator against the discriminator a few times, the results were promising.
This method can be used to generate any character, in order to create human like writings.

We used the mnist dataset to gather the images of zeros we needed.
the generator is made of an input layer , a dense layer with 128 neurons,a leaky relu layer , a dense output layer with 784 neurons.

The discriminator is made of an input layer , a dense layer with 300 neurons and a softmax layer.

The generator will be competing against the discriminator, in order to fool it, while the discriminator will try to learn how to distinguish between generated digits from real digits.

Our best results were obtained after training the generator against the discriminator and vice versa 500 times, each time until the log loss was under a threshold ( 0.1).The discriminator was fed batches of images  of fake or real zeros (unshuffled) . The input noise was random gaussian noice of dimension 100 .

The images obtained are very convising and show the power of the GAN models for image generation.
