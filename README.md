# **Autonomous Simulator Driving Using Convolutional Neural Networks** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[flipped]: ./examples/flipped.png "flipped"
[centerlane]: ./examples/centerlane.png "CenterLane"
[recovery]: ./examples/recovery.png "Recovery Image"
[left]: ./examples/left.png "left Image"
[center]: ./examples/center.png "center Image"
[right]: ./examples/right.png "right Image"
[simulator]: ./examples/simulator.jpg "simulator Image"
[rightcamunflipped]: ./examples/rightcamunflipped.png "unflipped image"
[title]: ./examples/title.png "title image"


## [Watch the successful autonomous lap around the track!](https://youtu.be/UvFYcYJkhoM)

[![alt text][title]](https://youtu.be/ftG8raDrU0Q)

## [Here is the cockpit view version](https://youtu.be/UvFYcYJkhoM)

### **Rubric Points**
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator ( [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip), [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip), [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip) ) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
and then opening the simulator program for your operating system and selecting the `AUTONOMOUS MODE`.

![alt text][simulator]

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with a lambda layer for normalizing the images, then three layers of 5x5 filters and two layers of 3x3 filters. The filter depths begin at 24 and end at 64. Finally I finish with four flattened step down layers of depth 100, 50, 10, and 1. This is based on the Nvidia end-to-end training model. (model.py lines 107-136)

To ensure the model can handle non-linearities each convolutional layer is activated by a rectified linear unit (ReLU). ReLU layers ensure that the derivative can be back-propogated for error reduction between epochs. This is the key to iterative learning of the model.

#### 2. Attempts to reduce overfitting in the model

Overfitting was a big problem on my first iterations of the model. I was getting very good performance on the training data but the validation never improved. I fixed this by adding very aggressive dropout to the model, .25 between each layer. This improved overall performance tremendously.

I split the data set into 80% training and 20% validation sets (model.py line 41). 
```
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```
This gave me a set of data that was never used for training the model to ensure I was not overfitting to the training data. Normally I would set aside a test set of data that I would use only once when I was fully satisfied with my model performance. In this case the simulator in autonomous mode served as the test set. 

#### 3. Model parameter tuning

I leaned on previous experience with training models and decided to use an Adam optimizer which does not require a learning rate. This is due to the Adam optimizer's algorithm which defaults to a learning rate of 0.001 and decays throughout the iteration process (model.py line 142).

The main thing I tuned was the number of layers, dropout, and the filter depth of each layer. I can't stress enough how well the aggressive dropout worked to generalize the model. I also tuned batch size and epochs but they had less impact than the dropout. Lastly, I tuned the error function, this also helped a bit.

#### 4. Appropriate training data

I chose my training data initially by just driving on track 1 counter-clockwise and then clockwise. I thought that the better I drive the car the better the model will. That quickly changed after the first test in autonomous mode. I realized that was not a good strategy. The car hugged the edge of the lane and mostly went off the track.

The following section details the steps I took after the initial failure.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I approached this solution like most things I do. I began with something that already existed and tried it. Then I tweaked it and expanded based on what I have learned. 

I began my quest for finding a model architecture with what I new already from my Traffic Sign Classifier, LeNet. LeNet was a very good architecture for sign image classification so it seemed like a great start.

I knew it was not going to be the perfect model after my first round of validation. I ran the error as a mean squared and found that it was low for training but an order of magnitude higher for validation. This meant I was overfitting the training data which did not generalize well to new data.

This is when I realized I needed a trick to overcome the overfitting. I did a bit of research and found that dropout is a very strong combatant against overfitting. So I added a few layers of dropout at .5 and tested again. Validation error got better but still not good enough. So I went heavier and added dropout between every layer at .25. This brought the training and validation errors into the same ballpark.

Once all this was implemented the car began driving pretty well in the straights. However it still didn't do great in the turns. In the next few sections I will discuss how I fixed this.

By the end the car was able to drive around track 1 in both directions without going off the road and even with adding manual perturbations to try to send it off the track!

#### 2. Final Model Architecture

The final model architecture (model.py lines 107-136) consisted of a convolution neural network with the following layers and layer sizes ...

```
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(row, col, ch)))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu', init='normal'))
    model.add(Dropout(.25))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu', init='normal'))
    model.add(Dropout(.25))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu', init='normal'))
    model.add(Dropout(.25)
    model.add(Conv2D(64,3,3, activation='relu', init='normal'))
    model.add(Dropout(.25))
    model.add(Conv2D(64,3,3, activation='relu', init='normal'))
    model.add(Dropout(.25)
    model.add(Flatten())
    model.add(Dense(100, init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(50, init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(10, init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(1, init='normal'))
```

#### 3. Creation of the Training Set & Training Process

As mentioned before I began by teaching the model to drive in the center of the lane. It looked something like this:

![alt text][centerlane]

This led to the car hugging a side and always going off the course.

So I then took the advice from class and started teaching some recovery which was meant to show that in certain cases drastic steering angles were required to bring back to center. This way the model did think only small corrections were allowed. Here is an image starting a recovery from the left:

![alt text][recovery]

I also trained the car to properly turn by creeping very slowly through turns which gave me a lot of images in turns to train the model. I especially had to show it many examples of the turn with the dirt road on the side. It kept trying to veer off the road into the dirt.

I did some data augmentations as recommended in class. I doubled the data set by flipping all images and keeping the originals. The rationale for this was that I could drive one direction around the track and it would learn to steer in both directions instead of just turning left. I had to multiply the steer angle by -1 when the image was flipped so it knew the proper way to turn in those cases.

Original: ![alt text][rightcamunflipped] Flipped:![alt text][flipped]

I also added all three camera angles adding a correction factor of 0.05 plus for the left camera and minus for the right camera. This tripled the already doubled dataset. So in total I had six times the image data than the original set. Here are the three views:

left cam:![alt text][left]center cam:![alt text][center] right cam:![alt text][right]

I shuffled the data set each time prior to training and set aside 20% for validation. I used a batch size of  `Batch size =  6  * 3cams  * 2 for flips =  36`. Since I used an Amazon Web Services GPU I was able to run 10 epochs very quickly. Error was continually reduces across the 10 epochs with a few exceptions.


