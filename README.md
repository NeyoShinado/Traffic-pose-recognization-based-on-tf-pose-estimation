# tf-pose-estimation-Traffic-pose-version

'Openpose', human pose estimation algorithm, have been implemented using Tensorflow. It also provides several variants that have some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.** This version for estimating traffic policeman's pose are powered by CMU-Perceptual-Computing-Lab's work: Real-time multi-person pose estimation, and depends on its Tensorflow verion.

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose
Tensorflow version : https://github.com/ildoonet/tf-pose-estimation


## Install

Install the packages & dependencies follow :https://github.com/ildoonet/tf-pose-estimation.


## Download Tensorflow Graph File

Before running demo, you should download the tensorflow Graph file for human keypoints estimation. See [experiments.md](https://github.com/ildoonet/tf-pose-estimation/blob/master/etcs/experiments.md)
Which cmu (trained in 656x368) is recommended.


## File modification

After satisfied the requirements. Move the folder _"code"_ into the project's path _"./openpose_tf"_ and rename it as _"Traffic_pose"_.
And move the [file](code/run_directory.py) to the project's path.
In the end, move the model files[Pose_recg] into the folder models.


## Test Inference

You can test the inference feature with a set of images in the src folder which can be check by the [file](run_directory.py).

## Notation
The testing accuracy is for training or *one-man situation*. A more complex applications scenarios of multi-humans will reduce the score of acc cuz the *pose of pedestrians* are also detected.

## References

See : [etcs/reference.md](https://github.com/ildoonet/tf-pose-estimation/blob/master/etcs/reference.md)
