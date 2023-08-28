# Eye Blink Detection with Event Camera Data

This repository contains the code for re-implementing the paper "Real Time Face and Eye Tracking and Blink Detection Using Event Cameras" paper on eye blink detection using event camera data. The goal of this project is to develop a system that can accurately detect eye blinks in real-time using event camera technology, which provides asynchronous and high-speed visual information.Eye blink detection plays a crucial role in various applications such as driver drowsiness detection, human-computer interaction, and facial expression analysis. In this project, we focus on leveraging event camera data, which provides a stream of asynchronously generated events, to detect eye blinks. By exploiting the high temporal resolution and low latency of event cameras, we aim to achieve accurate and real-time eye blink detection.


## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## DATASET

Event datasets are usually difficult to come by. We use a synthetically generated dataset called the Neuromorphic Helen dataset. This dataset was generated using Event Simulaators on helen images. These simulators are designed to take in videos as input and output event representation of these. we therefore generated crops of single images of Helen, moved the crop around randomly to simulate a 6-dof camera movement. Follow the steps below to generate the dataset

1. Download [Helen](http://www.ifp.illinois.edu/~vuongle2/helen/)
2. Generate croppings of single images by running  `python3 Image_Cropping.py`
3. Follow the instructions at [V2E](https://github.com/SensorsINI/v2e) or [VID2E](https://github.com/uzh-rpg/rpg_vid2e) to generate event simulations on your cropped images


## Event Representation for CNN Architecture
After generating events from single images, we then need to convert our events into 2D frames as accepted by CNNS, we do this by accumulating polarity pixelwise using the method  introduced in the paper "High Speed and High Dynamic Range Video with an Event Camera"

To generate 2d frames, follow the steps below;

- Follow the instructions in the repo [E2VID](https://github.com/uzh-rpg/rpg_e2vid)

## Generating Bounding Boxes on Frames

dencies by running the following command:
pip install -r requirements.txt

## Training GR-YOLO Model

-Download pretrained weights [Link](URL)

## Results

Include any relevant results or performance metrics obtained from running the code. You can also provide visualizations or examples of the output produced by the algorithm.

## Contributing

Contributions to this project are welcome. If you encounter any issues, have ideas for improvements, or want to contribute enhancements, feel free to submit a pull request or open an issue on the repository.

## License

[MIT License](LICENSE)

##REFERENCES

