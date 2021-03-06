# Computer. See. Buildings.
### Semantic segmentation for building footprint detection with applications in urban development


## Background

A rich scientific literature has identified many advantages to urban density but in practice North American cities tend to be sprawling. One option for promoting urban densification is to loosen zoing regulations to allow property owners to build multiple residences on their properties, such as in-law suites, [laneway houses](https://www.toronto.ca/city-government/planning-development/planning-studies-initiatives/changing-lanes-the-city-of-torontos-review-of-laneway-suites/), or tiny homes. However, even with looser regulations infill development will not occur if it is not economically viable.


## Research Question

Can I use deep learning to create a tool that can empower small-scale developers by facilitating the development of small homes?


## Data and Methods

In this project I use semantic segmentation to attempt to identify building footprints which can be used for urban density calculations. I employ the convolutional neural network called [U-Net](https://en.wikipedia.org/wiki/U-Net), training it on labelled satellite images from SpaceNEt. The data are from their [SpaceNet 2: Building Detection v2 dataset](https://spacenet.ai/spacenet-buildings-dataset-v2/), and I only use images from Las Vegas as I had limited computational resources and needed to choose training images from North American, which would resemble the urban forms we see in Canada.


## Results

The model performs well on training and validation data but struggles with unseen data. Its ability to produce accurate predictions depends on qualities of the satellite image, such as its colour map, brightness, and level of zoom. This is likely due to the homogeneity of training images. Results could be improved by included images from diverse locations, but also through data augmentation to vary the zoom, clarity, brightness, and other qualities of the training images.

In addition, even if performance were improved for building footprint detection, the model would need to be expanded in order to be useful for the stated purpose because, presently, it cannot distinguish between roads/parking lots and actual useable space. However, given these caveats this stands as a useful proof-of-concept, and serves as a solid foundation for further exploration.

Training Image:

![image](https://user-images.githubusercontent.com/91850081/161823227-807c23ac-aec4-418c-af02-be5860715125.png)

Prediction on Training Image:

![image](https://user-images.githubusercontent.com/91850081/161823276-cf8f3550-d3fd-4dad-8ffd-4571b235ab56.png)

Prediction on Unseen Image (screenshot of Toronto neighbourhood from Google Earth)

![image](https://user-images.githubusercontent.com/91850081/161821190-e04b5cb2-ee80-49a7-bb96-6c98e4971b30.png)
