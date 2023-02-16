# SMART FILTER
Haar Cascade Classifiers : For this Smart filter Haar Cascade Classifiers was implemented. This is basically a machine learning based approach where a cascade function is trained from a lot of images both positive and negative. Based on the training it is then used to detect the objects in the other images.

## Installation

### Dependencies
- [OpenCV](http://opencv.org/) (Computer Vision Library),
- [ThreadPoolExecutor ](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/scheduling/concurrent/ThreadPoolTaskExecutor.html)(Multiprocessing)

## Getting started
* Create a **`Video`** path where the Video's will be stored.
* Run main.py with following arguments

  
```bash
 python main.py -v 'Path_to_video' -o 'Path_to_output'

```
* Output frames and data will be saved to Output Path. Frames containing people and CSV file with number of people with their Co-ordinates.    
