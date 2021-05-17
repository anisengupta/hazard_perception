# hazard_percpetion
A method of analysing and identifying hazards in an automated fashion for the Hazard Perception Test that is required to be undertaken by all learner drivers when undertaking the UK Theory Test.

A vital part of the UK Theory Test is the Hazard Perception which is a seris of clips that features everyday road scenes and contains a developing hazard. Candidates are required to identify the hazard in the clip by clicking on it at the appropriate time. Points are awarded for spotting the hazards as soon as they start to occur. These are hazards that would cause you to take action to either change your speed or direction.

Candidates can score up to five points for each hazard and this depends on how early the spot it. There is a time limit for each hazard and within this frame, if a candidate identifies a hazard early then they rewarded highly for that clip. If however, they react late, they can gain only one or two points (or none at all if they miss the hazard completely). More information on this can be found here: https://www.safedrivingforlife.info/free-practice-tests/hazard-perception-test/

It is also important to note that if a candidate clicks too often on a clip, it will count as a fail for the clip and the candidate will receive no points. Clicking continously is deemed to be cheating and will not be tolerated during the real exam. This test is inherently flawed since they contain plenty of hazards (not including the developing one) which would cause a driver to at least consider being cautious and drive carefully. The hazard perception test sometimes requires inhuman levels of reaction speeds as on some clips, hazards can develop rapidly and unless the candidate is familiar with the clip, it is impossible to score highly on it.

We have therefore devised a method of automatically identifying hazards on a clip using machine and deep learning which will enable us to feed in a video clip and let the algorithm decide the hazard for us. We will use the ImageAI libary which contains algorithms for image prediction, object detection, video object tracking and many more (https://github.com/OlafenwaMoses/ImageAI).

Using this libary, we have produced a few lines of code which analyses the hazard perception video clip and outputs an analysed clip with the hazards identified along with a percentage of certainty, please see the example below:

![image](https://user-images.githubusercontent.com/62817385/118469160-5db42400-b6fd-11eb-9a26-2160f7c35b7b.png)
