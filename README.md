# SLAM
Video Navigation

Submitted By: Dor Messica, Ron Kobrowski

Draft made with the help of copilot:
## Introduction
In this project we will implement a SLAM algorithm.
SLAM stands for Simultaneous Localization and Mapping.
This means that we will create a map of the environment, and at the same time we will localize ourselves in the map.

<img src=VAN_ex/media/path_start_gif.gif width="500" height="" alt="">

## show stuff from media directory
<img src=VAN_ex/media/trajectory.png width="500" height="" alt="initial trajectory">

## part 1 - ex1 folder
connecting the stereo by detecting features in the images and matching them.

## part 2 - ex2 folder
in this part we will create a map of the environment.
this means that we will create a 3d scatter plot of the points in the environment.

## part 3 - ex3 folder
now, once we have the map, we can use it to make the localization part of the SLAM.
we will use the PNP algorithm, with added RANSAC and refinement.
