#!/bin/bash
ffmpeg -f image2 -r 2 -i door_wood_knock_relation_grid_%d.png video.avi
ffmpeg -i video.avi -pix_fmt rgb24 -loop_output 0 out.gif