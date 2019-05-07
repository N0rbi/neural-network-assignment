#!/bin/bash

docker build --tag local:cnn_grid  . 
docker run local:cnn_grid cnn.py -v ${PWD}/models:/saved_models
