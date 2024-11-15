#!/bin/bash
curl -L -o ./data/archive.zip \
https://www.kaggle.com/api/v1/datasets/download/andrewmvd/brazilian-stock-market
unzip ./data/archive.zip -d ./data
rm ./data/archive.zip ./data/economic_indicators.csv