#!/bin/bash


echo "Writing to distination: $2";
echo "with debugging = $4";
echo "..../..../...../"
python3 main.py $1 $2 $3 $4 $5

if [ $# -lt 5 ]
  then
    echo "USAGE: TYPE<vid/img> <Input_path/*.jpg> <output_path/*.jpg> mode <1/0>"
fi