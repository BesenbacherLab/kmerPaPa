#!/bin/zsh

#cat test_data/joint_5mers.txt | awk '{print $1,int((rand()+0.5)*$2),int((rand()+0.5)*$3),int((rand()+0.5)*$2),int((rand()+0.5)*$3)}' > test_data/pairwise_joint_5mers.txt

#cat test_data/joint_5mers.txt | awk '{print $1,int((rand()+0.5)*$2),int((rand()+0.5)*$3),int((rand()+0.5)*$2),int((rand()+0.5)*$3),int((rand()+0.5)*$2),int((rand()+0.5)*$3)}' > test_data/pairwise_3_joint_5mers.txt

cat test_data/joint_5mers.txt | awk -vORS="" '{print $1; for(i=0; i<30; i++) print " "int((rand()+0.5)*$2),int((rand()+0.5)*$3);print "\n"}' > test_data_pairwise_30_joint_5mers.txt

