# Sequence Alignment Example 
# Univie

On Alma, compile with:
   /opt/global/gcc-11.2.0/bin/g++ -O2 -std=c++20 -fopenmp -o gpsa  main.cpp

To Run: 
   ./gpsa 

To run all configurations: 
   sh runall ./gpsa 

Optionally you can provide other program arguments,

To run with a specific data set use: ./gpsa --x <sequence1_filename> --y <sequence2_filename>

By default, your program will look for X.txt and Y.txt. 

Here are the available sequences: 

1. X.txt, Y.txt, size: [51480x53640] 
Random, big sequences.

2. X2.txt, Y2.txt, size: [32768x32768] 
Same, but this has the same size dimensions that nicely divide.

3. simple1.txt, simple2.txt, size: [3x5] 
Small sequences that you can use for debugging. 
You can change this file as you please.

3. simple-longer-1.txt, simple-longer-2.txt, size: [20x20] 
Small sequences that you can use for debugging. 
You can change this file as you please.