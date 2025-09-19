
echo "run \`sudo jetson_clocks\` first" &&
nvcc -arch=sm_87 -o C3_baseline -w C3_baseline.cu ./cnpy/cnpy.cpp -lz && ./C3_baseline 1 11110

