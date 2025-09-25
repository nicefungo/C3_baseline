
echo "run \`sudo jetson_clocks\` first" &&
nvcc -arch=sm_87 -o C3 C3_kernel_mem_opt.cu ./cnpy/cnpy.cpp -lz && ./C3 $1 $2 >> output.txt

