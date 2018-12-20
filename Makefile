NVCC=nvcc
SRC_PATH="./src"
BIN_PATH="./bin"

all: 
	mkdir -p ${BIN_PATH}	
	${NVCC} -arch=sm_35 -O3 -o ${BIN_PATH}/miog_cuda ${SRC_PATH}/main.cu

run:
	${BIN_PATH}/miog_cuda "data/miog.bin" "result.txt"
clean:
	rm -rf ${BIN_PATH}
	rm -f result.txt