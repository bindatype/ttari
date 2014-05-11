################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../ttari.cu 

CU_DEPS += \
./ttari.d 

OBJS += \
./ttari.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


