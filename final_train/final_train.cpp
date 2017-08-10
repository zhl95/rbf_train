// #include "opencl_pre.h";
// #include "initialize.h";

// make p = 1, single run.

//when parallelism (p) is larger than 8, it gives the out of resources error. 
// and the multi queue is not as fast as expected

#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "distance.cl"

#define ARRAY_SIZE 4096

#define MANHATTAN_DIS_KERNEL "distance_sq_ker"
#define MIN_DISTANCE_KERNEL "min_distance_reduction"
#define UPFATE_WEIGHT_KERNEL "update_weight"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// #include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;


	


	// claim parameters
	cl_int row, col, weight_ind;
	cl_int const train_sample_num = 900;
	cl_int const total_sample_num = 900;
	cl_int const neuron_num = 80;
	cl_int interations = 2;


	cl_int const max_weight_size = 250;
	float inputs[total_sample_num][max_weight_size] = {0};


	/* OpenCL structures */
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel vector_kernel, complete_kernel, distance_sq_kernel;
	cl_kernel min_distance_kernel, update_weight_kernel;
	cl_command_queue queue;
	cl_event start_event, end_event;
	cl_int i, err;
	size_t local_size, global_size, net_size_cl;


	/* data_test and buffers */
	float data_test[ARRAY_SIZE];
	// float sum, actual_sum;
	// cl_mem data_buffer, sum_buffer;
	cl_mem weight_buffer, input_buffer;
	cl_mem distance_buffer, diff_buffer;
	cl_mem winner_dis_buffer;
	cl_mem winner_ind_buffer;
	cl_mem neighbourhood_value_buf;
	cl_mem local_weight_size_buf;
	cl_ulong time_start, time_end, total_time;


	// int *winner_ind = (int*)malloc(sizeof(int)); 


/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}


	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}



// find winner function
void distance_sq_func(int input_index, cl_int my_neuron_num, cl_int my_max_weight_size, cl_int my_local_weight_size,size_t my_size_tmp,cl_mem my_dis_sq_buf, cl_mem my_diff_buf, cl_kernel my_dis_kernel, cl_command_queue my_queue){

	// Assign the index representing the start of the current input vector to the distance kernel
	err = clSetKernelArg(my_dis_kernel, 5, sizeof(int), &input_index);
	// if (err < 0) {
	// 	perror("Couldn't create a kernel argument for distance_sq_kernel");
	// 	exit(1);
	// }

	// Enqueue kernel with as many work items as there are neurons in the map
	err = clEnqueueNDRangeKernel(my_queue, my_dis_kernel, 1, NULL, &my_size_tmp, NULL, 0, NULL, NULL);
	// if (err < 0) {
	// 	perror("Couldn't enqueue the distance_sq_kernel");
	// 	cout<< err;
	// 	exit(1);
	// }

	//enqueue is non-blocking, so we need wait using clfinish to make sure this queue is finished before proceeding.
	clFinish(my_queue);

// 	/* Read the result */

	float *distance_sq = (float*)malloc(sizeof(float)*my_neuron_num);
	err = clEnqueueReadBuffer(my_queue, my_dis_sq_buf, CL_TRUE, 0,
		my_neuron_num * sizeof(float), distance_sq, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read the distance_buffer");
		cout << err << ":err:\n";
		exit(1);
	}

	for(int i = 0; i<my_neuron_num; i++){
		cout << "distance_sq[" << i << "] = " << distance_sq[i] << "\n";
	}

	

}


/*
	Function that changes the weights of the map according to their position relative to the winning point and the input vector.
*/


int main() {

	time_t t_start, t_end;

	// remove("updated_weights.txt");
	// remove("radius.txt");
	// remove("inputs.txt");
    

		


	// read input file
	ifstream in("ECGdata.csv");
	string line, field;
	vector< vector<string> > array;  // the 2D array
	vector<string> v;                // array of values for one line only

	while (getline(in, line))    // get next line in file
	{
		v.clear();
		stringstream ss(line);

		while (getline(ss, field, ','))  // break line into comma delimitted fields
		{
			v.push_back(field);  // add each field to the 1D array
			// cout << field << "\n";
		}

		array.push_back(v);  // add the 1D array to the 2D array
	}

	// print out what was read in


// string str;
// float f=atof(str.c_str());



	// for (int p = 0; p< net_size_num; p++){
		for (size_t i = 0; i<total_sample_num; i++)
		{
			for (size_t k = 0; k<max_weight_size; k++){
				// cout << array[i][top_feature_inds[k]] << "|"; // (separate fields by |)
				// inputs[i][k] = atof(array[i][top_feature_inds[k]].c_str());
				inputs[i][k] = atof(array[i][k].c_str());

				// cout << "i k:" << i << ", " << k << "\n";
			}
			
			// cout << "\n";
		}
	// }

	/* Create device and determine local size */
	device = create_device();
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(local_size), &local_size, NULL);
	if (err < 0) {
		perror("Couldn't obtain device information");
		exit(1);
	}

	cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << CL_DEVICE_MAX_WORK_GROUP_SIZE << "\n";
	cout << "local size is :" <<local_size << "\n";

	/* Create a context */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	/* Build program */
	program = build_program(context, device, PROGRAM_FILE);

	/* Create data_test buffer */
	// float inputs_1[2] = {1.01, 0.99};
	float inputs_1[max_weight_size] = {0};

	// for (int i = 0; i < weight_size[0]; i++){
	// 	cout << "input " << inputs_1[i] << "\n";
	// }

	time(&t_start) ;

	int neighbourhood_threshold = 2;
	int winner_ind = {0};
	float learning_rate = 0.07;

	// float weights_1[32] = {0.100, 	0.200, 	0.300, 	0.400, 	0.50,  0.60,  0.70,  0.80,  0.90,  1,	1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2,	2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3,	3.1,  3.2};
	float weights_1[max_weight_size*neuron_num] = {0};
	// float weights_1[weight_size_tmp*net_size_tmp*net_size_tmp];

		// cout << "ppppp1: "  << "\n";


	ifstream in_local("local_size.csv",ios::in); 
	string line_local, field_local;
	vector< vector<string> > array_local;  // the 2D array
	vector<string> v_local;                // array of values for one line only

	while (getline(in_local, line_local))    // get next line in file
	{
		v_local.clear();
		stringstream ss(line_local);

		while (getline(ss, field_local, ','))  // break line into comma delimitted fields
		{
			v_local.push_back(field_local);  // add each field to the 1D array
			// cout << field << "\n";
		}
		array_local.push_back(v_local);  // add the 1D array to the 2D array
	}
	int local_weight_size_array[neuron_num] = {0};
	for (size_t i = 0; i<neuron_num; i++)
	{
		local_weight_size_array[i] = atof(array_local[i][0].c_str());
		// cout << "\n";
	}




	ifstream in_centors("updated_weights.csv",ios::in);
	string line_cen, field_cen;
	vector< vector<string> > array_cen;  // the 2D array
	vector<string> v_cen;   
	while (getline(in_centors, line_cen))    // get next line in file
	{
		v_cen.clear();
		stringstream ss(line_cen);

		while (getline(ss, field_cen, ','))  // break line into comma delimitted fields
		{
			v_cen.push_back(field_cen);  // add each field to the 1D array
			// cout << field << "\n";
		}

		array_cen.push_back(v_cen);  // add the 1D array to the 2D array
	}

	float centors_array [neuron_num][max_weight_size] = {0};
	for (size_t i = 0; i<neuron_num; i++)
	{
		for (size_t k = 0; k<local_weight_size_array[i]; k++){
			centors_array[i][k] = atof(array_cen[i][k].c_str());
			// cout << "i k:" << i << ", " << k << "\n";
		}
		// cout << "\n";
	}



	ifstream in_radius("radius.csv",ios::in); 
	string line_rad, field_rad;
	vector< vector<string> > array_rad;  // the 2D array
	vector<string> v_rad;                // array of values for one line only

	while (getline(in_radius, line_rad))    // get next line in file
	{
		v_rad.clear();
		stringstream ss(line_rad);

		while (getline(ss, field_rad, ','))  // break line into comma delimitted fields
		{
			v_rad.push_back(field_rad);  // add each field to the 1D array
			// cout << field << "\n";
		}

		array_rad.push_back(v_rad);  // add the 1D array to the 2D array
	}
	float radius_array [neuron_num] = {0};
	for (size_t i = 0; i<neuron_num; i++)
	{
		
		radius_array[i] = atof(array_rad[i][0].c_str());

	}

	


	// weight_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 16*2 * sizeof(float), weights_1, &err);
	weight_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, max_weight_size*neuron_num * sizeof(float), centors_array, &err);
	if (err < 0) {
		cout << " weight_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}


	local_weight_size_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  neuron_num*sizeof(int), local_weight_size_array, &err);
	if (err < 0) {
		cout << ": p weight_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}

	distance_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, neuron_num*sizeof(float), NULL, &err);
	if (err < 0) {
		cout << "distance_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}

	diff_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_weight_size*neuron_num*sizeof(float), NULL, &err);
	if (err < 0) {
		cout << "diff_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}



	/* Create a command queue */
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	/* Create kernels */


	distance_sq_kernel = clCreateKernel(program, MANHATTAN_DIS_KERNEL, &err);
	if (err < 0) {
		perror("Couldn't create a distance_sq_kernel");
		exit(1);
	};


	/* Set arguments for distance_sq_kernel*/

	err |= clSetKernelArg(distance_sq_kernel, 1, sizeof(cl_mem), &weight_buffer); // need to be in the loop
	err |= clSetKernelArg(distance_sq_kernel, 2, sizeof(cl_mem), &distance_buffer);
	err |= clSetKernelArg(distance_sq_kernel, 3, sizeof(cl_mem), &diff_buffer);
	err |= clSetKernelArg(distance_sq_kernel, 4, sizeof(cl_mem), &local_weight_size_buf);
	err |= clSetKernelArg(distance_sq_kernel, 5, sizeof(int), &max_weight_size);

	if (err < 0) {
		perror("Couldn't create a kernel argument for distance_sq_kernel");
		exit(1);
	}

	int start_index_1 = 0;

	for (int inte = 0; inte < interations; inte++){
		for (int i = 0; i < train_sample_num; i++){
			
			memcpy(inputs_1,inputs[i],sizeof(inputs_1)); //
			// cout << "inputs: " << inputs_1[0] << ", " << inputs_1[1] << "\n";


			
			input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, max_weight_size * sizeof(float), inputs_1, &err);
			if (err < 0) {
				cout << "input_buffer = clCreateBuffer error" <<err << "\n";
				exit(1);
			}
				
			err = clSetKernelArg(distance_sq_kernel, 0, sizeof(cl_mem), &input_buffer);
			if (err < 0) {
				cout << "set  distance_sq_kernel input_buffer error" <<err << "\n";
				exit(1);
			}

			distance_sq_func(start_index_1, neuron_num, weight_size, size_parallel, distance_buffer, diff_buffer, distance_sq_kernel, queue);

				// cout << "pppppp: " << p << "\n";

			// cout << "winner_ind " << winner_ind << "\n";

		}
	}

			// out.close();
			// out_inputs.close();
			// out_radius.close();
	

	time(&t_end) ;
    printf("time: %.3f s\n",  difftime(t_end,t_start)) ;


    return 0;
}
