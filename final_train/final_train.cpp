// #include "opencl_pre.h";
// #include "initialize.h";

// make p = 1, single run.

//when parallelism (p) is larger than 8, it gives the out of resources error. 
// and the multi queue is not as fast as expected

#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "distance.cl"

#define ARRAY_SIZE 4096

#define MANHATTAN_DIS_KERNEL "distance_sq_ker"
#define GAUSSIAN_KERNEL "gaussian_ker"
#define UPDATE_KERNEL "update_weight"



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
	cl_int const train_sample_num = 800;
	cl_int const total_sample_num = 800;
	cl_int const neuron_num = 68;
	cl_int const size_parallel = neuron_num;
	cl_int interations = 2;


	cl_int const max_weight_size = 250;
	float inputs[total_sample_num][max_weight_size] = {0};


	/* OpenCL structures */
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel vector_kernel, complete_kernel, distance_sq_kernel;
	cl_kernel gaussian_kernel, update_kernel;

	cl_command_queue queue;
	cl_event start_event, end_event;
	cl_int i, err;
	size_t local_size, global_size, net_size_cl;


	/* data_test and buffers */
	float data_test[ARRAY_SIZE];
	// float sum, actual_sum;
	// cl_mem data_buffer, sum_buffer;
	cl_mem centor_buffer, input_buffer;
	cl_mem distance_sq_buffer, diff_buffer;
	cl_mem winner_dis_buffer;
	cl_mem winner_ind_buffer;
	cl_mem neighbourhood_value_buf;
	cl_mem local_weight_size_buf;
	cl_mem radius_buf, gaussian_buf, weight_gau_buf;
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
float * distance_sq_func(int input_index, cl_int my_neuron_num, size_t my_size_tmp,cl_mem my_dis_sq_buf, cl_mem my_diff_buf, cl_kernel my_dis_kernel, cl_command_queue my_queue){

	// Assign the index representing the start of the current input vector to the distance kernel
	err = clSetKernelArg(my_dis_kernel, 6, sizeof(int), &input_index);
	if (err < 0) {
		perror("Couldn't create a kernel argument for distance_sq_kernel");
		exit(1);
	}

	// Enqueue kernel with as many work items as there are neurons in the map
	err = clEnqueueNDRangeKernel(my_queue, my_dis_kernel, 1, NULL, &my_size_tmp, NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the distance_sq_kernel");
		cout<< err;
		exit(1);
	}

	//enqueue is non-blocking, so we need wait using clfinish to make sure this queue is finished before proceeding.
	clFinish(my_queue);

// 	/* Read the result */

	float *distance_sq = (float*)malloc(sizeof(float)*my_neuron_num);
	err = clEnqueueReadBuffer(my_queue, my_dis_sq_buf, CL_TRUE, 0,
		my_neuron_num * sizeof(float), distance_sq, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read the distance_sq_buffer");
		cout << err << ":err:\n";
		exit(1);
	}

	for(int i = 0; i<my_neuron_num; i++){
		cout << "distance_sq[" << i << "] = " << distance_sq[i] << "\n";
	}

	return distance_sq;


}


float * gaussian_func(cl_mem my_gaussian_buf, cl_mem my_dis_sq_buf, cl_mem my_radius_buf, size_t my_size_tmp, int my_neuron_num, cl_kernel my_gaussian_kernel, cl_command_queue my_queue){

	// Assign the index representing the start of the current input vector to the distance kernel
	err = clSetKernelArg(my_gaussian_kernel, 0, sizeof(cl_mem), &my_dis_sq_buf);
	if (err < 0) {
		perror("Couldn't create a kernel argument 0 for my_gaussian_kernel");
		exit(1);
	}

	// err = clSetKernelArg(my_gaussian_kernel, 1, sizeof(cl_mem), &my_radius_buf);
	// if (err < 0) {
	// 	perror("Couldn't create a kernel argument 1 for my_gaussian_kernel");
	// 	exit(1);
	// }

	err = clSetKernelArg(my_gaussian_kernel, 2, sizeof(cl_mem), &my_gaussian_buf);
	if (err < 0) {
		perror("Couldn't create a kernel argument 2 for my_gaussian_kernel");
		exit(1);
	}




	// Enqueue kernel with as many work items as there are neurons in the map
	err = clEnqueueNDRangeKernel(my_queue, my_gaussian_kernel, 1, NULL, &my_size_tmp, NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the distance_sq_kernel");
		cout<< err;
		exit(1);
	}

	//enqueue is non-blocking, so we need wait using clfinish to make sure this queue is finished before proceeding.
	clFinish(my_queue);

// 	/* Read the result */

	float *gaussian_read = (float*)malloc(sizeof(float)*my_neuron_num);
	err = clEnqueueReadBuffer(my_queue, my_gaussian_buf, CL_TRUE, 0,
		my_neuron_num * sizeof(float), gaussian_read, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read the my_gaussian_buf");
		cout << err << ":err:\n";
		exit(1);
	}

	for(int i = 0; i<my_neuron_num; i++){
		cout << "gaussian_read[" << i << "] = " << gaussian_read[i] << "\n";
	}

	return gaussian_read;


}



void update_func(cl_mem my_weight_gau_buf, cl_mem my_gaussian_buf, float my_error, float my_neuron_num, size_t my_size_tmp, cl_kernel my_update_kernel, cl_command_queue my_queue){

	// Assign the index representing the start of the current input vector to the distance kernel
	err = clSetKernelArg(my_update_kernel, 0, sizeof(cl_mem), &my_weight_gau_buf);
	if (err < 0) {
		perror("Couldn't create a kernel argument 0 for my_update_kernel");
		exit(1);
	}

	err = clSetKernelArg(my_update_kernel, 1, sizeof(cl_mem), &my_gaussian_buf);
	if (err < 0) {
		perror("Couldn't create a kernel argument 1 for my_update_kernel");
		exit(1);
	}





	// Enqueue kernel with as many work items as there are neurons in the map
	err = clEnqueueNDRangeKernel(my_queue, my_update_kernel, 1, NULL, &my_size_tmp, NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the my_update_kernel");
		cout<< err;
		exit(1);
	}

	//enqueue is non-blocking, so we need wait using clfinish to make sure this queue is finished before proceeding.
	clFinish(my_queue);

// 	/* Read the result */

	float *weight_gau_read = (float*)malloc(sizeof(float)*my_neuron_num);
	err = clEnqueueReadBuffer(my_queue, my_weight_gau_buf, CL_TRUE, 0,
		my_neuron_num * sizeof(float), weight_gau_read, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read the my_gaussian_buf");
		cout << err << ":err:\n";
		exit(1);
	}

	for(int i = 0; i<my_neuron_num; i++){
		cout << "weight_gau_read[" << i << "] = " << weight_gau_read[i] << "\n";
	}

	// return weight_gau_read;
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

	float learning_rate = 0.008;

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
	float weight_gau[neuron_num] = {0};
	for (int i = 0; i<neuron_num; i++){
		weight_gau[i] =  (rand() / double(RAND_MAX))*1.9*0.1 + 0.1;   //1.9 is the range of actual value;
	}	

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
			cout << field_local << "\n";
		}
		array_local.push_back(v_local);  // add the 1D array to the 2D array
	}
	int local_weight_size_array[neuron_num] = {0};
	for (size_t i = 0; i<neuron_num; i++)
	{
		local_weight_size_array[i] = atof(array_local[i][0].c_str());
		// cout << "\n";
	}




	ifstream in_centors("centorss.csv",ios::in);
	// ifstream in_centors("updated_weights.csv",ios::in);

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



	ifstream in_radius("radiuss.csv",ios::in); 
	// ifstream in_radius("radius.csv",ios::in); 

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
		
		radius_array[i] = atof(array_rad[i][0].c_str())/6;

	}


	ifstream in_act("actual.csv",ios::in); 
	string line_act, field_act;
	vector< vector<string> > array_act;  // the 2D array
	vector<string> v_act;                // array of values for one line only

	while (getline(in_act, line_act))    // get next line in file
	{
		v_act.clear();
		stringstream ss(line_act);

		while (getline(ss, field_act, ','))  // break line into comma delimitted fields
		{
			v_act.push_back(field_act);  // add each field to the 1D array
			cout << field_act << "\n";
		}

		array_act.push_back(v_act);  // add the 1D array to the 2D array
	}
	float act_array [total_sample_num] = {0};
	for (size_t i = 0; i<total_sample_num; i++)
	{
		
		act_array[i] = atof(array_act[i][0].c_str());
		cout << "act_array [" << act_array[i] << "\n";

	}




	


	// centor_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 16*2 * sizeof(float), weights_1, &err);
	centor_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, max_weight_size*neuron_num * sizeof(float), centors_array, &err);
	if (err < 0) {
		cout << " centor_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}

	radius_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, neuron_num * sizeof(float), radius_array, &err);
	if (err < 0) {
		cout << " radius_buf = clCreateBuffer error" <<err << "\n";
		exit(1);
	}




	local_weight_size_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  neuron_num*sizeof(int), local_weight_size_array, &err);
	if (err < 0) {
		cout << ": p local_weight_size_buf = clCreateBuffer error" <<err << "\n";
		exit(1);
	}

	distance_sq_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, neuron_num*sizeof(float), NULL, &err);
	if (err < 0) {
		cout << "distance_sq_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}

	diff_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_weight_size*neuron_num*sizeof(float), NULL, &err);
	if (err < 0) {
		cout << "diff_buffer = clCreateBuffer error" <<err << "\n";
		exit(1);
	}

	gaussian_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, neuron_num*sizeof(float), NULL, &err);
	if (err < 0) {
		cout << "gaussian_buf = clCreateBuffer error" <<err << "\n";
		exit(1);
	}


	weight_gau_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, neuron_num * sizeof(float), weight_gau, &err);
	if (err < 0) {
		cout << "weight_gau_buf = clCreateBuffer error" <<err << "\n";
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

	gaussian_kernel = clCreateKernel(program, GAUSSIAN_KERNEL, &err);
	if (err < 0) {
		perror("Couldn't create a gaussian_kernel");
		exit(1);
	};

	update_kernel = clCreateKernel(program, UPDATE_KERNEL, &err);
	if (err < 0) {
		perror("Couldn't create a gaussian_kernel");
		exit(1);
	};



	/* Set arguments for distance_sq_kernel*/

	err |= clSetKernelArg(distance_sq_kernel, 1, sizeof(cl_mem), &centor_buffer); // need to be in the loop
	err |= clSetKernelArg(distance_sq_kernel, 2, sizeof(cl_mem), &distance_sq_buffer);
	err |= clSetKernelArg(distance_sq_kernel, 3, sizeof(cl_mem), &diff_buffer);
	err |= clSetKernelArg(distance_sq_kernel, 4, sizeof(cl_mem), &local_weight_size_buf);
	err |= clSetKernelArg(distance_sq_kernel, 5, sizeof(int), &max_weight_size);
	if (err < 0) {
		perror("Couldn't setup arg for distance_sq_kernel");
		exit(1);
	};

	err = clSetKernelArg(gaussian_kernel, 1, sizeof(cl_mem), &radius_buf);
	if (err < 0) {
		perror("Couldn't setup arg for gaussian_kernel");
		exit(1);
	};


	err = clSetKernelArg(update_kernel, 3, sizeof(float), &learning_rate);
	if (err < 0) {
		perror("Couldn't setup arg 3 for update_kernel");
		exit(1);
	};





	int start_index_1 = 0;
	float * distance_sq_read;
	float * gaussian_read;	
	float predict_err;	
	float predicted = 0;	

		



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

			distance_sq_read = distance_sq_func(start_index_1, neuron_num, size_parallel, distance_sq_buffer, diff_buffer, distance_sq_kernel, queue);
			// cout << "distance_sq_read[0]: " << distance_sq_read[0] << "\n";

			gaussian_read =  gaussian_func(gaussian_buf, distance_sq_buffer, radius_buf, size_parallel, neuron_num, gaussian_kernel, queue);
			// cout << "gaussian_read[0]: " << gaussian_read[0] << "\n";

			// void gaussian_func(cl_mem my_gaussian_buf, cl_mem my_dis_sq_buf, cl_mem my_radius_buf, size_t my_size_tmp, int my_neuron_num, cl_kernel my_gaussian_kernel, cl_command_queue my_queue){
			#pragma unroll
			for (int neuron_id = 0; neuron_id < neuron_num; neuron_id++){
				predicted += gaussian_read[neuron_id] * weight_gau[neuron_id];
			}

			predict_err =  act_array[i] - predicted;

			cout << "actual[] " << i << act_array[i] << "\n";
			cout << "predicted: " << predicted << "\n";
			cout << "predict_err[] " << i << predict_err << "\n";

			//update weight_gau

			err = clSetKernelArg(update_kernel, 2, sizeof(float), &predict_err);
			if (err < 0) {
				perror("Couldn't setup arg 2 for update_kernel");
				exit(1);
			};

			update_func(weight_gau_buf, gaussian_buf, predict_err, neuron_num, size_parallel, update_kernel, queue);
			// float * update_func(cl_mem my_weight_gau_buf, cl_mem my_gaussian_buf, float my_error, float my_neuron_num, size_t my_size_tmp, cl_kernel my_update_kernel, cl_command_queue my_queue){




		}
	}

			// out.close();
			// out_inputs.close();
			// out_radius.close();
	

	time(&t_end) ;
    printf("time: %.3f s\n",  difftime(t_end,t_start)) ;


    return 0;
}
