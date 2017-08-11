
// Each of these kernels deals with one neuron on the map, the for loop
// iterates over the components of the neuron's vector and the input
// vector
__kernel void distance_sq_ker(
   __global float * input,
   __global float * map,
   __global float * distance_sq_map,
   __global float * diff_map,
   __global int * local_weight_size,
   int max_vector_length,  //vector size is the weight size.
   int input_start_index)
{
   size_t tid = get_global_id(0);
   int base_map_position = tid*max_vector_length;   //tid here represents the neuron ID. all neuron process in parallel.
      
   float sum_a = 0;
   int current_map_position;
   #pragma unroll
   for (int component = 0; component < local_weight_size[tid]; component++)    //component is the element of a vector
   {
    current_map_position = base_map_position + component;
    sum_a += pow((input[input_start_index + component] - map[current_map_position]),2);
    barrier(CLK_LOCAL_MEM_FENCE);
    

    diff_map[current_map_position] = input[input_start_index + component] - map[current_map_position];

   }

   distance_sq_map[tid] = sum_a;
}


__kernel void gaussian_ker(
   __global float * distance_sq,
   __global float * radius,
   __global float * gaussian)
{
   size_t tid = get_global_id(0);
   gaussian[tid] = exp(-distance_sq[tid]/(2*pow(radius[tid],2)));
}


// __kernel void predict_error_ker (
//       __global float* actual, 
//       __global float* gaussian,
//       __global float* weight_gau,
//       int neuron_num,
//       __global float error
      
// ) {
//   size_t tid = get_global_id(0);


// }




__kernel void update_weight(
  __global float *weight_gau,
  __global float *gaussian,
  // int neuron_num,
  float error,
  float learning_rate
  )
{
  int tid = get_global_id(0);
  weight_gau[tid] = weight_gau[tid] + learning_rate*error*gaussian[tid];

}


// __kernel void update_weight(
//   __global float *map,
//   __global float *input,
//   __global float *diff_map,
//   int neighbourhood_threshold,
//   float learning_rate,
//   int winner_index,
//   int input_start_index,
//   int vector_length,
//   int map_side_size,
//   __global int *neighbourhood_value
//   )
// {
//   int current_id = get_global_id(0);
//   neighbourhood_value[current_id] = manhattan_neighbourhood(winner_index, map_side_size, current_id);
//   int neighbourhood_rate = (neighbourhood_value[current_id]<neighbourhood_threshold)?1:0;

//   int current_map_position;
//   for (current_map_position = current_id*vector_length; current_map_position < current_id*vector_length + vector_length; current_map_position++){
//     map[current_map_position] = map[current_map_position] + 
//           (diff_map[current_map_position] * learning_rate * neighbourhood_rate);

//   }
// }



