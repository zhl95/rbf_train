
__kernel void min_distance_reduction (__global float* data, 
      __local float* partial_min, __global float* output, __local int* local_winner_index, __global int* winner_index) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_min[lid] = data[get_global_id(0)];
   local_winner_index[lid] = get_global_id(0);
   barrier(CLK_LOCAL_MEM_FENCE);
   #pragma unroll
   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
        if (partial_min[lid] < partial_min[lid + i]){
          partial_min[lid] = partial_min[lid];
          local_winner_index[lid] = local_winner_index[lid];
        } else {
          partial_min[lid] = partial_min[lid + i];
          local_winner_index[lid] = local_winner_index[lid + i];
        }
        // partial_min[lid]  =  (partial_min[lid] < partial_min[lid + i])?partial_min[lid]:partial_min[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      output[get_group_id(0)] = partial_min[0];
      winner_index[get_group_id(0)] = local_winner_index[0];
   }
}


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
   float sum = 0;
   int current_map_position;
   #pragma unroll
   for (int component = 0; component < local_weight_size[tid]; component++)    //component is the element of a vector
   {
      // sum += input[input_start_index + component] - map[base_map_position+component];
      // sum_a += fabs(sum);
    current_map_position = base_map_position + component;
    sum_a += pow(fabs(input[input_start_index + component] - map[current_map_position]),2);
    barrier(CLK_LOCAL_MEM_FENCE);
    

    diff_map[current_map_position] = input[input_start_index + component] - map[current_map_position];

   }

   distance_sq_map[tid] = sum_a;
}




int manhattan_neighbourhood(
  int winner_index,
  int map_side_size,
  int current_id)
{
  int a_x, a_y, b_x, b_y;
  a_x = current_id % map_side_size;
  a_y = current_id / map_side_size;
  b_x = winner_index % map_side_size;
  b_y = winner_index / map_side_size;

  return abs(a_x-b_x) + abs(a_y-b_y);
}



__kernel void update_weight(
  __global float *map,
  // __global float *input,
  __global float *diff_map,
  int neighbourhood_threshold,
  float learning_rate,
  int winner_index,
  int input_start_index,
  int vector_length,
  int map_side_size,
  __global int *neighbourhood_value
  )
{
  int current_id = get_global_id(0);
  neighbourhood_value[current_id] = manhattan_neighbourhood(winner_index, map_side_size, current_id);
  int neighbourhood_rate = (neighbourhood_value[current_id]<neighbourhood_threshold)?1:0;

  int current_map_position;
  #pragma unroll
  for (current_map_position = current_id*vector_length; current_map_position < current_id*vector_length + vector_length; current_map_position++){
    map[current_map_position] = map[current_map_position] + 
          (diff_map[current_map_position] * learning_rate * neighbourhood_rate);

  }
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



