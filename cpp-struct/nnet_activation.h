#pragma once 
#include <iostream> 
#include <cmath> 
#include <vector> 
#include <stdlib.h> 
#include "layer.h"
#include "math.h"
#include <time.h> 

using namespace std; 



namespace nnet{

// only used to pase into the type of the layer
struct tanh_type_config
{

    typedef float input_t;
    typedef float result_t;

    static const unsigned n_in;    
   // here just set the default input node number and output node number
};







// weights should be initailized, and managed directly by the main.cc
template<typename CONFIG_T>
void active_tanh(
    // datainput_T datainput[CONFIG_T::datainput_t],
    // result_T    result[CONFIG_T::result_t],
    typename CONFIG_T::input_t datainput[CONFIG_T::input_t],
    typename CONFIG_T::result_t    result[CONFIG_T::result_t]

    // the input data type and output data type are defined by the struct   
   )
{
    RESULT:for(int ires = 0; ires < CONFIG_T::n_in; ires++)
    {
        result[ires] = (typename CONFIG_T::result_t) tanh(datainput[ires]);
    }

}// end of the tanh(x), or you call cpu_forward()





/************************************ */
/***       BackPropagation          ***/
/************************************ */















/************************************ */
/***             SOFTMAX          ***/
/************************************ */
// only used to pase into the type of the layer
struct softmax_type_config
{
    typedef float input_t;
    typedef float result_t;

    static const unsigned n_in;
    // THE INPUT AND OUTPUT NUMBER OF SOFTMAX ARE SAME
};



template<typename CONFIG_T>
void softmax_layer(
    typename CONFIG_T::input_t datainput[CONFIG_T::input_t],
    typename CONFIG_T::result_t    result[CONFIG_T::result_t],
    typename CONFIG_T::result_t    *sum
)
{
    typename CONFIG_T::result_t sum_cache;

    // typename CONFIG_T::result_t mult[CONFIG_T::n_in]
    // typename CONFIG_T::result_t acc[CONFIG_T::n_in];

    Product1: for(int ii = 0; ii<CONFIG_T::n_in; ii++)
    {
        sum_cache += (typename CONFIG_T::result_t)  exp(datainput[ii]);

    }

    DIVISION1: for(int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
        result[ii] = (typename CONFIG_T::result_t) exp(datainput[ii] / sum_cache) ;
    }

    *sum = sum_cache;
    // store the result to the struct

}// end of the compute_layer, or you call cpu_forward()





/************************************ */
/***       BackPropagation          ***/
/************************************ */



















}// end of the nnet namespace defination