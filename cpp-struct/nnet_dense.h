#ifndef NNET_DENSE_H
#define NNET_DENSE_H

#include "nnet_common.h"
#include <math.h>
#include <iostream> 
#include <cmath> 
#include <vector> 
#include <stdlib.h> 
#include <time.h> 

using namespace std; 
// #define innode 2        
//输入结点数 
// #define hidenode 4      
//隐含结点数 
// #define hidelayer 1     
//隐含层数 
// #define outnode 1       
//输出结点数 
// #define learningRate 0.9
//学习速率，alpha 

namespace nnet{

// only used to pase into the type of the layer
struct dense_type_config
{
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    typedef float input_t;
    typedef float result_t;

    static const unsigned n_in;
    static const unsigned n_out;
   // here just set the default input node number and output node number
};







// weights should be initailized, and managed directly by the main.cc
template<typename CONFIG_T>
void compute_layer(
    // datainput_T datainput[CONFIG_T::datainput_t],
    // result_T    result[CONFIG_T::result_t],
    typename CONFIG_T::input_t datainput[CONFIG_T::input_t],
    typename CONFIG_T::result_t    result[CONFIG_T::result_t],

    // the input data type and output data type are defined by the struct
    
    // also the weights and bias type are defined by the struct
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in][CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]
   )
{
    typename CONFIG_T::input_t cache;

    typename CONFIG_T::accum_t mult[CONFIG_T::n_in][CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    Product1: for(int ii = 0; ii<CONFIG_T::n_in; ii++)
    {
        cache = datainput[ii];
        Product2: for(int jj=0; jj<CONFIG_T::n_out; jj++)
        {
            mult[ii][jj] = cache*weights[ii][jj];
        }
    }

    RESETACCUM:for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++)
    {
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    ACCUM1:for(int ii=0; ii < CONFIG_T::n_in; ii++)
    {
        ACCUM2:for(int jj =0; jj <CONFIG_T::n_out; jj++)
        {   
            acc[jj] +=mult[ii][jj]; 
        }
    }

    RESULT:for(int ires = 0; ires < CONFIG_T::n_out; ires++)
    {
        result[ires] = (typename CONFIG_T::result_t) acc[ires];
    }

}// end of the compute_layer, or you call cpu_forward()





/************************************ */
/***       BackPropagation          ***/
/************************************ */

















}// end of the nnet namespace defination

#endif