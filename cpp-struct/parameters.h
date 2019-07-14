#ifndef PARAMETERS_H_
#define PARAMETERS_H_

// defines all the input number, output number , data type
// and the whole nnet structure

#include <complex>

#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_dense.h"




typedef nnet::accum_t_def accum_default_t;
typedef nnet::weight_t_def weight_default_t;
typedef nnet::bias_t_def bias_default_t;
typedef nnet::input_t_def input_default_t;
typedef nnet::result_t_def result_default_t;


#define Y_INPUTS 4
#define N_CHAN 6
#define Y_FILT 2
#define N_FILT 3
#define STRIDE 1
#define PAD_LEFT 0
#define PAD_RIGHT 1
#define Y_OUTPUTS 4
#define N_OUTPUTS 6

/************************************ */
/***         first layer            ***/
/************************************ */

// initialization of the type of the first layer
struct layer1_dense_type: nnet::dense_type_config{

        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;

        typedef input_default_t input_t;
        typedef result_default_t result_t;


        static const unsigned n_in  = 2;
        static const unsigned n_out  = 4;

};
// the first layer struct defination, store the intermidiete value for backpropogation
struct layer1_dense{
        weight_default_t weights[2][4];
        bias_default_t   biases[4];
        accum_default_t  acc[4];

        input_default_t  datainput[2];
        result_default_t result[4];
}layer1_dense;






/************************************ */
/***         tanh layer            ***/
/************************************ */
// initialize of the type of the fisrt tanh layer
struct layer1_tanh_type: nnet::tanh_type_config{
    typedef input_default_t input_t;
    typedef result_default_t result_t;

    static const unsigned n_in = 4;    
   // here just set the default input node number and output node number
};
// the first layer struct defination, store the intermidiete value for backpropogation
struct layer1_tanh{
        input_default_t  datainput[4];
        result_default_t result[4];
}layer1_tanh;




/************************************ */
/***         second layer            ***/
/************************************ */

// initialization of the type of the first layer
struct layer2_dense_type: nnet::dense_type_config{

        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;

        typedef input_default_t input_t;
        typedef result_default_t result_t;


        static const unsigned n_in  = 4;
        static const unsigned n_out  = 2;

};
// the second layer struct defination, store the intermediate value for backpropogation
struct layer2_dense{
        weight_default_t weights[4][2];
        bias_default_t   biases[2];
        accum_default_t  acc[2];

        input_default_t  datainput[4];
        result_default_t result[2];
}layer2_dense;



struct layer2_softmax_type: nnet::softmax_type_config{
        typedef input_default_t input_t;
        typedef result_default_t result_t;
        static const unsigned n_in  = 2;
};
// the softmax layer struct defination, store the intermediate value for backpropogation
struct layer2_softmax{
        input_default_t  datainput[2];
        result_default_t result[2];
        result_default_t sum;
}layer2_softmax;




#endif 