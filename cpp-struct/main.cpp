#include <iostream>
//#include "layer.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "doublylisted.h"
#include "parameters.h"
#include <stdio.h>

int main()
{
    std::cout<<"enter in the main.cpp"<<std::endl;
    /* 
    BpNet testNet; 
    // 学习样本 
    
    vector<double> samplein[4]; 
    vector<double> sampleout[4]; 
    samplein[0].push_back(0); 
    samplein[0].push_back(0); 
    sampleout[0].push_back(0); 
    samplein[1].push_back(0); 
    samplein[1].push_back(1); 
    
    sampleout[1].push_back(1); 
    samplein[2].push_back(1); 
    samplein[2].push_back(0); 
    sampleout[2].push_back(1); 
    samplein[3].push_back(1); 
    samplein[3].push_back(1); 
    sampleout[3].push_back(0); 
    sample sampleInOut[4]; 
    
    for (int i = 0; i < 4; i++) 
    { 
        sampleInOut[i].in = samplein[i]; 
        sampleInOut[i].out = sampleout[i]; 
    } 
    
    vector<sample> sampleGroup(sampleInOut, sampleInOut + 4); 
    
    testNet.training(sampleGroup, 0.0001); 
    // 测试数据 
    
    vector<double> testin[4]; 
    vector<double> testout[4]; 
    testin[0].push_back(0.1); 
    testin[0].push_back(0.2); 
    testin[1].push_back(0.15); 
    testin[1].push_back(0.9); 
    testin[2].push_back(1.1); 
    testin[2].push_back(0.01); 
    testin[3].push_back(0.88); 
    testin[3].push_back(1.03); 
    sample testInOut[4]; 
    for (int i = 0; i < 4; i++) 
    testInOut[i].in = testin[i]; 
    
    vector<sample> testGroup(testInOut, testInOut + 4); 
    // 预测测试数据，并输出结果 
    testNet.predict(testGroup); 
    for (int i = 0; i < testGroup.size(); i++) 
    { 
        for (int j = 0; j < testGroup[i].in.size(); j++) 
        cout << testGroup[i].in[j] << "\t"; 
        cout << "-- prediction :"; 
        for (int j = 0; j < testGroup[i].out.size(); j++) 
        cout << testGroup[i].out[j] << "\t"; 
        cout << endl; 
    }  
   */









/***************************************************** */
/***       dense layer forward compute test bench    ***/
/****************************************************** */



    // testbench for the cpu_forward dense layer
    




    layer1_dense_type::accum_t inputdata_test[2]= {5,6};
    layer1_dense_type::accum_t result_test[4] = {0,0,0,0};

    layer1_dense_type::accum_t w1[2][4] = {{0.1f,0.1f,0.1f,0.1f},{0,2,0,0}};
    layer1_dense_type::accum_t b1[4] = {1,1,1,1};

    layer1_dense_type::accum_t w2[4][2] = {{0.1f,0.1f}, {0.1f,0.1f},{0,2},{0,0}};
    layer1_dense_type::accum_t b2[2] = {1,1};



    
    Initialize_weight1: for(int ii = 0; ii<layer1_dense_type::n_in; ii++)
    {
        for(int jj=0; jj<layer1_dense_type::n_out; jj++)
        {
            layer1_dense.weights[ii][jj] = w1[ii][jj];
        }
    }
    Initialize_bias1: for(int jj = 0; jj<layer1_dense_type::n_out; jj++)
    {
        layer1_dense.biases[jj] = b1[jj];
    }



    Initialize_weight2: for(int ii = 0; ii<layer2_dense_type::n_in; ii++)
    {
        for(int jj=0; jj<layer2_dense_type::n_out; jj++)
        {
            layer2_dense.weights[ii][jj] = w2[ii][jj];
        }
    }
    Initialize_bias2: for(int jj = 0; jj<layer2_dense_type::n_out; jj++)
    {
        layer2_dense.biases[jj] = b2[jj];
    }


    Initialize_datainput: for(int ii = 0; ii<layer1_dense_type::n_in; ii++)
    {
        layer1_dense.datainput[ii] = inputdata_test[ii];
    }

    Initialize_layer1_result: for(int jj = 0; jj<layer1_dense_type::n_out; jj++)
    {
        layer1_dense.result[jj] = result_test[jj];
    }





     

    // nnet::compute_layer<nnet_dense_layer1>(data,output,w1,b1);
    nnet::compute_layer<layer1_dense_type>(layer1_dense.datainput,layer1_dense.result,layer1_dense.weights,layer1_dense.biases);
    nnet::active_tanh<layer1_tanh_type>(layer1_dense.result,layer1_tanh.result);

    nnet::compute_layer<layer2_dense_type>(layer1_tanh.result,layer2_dense.result,layer2_dense.weights,layer2_dense.biases);
    nnet::softmax_layer<layer2_softmax_type>(layer2_dense.result,layer2_softmax.result,&layer2_softmax.sum);
    


    std::cout<< "layer2 output[0] is "<<layer2_dense.result[0]<<std::endl;
    std::cout<< "layer2 output[1] is "<<layer2_dense.result[1]<<std::endl;

    std::cout<< "softmax output[0] is "<<layer2_softmax.result[0]<<std::endl;
    std::cout<< "softmax output[1] is "<<layer2_softmax.result[1]<<std::endl;


    std::cout<< "softmax exp sum is "<<layer2_softmax.sum<<std::endl;

    // std::cout<< "output[3] is "<<layer2_dense.result[3]<<std::endl;
    // after two layers the output is just have 2 value.


    // ***************** finish of the forward propagation *********************//




    // ****************** test bench for the double list ************************//


	//printf("%d is 1/2", 0/1);
	List list1;
	NormalNode node0, node1, node2, node3, node4;
	list1 = CreateNewList();
	if( Insert_At_Location(int(0), int(9), list1 ))
		{
			std::cout<<""<<std::endl;
		}

	
	 if( Insert_At_Location(1, 6, list1 ))
	    {
			std::cout<<""<<std::endl;
	  }



	 /*
	 if( Insert_At_Location(0, 1, list1) )
	    {
			printf("\n");
	  }
	  */


	std::cout<<""<<std::endl;
	std::cout<<"print the whole list\n"<<std::endl;
	std::cout<<""<<std::endl;


if(  PrintList(list1))
	{	
		std::cout<<"print the whole list\n"<<std::endl;
	}

   /*
	 if( Insert_At_Location(2, 2, list1) )
	  {
			printf("\n");
	  }
*/

	 node0 = Get_Node(0, list1);
	
	 node1 = Get_Node(1, list1);
	 node2 = Get_Node(2, list1);


	 // node4 = Get_Node(4, list1);

	  PrintNode(node0);
	    std::cout<<""<<std::endl;
	 
	  PrintNode(node1);
	    std::cout<<""<<std::endl;


	  PrintNode(node2);
	  	std::cout<<""<<std::endl;

	  //PrintNode(node4);
	  // printf("\n");
	  // node2= Get_Node(2, list1);
	  // node3= Get_Node(3, list1);

	  //  PrintNode(node1);
	  //  PrintNode(node2);
	  //  PrintNode(node3);

	std::cout<<""<<std::endl;


     if(  PrintList(list1))
	 {	
			std::cout<<""<<std::endl;
	 }
























    std::cout<<"finish of the main.cpp"<<std::endl;




    return 0;
}