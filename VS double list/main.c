#include "doublylisted.h"
#include <stdio.h>

int main()
{
	//printf("%d is 1/2", 0/1);
	List list1;
	NormalNode node0, node1, node2, node3, node4;
	list1 = CreateNewList();
	if( Insert_At_Location(0, 9, list1) )
		{
			printf("\n");
		}

	
	 if( Insert_At_Location(1, 6, list1) )
	    {
			printf("\n");
	  }



	 /*
	 if( Insert_At_Location(0, 1, list1) )
	    {
			printf("\n");
	  }
	  */


	printf("\n");
	printf("print the whole list\n");
	printf("\n");


if(  PrintList(list1))
	{	
		printf("\n");
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
	  printf("\n");
	 
	 PrintNode(node1);
	 printf("\n");


	 PrintNode(node2);
	 printf("\n");

	 //PrintNode(node4);
	// printf("\n");
	// node2= Get_Node(2, list1);
	// node3= Get_Node(3, list1);

	//  PrintNode(node1);
	//  PrintNode(node2);
	//  PrintNode(node3);

	printf("\n");
	printf("print the whole list\n");
	printf("\n");


if(  PrintList(list1))
	{	
		printf("\n");
	}
}



