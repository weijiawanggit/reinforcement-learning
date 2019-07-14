/*
* doublylisted.h
*
*  Created on: 20171016
*      Author: weijia
*/

#ifndef DOUBLYLISTED_H
#define DOUBLYLISTED_H
#include "stdio.h"
#include <stdlib.h>



typedef int ElementType;

struct Node
{
	ElementType element;
	int Location_Num;
	// the location of the Node in the List
	struct Node* next;
	struct Node* prev;
};


struct Node; // the defination of the node is in the c file
typedef struct Node *PtrToNode;
typedef PtrToNode List;  // the pointer to the head node of the list
typedef PtrToNode NormalNode;


// here, the List and NormalNode both are in the type of Node *
// they are both the pointer


List CreateNewList();
/*if successful, return 0*/
int DeleteList(List L);
int IsEmpty(List L);
int IsLast(NormalNode P, List L);


// Only create a new node, not assign the prev pointer nor the next pointer
PtrToNode CreateNewNode(ElementType element);


/* Find the Node containing the element in the list and return the first pointer to the Node,print all the Node containing the element*/
NormalNode Find_First(ElementType element, List L);
int Find_All(ElementType element, List L);


/* Get the pointer of the node given by the location */
NormalNode Get_Node(int index, List L);




// Delete the Node according to the Location number
int DeleteLocation_Num(int index, List L);

int DeleteNodebyPointer(NormalNode P, List L);

int Insert_At_Location(int index, ElementType element, List L);

int PrintList(List L);
void PrintNode(NormalNode P);


// List list_head = NULL; // global variable - pointer to head node.
// static int  count = 0;
// there shoud be a count in the .c file, but we put this counter into the head node, store at the element of head node
// In vivado, this kind of type may be mapped into a register 



List CreateNewList()
{
	List list_head = CreateNewNode(0);
		if (!list_head)
		{
			printf("Create List head in failure!\n");
			return NULL;
		}
			
		list_head->Location_Num = 0;
		list_head->prev = list_head;
		list_head->next = list_head;
		list_head->element = 0; // the element number in the list now is 0 
		printf("Create List head successful!\n\n");
	  return list_head;
}




/* Delete the Whole Node in the List */
/*if successful, return 0*/
int DeleteList(List L)
{
	NormalNode P, Tmp;

	P = L->next;  // P point to the first node after the head
	L->next = NULL;
	free(L);   // delete the head of the List
	while (P != NULL)
		{
			Tmp = P->next;
			free(P);
			P = Tmp;
		}
	// when the P come to the last Node Null, the delete work will be finished
	return 0;
}




/* judge is the list empty */

int IsEmpty(List L)
{
     return L->next == NULL;
}



/* Return true if P is the last NormalNode/location in list L */

int IsLast(NormalNode P, List L)
{
	return P->next == NULL;
}




/* Creates a new Node by a element, and returns pointer to the new node */







NormalNode CreateNewNode(ElementType element)
{
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
				//note that malloc(sizeof(PtrToNode)) is leagal, but could not allocate enough space
                // because it is just a pointer, the space size is just int  (32 bit system)

	newNode->element = element;
	newNode->Location_Num = 0;
	newNode->prev = NULL;
	newNode->next = NULL;

	if (!newNode)
	{
		printf("create node error!\n");
		return NULL;
	}
	else
	{ 
		printf("create node SUCCESSFULLY!\n");
		return newNode;
	}
}
//只是先创建了一个node, 还没有插入到list中，如果插入的话，应该更新node中的所有参数。





/* Find the Node containing the element in the list and return the first pointer to the Node,
print all the Node containing the element*/

NormalNode Find_First(ElementType element, List L)
{
	NormalNode P;
	//NormalNode P_First_element;

	P = L->next;
	while (P != NULL && (P->element = !element))
	{

		if (P == NULL)
			{	printf("There is no Node containing the element\n");
				return NULL;
			}
		if (P->element == element)
		{
			printf("The first Node containing element finded, the location of the node is %d\n", P->Location_Num);
			return P;
		}
		P = P->next;
	} // while loop is end
}





/*find all the node containing the same element*/

int Find_All(ElementType element, List L)
{
	NormalNode P;
//NormalNode P_First_element;

	P = L->next;
	while (P != NULL)
		{
			if (P->element == element)
		{
		printf("The Node containing element is in the location%d\n", P->Location_Num);
	}

P = P->next;
}
return 0;
}





/* find a node in a List by the location */

NormalNode Get_Node(int index, List L)
{
	 NormalNode P = L;
	if(!L)
	{			
				printf("%s failed! The list is not exist!\n", __FUNCTION__);
				return NULL;
	}

	if (index < 0 || index > L->element)
	{
				printf("%s failed! index out of bound!\n", __FUNCTION__);
				return NULL;
	}
	
    if (index == 0)
	{  return L;}

		// note that the L->element is the number of node in a list
		
	
	
		P = P->next;
		while (P->Location_Num < index )
		{
			   P = P->next;
		}
		return P;
}




/* If successfully delete the node, return 0, or return NULL */
int DeleteLocation_Num(int index, List L)
{
		NormalNode P;
		if (( index <= 0) || (index>L->element))
		{
			printf("%s failed! index out of bound!\n", __FUNCTION__);
			return 1;
		}

		P = L->next;
		while (P != NULL)
		{
					if (P->Location_Num == index)
					{
							("The Node in %d location will be deleted\n", P->Location_Num);
							P->prev->next = P->next;
							P->next->prev = P->prev;

							free(P);
							L->element--;
							printf("The Node %d has been deleted successfully\n", P->Location_Num);
							return 0;
					}
		P->next;
		}
}




int DeleteNodebyPointer(NormalNode P, List L)
{
	if (!P)
		{
			printf("The pointer of the Node is wrong!");
			return 1;
		}

			printf("The Node in %d location will be deleted\n", P->Location_Num);
			P->prev->next = P->next;
			P->next->prev = P->prev;

			free(P);
			L->element--;
			printf("The Node %d has been deleted successfully\n", P->Location_Num);
		return 0;
}






int Insert_At_Location(int index, ElementType element, List L)
{
		NormalNode P, P_last, P_next, Tmp;

		if ( (index < 0)   ||  (index > L->element) )
		{printf("%s failed! index out of bound!\n", __FUNCTION__);}


                P = Get_Node(index, L);
				
				// use the element to create a new node
				Tmp = CreateNewNode(element);


				// The Tmp will replace the location of the current P, P go to the next NormalNode
				Tmp->prev = P;
				Tmp->next = P->next;
				Tmp->Location_Num = index+1;

				P->next = Tmp;
				P->Location_Num = index;


				P_next = Tmp->next;

				// the index after the Tmp has to be updated
				while (P_next->Location_Num > 0 )
				{    
					P_next->Location_Num= P_next->Location_Num + 1;
					P_next = P_next->next;
				}

				// the total number of element in a list should increment.
				L->element++;
				

				// get the last node in the list
				P_last = Get_Node(L->element, L);
				L->prev = P_last;
				
			   printf("The Node has been added in the location %d \n\n", Tmp->Location_Num);
		return 1;
}





/*
		if (index == 0 && L->element == 0)
		{ 
				printf("The Node will be added in the first location");
				P = Get_Node(index, L);
				// use the element to create a new node
				Tmp = CreateNewNode(element);


				// The Tmp will replace the location of the current P, P go to the next NormalNode
				Tmp->prev = P;
				Tmp->next = L;
				Tmp->Location_Num = index;
				P->next = Tmp;
				L->element++;
		return 1;
		}

	if (index == (++L->element))
	{
			printf("The Node will be added in the final location %d", P->Location_Num);

			// use the element to create a new node
			Tmp = CreateNewNode(element);


			// The Tmp will replace the location of the current P, P go to the next NormalNode
			Tmp->prev = P;
			Tmp->next = L;
			Tmp->Location_Num = index;
			P->next = Tmp;
			return 1;
	}



	else
	{
			printf("The Node will be added in the location %d", P->Location_Num);

// use the element to create a new node
			Tmp = CreateNewNode(element);


// The Tmp will replace the location of the current P, P go to the next NormalNode
			Tmp->prev = P->prev;
			Tmp->next = P;
			Tmp->Location_Num = index;
			P->prev = Tmp;

			L->element++;
			return 1;
		}
*/
int PrintList(List L)
{       NormalNode P;
		P = L->next;
		if(!L)
		{
			printf("The list inputted is not exist\n");
			return 0;
		}

		printf("The Head Node in %d location with the number %d\n", L->Location_Num, L->element );

		
		while (P->Location_Num > 0 )
		{
			   printf("The Node in %d location with the number %d\n", P->Location_Num, P->element );
               P = P->next;
		}
		
		
	return 1;
}



void PrintNode(NormalNode P)
{	
			printf("The Node in %d location with the number %d\n", P->Location_Num, P->element );
			printf("The Node Prev in %d location with the number %d\n", P->prev->Location_Num, P->prev->element );
			printf("The Node Next in %d location with the number %d\n", P->next->Location_Num, P->next->element );
}





#endif  /* _List_H */



