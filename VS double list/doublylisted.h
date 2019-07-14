/*
* doublylisted.h
*
*  Created on: 20171016
*      Author: weijia
*/

#ifndef _List_H

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


#endif  /* _List_H */

/* Place the defination of the sturct Node in the implementation file */

#pragma once



