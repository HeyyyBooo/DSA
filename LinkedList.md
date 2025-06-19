Report of Target 2025

# LINKED LIST

1. **REVERSE NOD IN K GROUPS**  
   Given the head of a linked list, reverse the nodes of the list k at a time, and return *the modified list*.  
   k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.  
   You may not alter the values in the list's nodes, only nodes themselves may be changed.

   

Just directly implementing the use case using recursion call, i am just using recursion as stack operations.  
```cpp     
   class Solution {  
   public:  
      ListNode* ans;  
      ListNode* ptr;  
      int n;  
      ListNode* last;  
      void rec(ListNode* head, int k) {  
          if (head) {  
              if (k == 1) {  
                  int v = head->val;  
                  ListNode* temp = new ListNode(v);  
                  ptr->next = temp;  
                  ptr = ptr->next;  
                  last = head->next;  
                  return;  
     
              } else {  
                  rec(head->next, k - 1);  
                  int v = head->val;  
                  ListNode* temp = new ListNode(v);  
                  ptr->next = temp;  
                  ptr = ptr->next;  
              }  
          }  
      }  
     
      ListNode* reverseKGroup(ListNode* head, int k) {  
          ans = new ListNode();  
          ptr = ans;  
          n = k;  
          int size = 0;  
          ListNode* sizetracker = head;  
          while (sizetracker) {  
              size++;  
              sizetracker = sizetracker->next;  
          }  
          int loop = size / k;  
          last=head;  
          while (loop--) {  
              rec(last, k);  
          }  
          ptr->next = last;  
          return ans->next;  
      }  
   };  
``` 
Here size tracker is used to determine the extra remaining part of the linked list that remain unchanged.

2. **Rotate List**  
   Given the head of a linked list, rotate the list to the right by k places.  
     
As k can be greater than n, the size of I am using modulo mathematical operator to simulate the ring that i will be rotating.  

```cpp
    ListNode* rotateRight(ListNode* head, int k) {  
          if(head==NULL){  
              return head;  
          }  
          ListNode* temp=head;  
          int n=1;  
          while(temp->next!=NULL){  
              temp=temp->next;  
              n++;  
          }  
          temp->next=head;  
          ListNode* trav=head;  
          int l=n-k%n;  
          L--;  
     
          while(l--){  
              trav=trav->next;  
          }  
          ListNode* ans=trav->next;  
          trav->next=NULL;  
          return ans;  
     
     
      }  
```    
First of all I am joining the head at the tail that forms the ring then finding the exact breaking point . and breaking the ring into new list.  
     
3. **FLATTEN BINARY TREE (PREORDER)**

Given the root of a binary tree, flatten the tree into a "linked list":

* The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.  
* The "linked list" should be in the same order as a [**pre-order traversal**](https://en.wikipedia.org/wiki/Tree_traversal#Pre-order,_NLR) of the binary tree.  
    
    
Basic approach is just find the preorder using the basic pre order recursive travel. And then blend the list accordingly  
```cpp
  void preorder(TreeNode *root) {  
     if (root != NULL) {  
        v.push_back(root->val);  
        preorder(root->left);  
        preorder(root->right);  
     }  
  }  
  void flatten(TreeNode* root) {  
          if(root==NULL) return ;  
          TreeNode* temp=root;  
          preorder(temp);  
          for(int i=0;i<v.size()-1;i++){  
              temp->val=v[i];  
              temp->left=NULL;  
              temp->right=new TreeNode();  
              temp=temp->right;  
          }  
          temp->val=v[v.size()-1];  
          temp->left=NULL;  
            
          return;  
      }  
```
But the follow-up is to use no extra space change it in place. So its  a relative to morris algo.  
here we need to traverse right most element of left subtree and connect it with the right subtree and make left subtree null until everything is flattened.  
    
    

```cpp 
  void flatten(TreeNode* root) {  
         if (!root)  
             return;  
         TreeNode* mover = root;  
         while (mover) {  
             if (mover->left) {  
                 TreeNode* temp = mover->left;  
                 while (temp->right)  
                     temp = temp->right;  
                 temp->right = mover->right;  
                 mover->right = mover->left;  
                 mover->left = nullptr;  
             }  
             mover = mover->right;  
         }  
     }  
```
  So now the space complexity is O(1).  
    
4. **COPY LIST WITH RANDOM POINTER**  
   A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.  
   Construct a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) of the list. The deep copy should consist of exactly n **brand new** nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. **None of the pointers in the new list should point to nodes in the original list**.  
     
     
   
I am going to use map to maintain mapping between original and deep copy then using that map i am going to connect the copy list and then simply execute this,  

```cpp
   Node* copyRandomList(Node* head) {  
          if (head == NULL) {  
              return head;  
          }  
          map<Node*, Node*> mp;  
          Node* copy = new Node(head->val);  
          Node* p1 = head;  
          Node* p2 = copy;  
          p1 = p1->next;  
          mp[head] = copy;  
          while (p1) {  
              Node* temp = new Node(p1->val);  
              mp[p1] = temp;  
              p1 = p1->next;  
              p2->next = temp;  
              p2=p2->next;  
          }  
          p1 = head;  
          p2 = copy;  
          while (p1) {  
              if (p1->random!=NULL) {  
                  p2->random = mp[p1->random];  
              }  
              p1 = p1->next;  
              p2 = p2->next;  
          }  
          return copy;  
      }  
```   
5. **Linked List Cycle**  
   Given head, the head of a linked list, determine if the linked list has a cycle in it.  
     
   
Approach one is that i am going to use slow &  fast pointer approach when these two pointer will match then thats the point of cycle.  
```cpp
   bool hasCycle(ListNode *head) {  
          ListNode* fast = head;  
          ListNode* slow = head;  
     
          while (fast != nullptr && fast->next != nullptr) {  
              fast = fast->next->next;  
              slow = slow->next;  
     
              if (fast == slow) {  
                  return true;  
              }  
          }  
     
          return false;         
      }  
  ```
Another approach is that i can use a footprint where i travel , if i find footprint while traversing that means its already visited. I used infinity as foot print as according to constraints its not a valid value of linked list.  
   
```cpp
   bool hasCycle(ListNode *head) {  
         ListNode* temp=head;  
         bool ans=false;  
         while(temp!=NULL){  
             if(temp->val==INT_MIN){  
                 ans=true;  
                 break;  
             }  
             temp->val=INT_MIN;  
             temp=temp->next;  
     
         }  
         return ans;    
       }  
```
   Both uses constant extra space.   
     
6. **Linked List Cycle II**

   Given the head of a linked list, return *the node where the cycle begins. If there is no cycle, return* null.

   There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (**0-indexed**). It is \-1 if there is no cycle. **Note that** pos **is not passed as a parameter**.

   

   

As used in earlier problem i am just using flyods algo for cycle detection.Use two pointers slow and fast. Move slow by one step and fast by two. If they meet, a cycle exists. 

Then: Reset slow to head and move both one step at a time — they’ll meet at the cycle start.

   
```cpp
   ListNode* detectCycle(ListNode* head) {
          if(!head) return NULL;
          if(head->next==NULL) return NULL;
          ListNode* slow = head;
          ListNode* fast = head;
          bool flag=false;
          while(fast and fast->next){
              slow=slow->next;
              fast=fast->next->next;
              if(slow==fast){
                flag=true;
                break;
              }
          }
          if(flag){
              slow=head;
              while(slow!=fast){
                  slow=slow->next;
                  fast=fast->next;
              }
              return slow;
          }
          return NULL;
      }
```
It has a linear time complexity.

   

7. **Reverse Linked List**  
   Given the head of a singly linked list, reverse the list, and return *the reversed list*.  
Iterative: 

```cpp 
     ListNode* reverseList(ListNode* head) {  
           ListNode* node = nullptr;  
     
           while (head != nullptr) {  
               ListNode* temp = head->next;  
               head->next = node;  
               node = head;  
               head = temp;  
           }  
     
           return node;          
       }  
```
Recursive:
```cpp
    ListNode* headR;  
    ListNode* ptr;  
    void rec(ListNode* node){  
       if(node){  
           rec(node->next);  
           ptr->next=node;  
           ptr=ptr->next;  
       }  
      return;  
     }  
   ListNode* reverseList(ListNode* head) {  
       headR= new ListNode();  
       ptr=headR;  
       rec(head);  
       ptr->next=NULL;  
       return headR->next;  
   } 
``` 
Both are equivalent approach just using a temporary pointer to reverse the list and return the new head.

8. **Palindrome Linked List**  
   Given the head of a singly linked list, return true *if it is a palindrome or* false *otherwise*.  
     
I am first finding the mid and pushing all values before mid into a stack then checking if its odd or even . If odd then skipping the middle one then start popping and matching with the second half of the list. If found a mismatch then  *false*  else its  *true.*  


```cpp    
   bool isPalindrome(ListNode* head) {  
      if (!head || !head->next) return true;  
     
      ListNode* slow = head;  
      ListNode* fast = head;  
      stack<int> st;  
      while (fast && fast->next) {  
          st.push(slow->val);  
          slow = slow->next;  
          fast = fast->next->next;  
      }  
      if (fast) {  
          slow = slow->next;  
      }  
      while (slow) {  
          int top = st.top();  
          st.pop();  
          if (top != slow->val) return false;  
          slow = slow->next;  
      }  
     
      return true;  
   } 
``` 
The follow up of the problem is to do it in O(n) Time and O(1) extra space. So I am going to do it recursively .  
Using same reverse the list approach  and matching.  

```cpp   
   ListNode* ptr;  
      bool ans;  
      void rec(ListNode* head){  
          if(head){  
              rec(head->next);  
              if(head->val!=ptr->val){  
                  ans=false;  
              }  
              head=head->next;  
              ptr=ptr->next;  
          }  
          return;  
      }  
     
     
     
      bool isPalindrome(ListNode* head) {  
      if (!head || !head->next) return true;  
      ptr=head;  
      ans=true;  
      rec(head);  
      return ans;  
     
   }  
```
But if we consider the stack usage of recursion then its again O(n) space.  
So to do it without recursion and without extra space.  
I am going to find the half of the LInked list and going to reverse the second half iteratively then will compare the both lists values.  


```cpp   
   bool isPalindrome(ListNode* head) {  
          ListNode* slow=head;  
          ListNode* fast=head;  
          while(fast and fast->next){  
              fast=fast->next->next;  
              slow=slow->next;  
          }  
          if(fast){  
              slow->next;  
          }  
          ListNode*head2=NULL;  
          while(slow){  
              ListNode* temp=slow->next;  
              slow->next=head2;  
              head2=slow;  
              slow=temp;  
          }  
          while(head2){  
              if(head->val!=head2->val){  
                  return false;  
              }  
              head=head->next;  
              head2=head2->next;  
          }  
          return true;  
      }  
```  
Now its Time Complexity is O(n)  and Space Complexity is O(1).  
     
9. **Delete Node in a Linked List**  
   There is a singly-linked list head and we want to delete a node node in it.  
   You are given the node to be deleted node. You will **not be given access** to the first node of head.  
   All the values of the linked list are **unique**, and it is guaranteed that the given node node is not the last node in the linked list.  
   Delete the given node. Note that by deleting the node, we do not mean removing it from memory. We mean:  
* The value of the given node should not exist in the linked list.  
* The number of nodes in the linked list should decrease by one.  
* All the values before node should be in the same order.  
* All the values after node should be in the same order.  
    
As I only have that node so i cannt modify the storage so i will copy the next value in current and will delete the last memory cell of Linked list . As a result it will mimic as the List with removed node.  

```cpp    
  void deleteNode(ListNode* node) {  
         while(node->next->next){  
             node->val=node->next->val;  
             node=node->next;  
         }  
         node->val=node->next->val;  
         node->next=NULL;  
     }  
```   
    
10. **Reorder List**  
    You are given the head of a singly linked-list. The list can be represented as:

    L0 → L1 → … → Ln \- 1 → Ln  
    

    *Reorder the list to be on the following form:*

    L0 → Ln → L1 → Ln \- 1 → L2 → Ln \- 2 → …  
    

    You may not modify the values in the list's nodes. Only nodes themselves may be changed.

      
      
    

First Task is to find the middle of the list. And break it into two halves.  
```cpp
      ListNode*slow=head;  
            ListNode*fast=head;  
            while(fast and fast->next){  
                slow=slow->next;  
                fast=fast->next->next;  
            }  
            ListNode* prev=NULL;  
            ListNode* curr=slow->next;  
            slow->next=NULL;  
```      
Then reverse the second half.
```cpp  
    	while(curr){  
                ListNode*temp=curr->next;  
                curr->next=prev;  
                prev=curr;  
                curr=temp;  
            }  
```
Then at last using two pointers join the list in given pattern.  
```cpp
    	    ListNode* first=head;  
            ListNode* second=prev;  
            while(second){  
                ListNode* temp1=first->next;  
                ListNode* temp2=second->next;  
                first->next=second;  
                second->next=temp1;  
                first=temp1;  
                second=temp2;  
            }  
```
So the work is done in place without using dataset and done in Linear Time Complexity.  
      
    
