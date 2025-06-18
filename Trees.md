Report of Target 2025

# TREE

1. **Inorder Traversal  (ITERATIVE)**  
```cpp
   vector<int> inorderTraversal(TreeNode* root) {  
           vector<int>ans;  
           stack<TreeNode*>st;  
           TreeNode* curr=root;  
           while(curr or !st.empty()){  
               while(curr){  
                   st.push(curr);  
                   curr=curr->left;  
               }  
               curr=st.top();  
               st.pop();  
               ans.push_back(curr->val);  
               curr=curr->right;  
           }  
           return ans;  
       }  
```
2. **Preorder Traversal (ITERATIVE)**
```cpp
	vector<int> preorderTraversal(TreeNode* root) {  
        vector<int> ans;  
        stack<TreeNode*> st;  
        TreeNode* curr = root;  
        while (curr or !st.empty()) {  
            while (curr) {  
                ans.push_back(curr->val);  
                st.push(curr);  
                curr = curr->left;  
            }  
            curr = st.top();  
            st.pop();

            curr = curr->right;  
        }  
        return ans;  
    }
```
3. **Postorder Traversal (ITERATIVE)**  
```cpp
   vector<int> postorderTraversal(TreeNode* root) {  
           vector<int> ans;  
           stack<TreeNode*>st;  
           TreeNode* curr=root;  
           while(!st.empty() or curr){  
               if(curr){  
                   st.push(curr);  
                   ans.push_back(curr->val);  
                   curr=curr->right;  
               }  
               else{  
                   TreeNode* node=st.top();st.pop();  
                   curr=node->left;  
               }  
           }  
           reverse(ans.begin(),ans.end());  
           return ans;  
       }  
```
     
     
4. **Symmetric Tree**  
   Given the root of a binary tree, *check whether it is a mirror of itself* (i.e., symmetric around its center).  
```python 
   def isSymmetric(self, root: Optional[TreeNode]) -> bool:  
           def rec(node1,node2):  
               if not node1 and not node2:  
                   return True  
               if not node1 or not node2:  
                   return False  
               if node1.val!=node2.val:  
                   return False  
               return rec(node1.left,node2.right) and rec(node1.right,node2.left)  
           return rec(root.left,root.right) 
``` 
5. **Max Depth**  
   Given the root of a binary tree, return *its maximum depth*. 
```python  
           def rec(root):  
               if not root:  
                   return 0  
               return max(rec(root.left),rec(root.right))+1  
```
6. **Balanced Binary Tree**  
   Given a binary tree, determine if it is **height-balanced**.  
```python  
   def isBalanced(self, root: Optional[TreeNode]) -> bool:  
           def rec(root):  
               if not root:  
                   return 0,True  
               left,ansL=rec(root.left)  
               right,ansR=rec(root.right)  
               return max(left,right)+1, abs(left-right)<=1 and ansL and ansR  
           depth,ans=rec(root)  
           return ans  
```
7. **Binary Tree Level Order Traversal**  
   Given the root of a binary tree, return *the level order traversal of its nodes' values*. (i.e., from left to right, level by level).  
     
   This is the recursive solution, a size function for list determination followed by the main traversal.  
```python 
   def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:  
           n=0  
           def size(root,i):  
               nonlocal n  
               if not root:  
                   return  
               n=max(n,i)  
               size(root.left,i+1)  
               size(root.right,i+1)  
           size(root,1)  
           ans=[[] for _ in range(n)]  
           def rec(root,i):  
               if not root:  
                   return  
               ans[i].append(root.val)  
               rec(root.left,i+1)  
               rec(root.right,i+1)  
           rec(root,0)  
           return ans  
```
     
   The iterative solution using BFS queue is as follows  
```python   
   from collections import deque  
   class Solution:  
       def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:  
           q=deque()  
           if not root:  
               return []  
           q.append(root)  
           ans=[]  
           while len(q)>0:  
               n=len(q)  
               nodes=[]  
               for i in range(n):  
                   temp=q.popleft()  
                   nodes.append(temp.val)  
                   if temp.left :  
                       q.append(temp.left)  
                   if temp.right :  
                       q.append(temp.right)  
               ans.append(nodes)  
           return ans  
```
8. **Binary Tree ZigZag Traversal**  
   Given the root of a binary tree, return *the zigzag level order traversal of its nodes' values*. (i.e., from left to right, then right to left for the next level and alternate between)  
     
   Implementing same as above just for odd iteration reversing the list and then appending it in ans.  
```python 
   def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:  
           if not root:  
               return []  
           ans=[]  
           q=deque()  
           i=0  
           q.append(root)  
           while len(q)>0:  
               nodes=[]  
               for _ in range(len(q)):  
                   temp=q.popleft()  
                   nodes.append(temp.val)  
                   if temp.left:  
                       q.append(temp.left)  
                   if temp.right:  
                       q.append(temp.right)  
               if i%2==0:  
                   ans.append(nodes)  
               else:  
                   nodes.reverse()  
                   ans.append(nodes)  
               i+=1  
           return ans  
```  
9. **Diameter of a Binary Tree**  
   Given the root of a binary tree, return *the length of the **diameter** of the tree*.  
   The **diameter** of a binary tree is the **length** of the longest path between any two nodes in a tree. This path may or may not pass through the root.  
   The **length** of a path between two nodes is represented by the number of edges between them.

	Direct implementation of depth of binary tree using one global variable.  
```python 
	def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:  
        d=0  
        def rec(root):  
            nonlocal d  
            if not root :  
                return 0  
            l=rec(root.left)  
            r=rec(root.right)  
            d=max(d,l+r)  
            return max(l,r)+1  
        rec(root)  
        return d  
```

10. **Validate Binary Search Tree**  
    Given the root of a binary tree, *determine if it is a valid binary search tree (BST)*.  
    A **valid BST** is defined as follows:  
* The left subtree of a node contains only nodes with keys **less than** the node's key.  
* The right subtree of a node contains only nodes with keys **greater than** the node's key.  
* Both the left and right subtrees must also be binary search trees.  
    
  First approach is the list containing the inorder traversal of the tree will be strictly increasing for a Valid BST
```python 
  def isValidBST(self, root: Optional[TreeNode]) -> bool:

          inord=[]

          def rec(root):

              if root :

                  rec(root.left)

                  inord.append(root.val)

                  rec(root.right)

          rec(root)

          for i in range(1,len(inord)):

              if(inord[i]<=inord[i-1]):

                  return False

          return True  
```
  And secondly we can directly do it using O(n) extra space by maintaining a prev variable for containing previous value which we will check during traversal.
```python 
  def isValidBST(self, root: Optional[TreeNode]) -> bool:

          prev = float('-inf')

          def inorder(node):

              nonlocal prev

              if not node:

                  return True

              if not (inorder(node.left) and prev < node.val):

                  return False

              prev = node.val

              return inorder(node.right)

          return inorder(root)

```
  And another approach will be using min max counters for all states in recursion.

```python    
def isValidBST(self, root: Optional[TreeNode]) -> bool:  
        def rec(root,minVal,maxVal):  
            if not root:  
                return True  
            if root.val>=maxVal or root.val<=minVal:  
                return False  
return rec(root.left,minVal,root.val) and rec(root.right,root.val,maxVal)  
    return rec(root,-inf,inf)  
```
This is checking left and right limit for each state.

11. **Convert Sorted Array to Binary Balanced Tree**  
    Given an integer array nums where the elements are sorted in **ascending order**, convert *it to a **height-balanced*** *binary search tree*.  
      
    Using the approach of left right mid of binary search in nums we will build the tree.  
```python 
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:  
            def rec(nums,left,right):  
                if left<=right:  
                    mid=(left+right)//2  
                    root=TreeNode(nums[mid])  
                    root.left=rec(nums,left,mid-1)  
                    root.right=rec(nums,mid+1,right)  
                    return root  
                return None  
            return rec(nums,0,len(nums)-1) 
``` 
    On solving this recurrence relation the time complexity is O(n)  
      
12. **Two Sum BST Version**  
    Given the root of a binary search tree and an integer k, return true *if there exist two elements in the BST such that their sum is equal to* k, *or* false *otherwise*.  
```python 
    from collections import defaultdict  
        def findTarget(self, root: Optional[TreeNode], k: int) -> bool:  
            mp=defaultdict(int)  
            ans=False  
            def rec(root,k):  
                nonlocal ans  
                if not root :  
                    return  
                if k-root.val in mp:  
                    ans=True  
                    return  
                mp[root.val]+=1  
                rec(root.right,k)  
                rec(root.left,k)  
            rec(root,k)  
            return ans  
```
    The hashmap implementation just using recursion here.  
      
13. **Kth Smallest in BST**  
    Given the root of a binary search tree, and an integer k, return *the* kth *smallest value (**1-indexed**) of all the values of the nodes in the tree*.  
      
    Again the same inorder traversal technique will be used here.   
      
```python 
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:  
            i,ans=0,0  
            def rec(root):  
                nonlocal i,ans  
                if not root :  
                    return  
                if i>=k:  
                    return  
                rec(root.left)  
                i+=1  
                if i==k:  
                    ans=root.val  
                rec(root.right)  
            rec(root)  
            return ans 
``` 
    Again the time complexity is same as of inorder traversal.  
    And we are pruning when i value is greater than k.  
      
14. **Binary Search Tree Iterator**  
    Implement the BSTIterator class that represents an iterator over the [**in-order traversal**](https://en.wikipedia.org/wiki/Tree_traversal#In-order_\(LNR\)) of a binary search tree (BST):  
* BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.  
* boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.  
* int next() Moves the pointer to the right, then returns the number at the pointer.  
  Notice that by initializing the pointer to a non-existent smallest number, the first call to next() will return the smallest element in the BST.  
  You may assume that next() calls will always be valid. That is, there will be at least a next number in the in-order traversal when next() is called.  
    
* Could you implement next() and hasNext() to run in average O(1) time and use O(h) memory, where h is the height of the tree?  
    
  So Normal implementation will be collecting the inorder Traversal and fetching in constant time but its taking O(n) space where n is the number of nodes. To do it in O(h) space where h is the height of the tree surely this is the hint we have to do it using stack and simulate the inorder partial traversal.  
    
    
```cpp
  class BSTIterator {  
  public:  
      stack<TreeNode*>st;  
      TreeNode* node;  
    
      BSTIterator(TreeNode* root) {  
          node=root;  
          st={};  
      }  
       
      int next() {  
          while(node){  
              st.push(node);  
              node=node->left;  
          }  
          TreeNode* temp=st.top();st.pop();  
          node=temp->right;  
          return temp->val;  
      }  
    
      bool hasNext() {  
          return st.size()>0 or node;  
      }  
  };  
```
  So it will result a average case of O(1) Time and O(h) space complexity.  
    
15. **Lowest Common Ancestor of BST**  
    Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.  
      
    By utilizing the property of BST we can directly traverse to the nearest node that has the p and q at two different sides of the node.  
```python 
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':  
            while(root):  
                if p.val<root.val and q.val<root.val:  
                    root=root.left  
                elif p.val>root.val and q.val>root.val:  
                    root=root.right  
                else :  
                    return root  
            return root  
```
16. **Lowest Common Ancestor of Binary Tree**  
    Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.  
```python 
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':  
            def rec(root,p,q):  
                #if tree ended or got the targets p and q  
                if not root or root==p or root==q:  
                    return root  
                left=rec(root.left,p,q)  
                right=rec(root.right,p,q)  
                #Both Left and Right didn't find p and q then return NULL.  
                if not left and not right:  
                    return None  
                #If any one of them finded then returning that  
                if not left:  
                    return right  
                elif not right:  
                    return left  
                #if both got the p and q that's the Anc    
                else:  
                    return root  
            return rec(root,p,q)  
``` 
    Checking for targets otherwise returning a Null pointer.  
    If found then return them.  
    If a node has both left and right as given targets then that's our answer doing it in O(n).  
      
17. **Max Width of Binary Tree**  
    Given the root of a binary tree, return *the **maximum width** of the given tree*.  
    The **maximum width** of a tree is the maximum **width** among all levels.  
    The **width** of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes), where the null nodes between the end-nodes that would be present in a complete binary tree extending down to that level are also counted into the length calculation.  
    If we index the tree with 0 based tree indexing technique then the width in any level is the rightmostindex-leftmostindex \+1  
      
      
      
    So we just do level order traversing using queue.  
    And queue will contain pair of node and the index of that node.  
    And the last and first one on  a level will be considered for width calculation.  
```python  
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:  
            if not root:  
                return 0  
            ans=0  
            q=deque()  
            q.append((root,0))  
            while len(q)>0:  
                n=len(q)  
                for i in range(n):  
                    node,curr_id=q[0]  
                    q.popleft()  
                    if i==0 :  
                        first=curr_id  
                    if i==n-1:  
                        last=curr_id  
                    if node.left:  
                        q.append((node.left,curr_id*2+1))  
                    if node.right:  
                        q.append((node.right,curr_id*2+2))  
                ans=max(ans,last-first+1)  
            return ans  
```
    The time complexity of this code is O(n).  
18. **Count Complete Tree Nodes**  
    Given the root of a **complete** binary tree, return the number of the nodes in the tree.  
    According to [**Wikipedia**](http://en.wikipedia.org/wiki/Binary_tree#Types_of_binary_trees), every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.  
    Design an algorithm that runs in less than O(n) time complexity  
      
    So by using property of complete trees noNodes=2height-1 and for partial one we will call it recursively for both subtrees.
```python 
    def countNodes(self, root: Optional[TreeNode]) -> int:

            def right(node):

                h=0

                while node:

                    h+=1

                    node=node.right

                return h

            def left(node):

                h=0

                while node:

                    h+=1

                    node=node.left

            if not root :

                return 0

            lh=left(root)

            rh=right(root)

            if lh==rh:

                return 2**lh-1

            return 1 + self.countNodes(root.left)+self.countNodes(root.right)
```
    The right and left id calculating left and right depth/height of subtree.

    The average time complexity of this code is Ө(logn)2.

    

19. **Recover Binary Search Tree**  
    You are given the root of a binary search tree (BST), where the values of **exactly** two nodes of the tree were swapped by mistake. *Recover the tree without changing its structure*.  
      
    **Follow up:** A solution using O(n) space is pretty straight-forward. Could you devise a constant O(1) space solution?  
      
    So as we know we need to use inorder property but due to constraint we will use morris inorder.  
```python 
    def morris_inorder_traversal(root):  
        current = root  # Start with the root of the binary tree  
      
        while current:  # Continue until all nodes are visited  
            if current.left is None:  
                # Case 1: No left child → Visit this node and go right  
                print(current.val)  # "Visit" the node  
                current = current.right  
            else:  
                # Case 2: Has a left child → Find the rightmost node in the left subtree  
                predecessor = current.left  
                while predecessor.right and predecessor.right != current:  
                    predecessor = predecessor.right  # Go to the rightmost child of left subtree  
      
                if predecessor.right is None:  
                    # Subcase A: Establish a temporary link (thread) to the current node  
                    predecessor.right = current  
                    current = current.left  # Move to left child  
                else:  
                    # Subcase B: Thread already exists → Time to remove it and visit node  
                    predecessor.right = None  # Remove the thread  
                    print(current.val)  # "Visit" the current node  
                    current = current.right  # Move to right child  
```
    I will modify this , instead of printing compare with the prev number if there is a issue I will keep track of the potential two candidates. After traversal I will swap them.  
```python 
    def recoverTree(self, root: Optional[TreeNode]) -> None:  
            first,second,prev,pred=None,None,None,None  
            while root:  
                #if left subtree exist we find its predecessor  
                #predecessor is right most element in left subtree  
                if root.left:  
                    pred=root.left  
                    #There can be temp link to root  
                    while pred.right and pred.right!=root:  
                        pred=pred.right  
              #if preds right subtree is empty then  connect temp link to root  
                    if not pred.right:  
                        pred.right=root  
                        #then move current root  
                        root=root.left  
                    else:  
                        if prev and prev.val>root.val:  
                            if not first:  
                                first=prev  
                            second=root  
                        #Break the temp link  
                        pred.right=None  
                        #store the  root in prev before moving  
                        prev=root  
                        #move the curr  
                        root=root.right  
                else:  
                    if prev and prev.val>root.val:  
                        if not first:  
                            first=prev  
                        second=root  
                    prev=root  
                    root=root.right  
            #Atlast Swap the changes values  
            if first and second:  
                first.val,second.val=second.val,first.val  
```
    After swapping the Binary tree is fully BST now.  

20. **Construct Tree using inorder and Pre/Post Order**  
      
    Using Preorder:  
```python 
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:  
            q=deque(preorder)  
            def build(q,inorder):  
                if inorder:  
                    root=TreeNode(q[0])  
                    idx=inorder.index(q[0])  
                    q.popleft()  
                    root.left=build(q,inorder[:idx])  
                    root.right=build(q,inorder[idx+1:])  
                    return root  
            return build(q,inorder)  
```
    Using Postorder:  
```python 
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:  
            q=deque(postorder)  
            def build(q,inorder):  
                if inorder:  
                    node=TreeNode(q.pop())  
                    idx=inorder.index(node.val)  
                    node.right=build(q,inorder[idx+1:])  
                    node.left=build(q,inorder[:idx])  
                    return node  
                return None  
            return build(q,inorder)  
```
21. **Construct Binary Search Tree Using PreOrder**  
    Given an array of integers preorder, which represents the **preorder traversal** of a BST (i.e., **binary search tree**), construct the tree and return *its root*.  
    Now same as the above but the difference is we have to find index of the pre order split using BST property that is all numbers less than current node will help to build left subtree and greater than will be used to build right subtree.  
```python 
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:  
            def build(preorder):  
                if not preorder:  
                    return None  
                node=TreeNode(preorder[0])  
                idx=1  
                while idx<len(preorder) and preorder[idx]<node.val:  
                    idx+=1  
                node.left=build(preorder[1:idx])  
                node.right=build(preorder[idx:])  
                return node  
            return build(preorder)  
```
22. **Vertical Order Traversal**  
    Given the root of a binary tree, calculate the **vertical order traversal** of the binary tree.  
    For each node at position (row, col), its left and right children will be at positions (row \+ 1, col \- 1\) and (row \+ 1, col \+ 1\) respectively. The root of the tree is at (0, 0).  
    The **vertical order traversal** of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.  
    Return *the **vertical order traversal** of the binary tree*.  
      
    After seeing test case I came to the conclusion that I should use that given coordinate system.  
    While doing BFS levelwise I will keep track of horizontal coordinates also like if going to right then index \+1 if left the index \-1 and so on.

	And at last compile the result according to the asked constraints.
```python 
def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:  
        mp = defaultdict(list)  
        q = deque()  
        q.append((root, 0,0))  
        lis=defaultdict(set)  
        while q:  
            node, idx, lvl = q.popleft()  
            mp[(lvl,idx)].append(node.val)  
            lis[idx].add(lvl)  
            if node.left:  
                q.append((node.left, idx - 1, lvl + 1))  
            if node.right:  
                q.append((node.right, idx + 1, lvl + 1))

        ans = []  
        for i in sorted(lis.keys()):  
            l=[]  
            for j in sorted(lis[i]):  
                l+=sorted(mp[(j,i)])  
            ans.append(l)  
             
        return ans
```
	  
I used sorted environment as asked in question. The Time complexity is   
∑group∈mp:​O(len(group)⋅log(len(group)))=O(NlogN).

23. **All Nodes at K distance from a Target Node**  
    Given the root of a binary tree, the value of a target node target, and an integer k, return *an array of the values of all nodes that have a distance* k *from the target node.*  
    You can return the answer in **any order**.  
      
      
    So we will build/modify tree pointer and will add another pointer parent in that which will point towards up.  
    Now we have three directions from a node left,right and top.  
    Applying dfs till k distance will give the desired result.  
```python    
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:  
            def add_parent(node,par):  
                if node:  
                    node.parent=par  
                    add_parent(node.left,node)  
                    add_parent(node.right,node)  
            add_parent(root,None)  
            vis=set()  
            ans=[]  
            def dfs(node,dist):  
                if node:  
                    if node in vis:  
                        return  
                    vis.add(node)  
                    if dist==0 :  
                        ans.append(node.val)  
                        return  
                    dfs(node.parent,dist-1)  
                    dfs(node.left,dist-1)  
                    dfs(node.right,dist-1)  
            dfs(target,k)  
            return ans
```  
    This is done in O(n)  time and space.  
      
24. **Populating Next Right Pointers**  
    Given a binary tree
```cpp
    struct Node {  
      int val;  
      Node *left;  
      Node *right;  
      Node *next;  
    }
```

      
    Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.  
    Initially, all next pointers are set to NULL.  
    So I am doing level order traversal and pointing next of element to the one that is next in queue by checking if they are in same level.  
      
```python   
    def connect(self, root: 'Node') -> 'Node':  
            if not root:  
                return root  
            q=deque()  
            node=root  
            q.append((node,0))  
            while q:  
                node,lvl=q.popleft()  
                if q:  
                    nxt,lv2=q[0]  
                    if lvl==lv2 :  
                        node.next=nxt  
                if node.left:  
                    q.append((node.left,lvl+1))  
                if node.right:  
                    q.append((node.right,lvl+1))  
            return root
```
    

    **Follow-up:**  
* You may only use constant extra space.  
* The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.  
  As Recursion is fine I am doing that only.  
  I have two function findNext() it will find next of the children using current next and rec(node) the main recursive function.  
```python  
  def findNext(root):  
              while root:  
                  if root.left:  
                      return root.left  
                  if root.right:  
                      return root.right  
                  root=root.next  
              return root  
```
  We are passing parents next as argument, if there is any left in parents.next then thats our childs.next else if there is parents.next have right then thats our childs.next else we go next right.  
```python 
  def rec(node):  
              if not node:  
                  return  
              if node.right:  
                  node.right.next=findNext(node.next)  
              if node.left:  
                  if node.right:  
                      node.left.next=node.right  
                  else:  
                      node.left.next=findNext(node.next)  
              rec(node.right)  
              rec(node.left)  
```
  Now coming to the recursion part we are coming from right to left and filling the immediate next using findNext function.   
  For the right child the parents.next.left or available right child is the next and for the left child if there is a right available for parent then that is the next otherwise parents.next.child is the next.  
  The Time complexity is O(n) with a recursive stack.  
    
25. **Binary Tree Maximum Path Sum**  
    A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.  
    The **path sum** of a path is the sum of the node's values in the path.  
    Given the root of a binary tree, return *the maximum **path sum** of any **non-empty** path*.  
      
      
    For a given position in a state of recursion , standing at a node we have max path possibility is from top to right, top to left, left to right passing through that node.  
    And for the parent node it is the node of that and the maximum of any one left or right path.  
```python 
    def maxPathSum(self, root: Optional[TreeNode]) -> int:  
            max_path=float("-inf")  
            def get_max(node):  
                nonlocal max_path  
                if node is None:  
                    return 0  
                left=max(get_max(node.left),0)  
                right=max(get_max(node.right),0)  
                curr_max=node.val+left+right  
                max_path=max(max_path,curr_max)  
                return node.val+ max(left,right)  
            get_max(root)  
            return max_path 
``` 
    The time complexity upon solving the recurrence relation is O(n).  
      
26. **Maximum Sum BST**  
      
    Given a **binary tree** root, return *the maximum sum of all keys of **any** sub-tree which is also a Binary Search Tree (BST)*.  
    Assume a BST is defined as follows:  
* The left subtree of a node contains only nodes with keys **less than** the node's key.  
* The right subtree of a node contains only nodes with keys **greater than** the node's key.  
* Both the left and right subtrees must also be binary search trees.  
  On every step we have to keep track of if subtree from that is bst what's the max and min in subtree and what's the current sum going on.  
  If current node also satisfy the BST condition then update sum.  
```python 
  def maxSumBST(self, root: Optional[TreeNode]) -> int:  
          maxSum = float('-inf')  
          def rec(node):  
              nonlocal maxSum  
              if not node:  
                  return True, float('inf'), float('-inf'), 0  # isBST, min, max, sum  
              l_bst, l_min, l_max, l_sum = rec(node.left)  
              r_bst, r_min, r_max, r_sum = rec(node.right)  
               
              if l_bst and r_bst and l_max < node.val < r_min:  
                  curr = l_sum + r_sum + node.val  
                  maxSum = max(maxSum, curr)  
                  return True, min(l_min, node.val), max(r_max, node.val), curr  
              else:  
                  return False, 0, 0, 0  
    
          rec(root)  
          return max(0, maxSum)
```  
  We traverse each node exactly once so Time complexity is O(n).  
    
27. **Serialize and Deserialize Binary Tree**  
    Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.  
    Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.  
    Now I have to do a process which I will reverse.  
    So I am doing basic recursive traversal and for each value I am adding that to string and for next value I am adding ‘b’ as the breaking delimiter as length of value is not fixed and it may contain negative sign.  
    And finally for null I am appending \# and ending recursion.  
      
      
```python 
    def serialize(self, root):  
            """Encodes a tree to a single string.  
             
            :type root: TreeNode  
            :rtype: str  
            """  
            s=""  
            def rec(node):  
                nonlocal s  
                if not node:  
                    s+="#"  
                    return  
                s+="b"  
                s+=str(node.val)  
                rec(node.left)  
                rec(node.right)  
            rec(root)  
            print(s)  
            return s  
```
    Now to deserialize I am first converting it into list of values and Nulls that is \# in our case and recursively building in the same order.  
```python 
    def deserialize(self, data):  
            """Decodes your encoded data to tree.  
             
            :type data: str  
            :rtype: TreeNode  
            """  
      
            tree = []  
            temp = ""  
            i = 0  
            while i < len(data):  
                if data[i] == '#':  
                    tree.append('#')  
                    temp = ""  
                    i += 1  
                elif data[i] == 'b':  
                    i += 1  
                    temp = ""  
                    while i < len(data) and (data[i].isdigit() or data[i] == '-'):  
                        temp += data[i]  
                        i += 1  
                    if temp:    
                        tree.append(int(temp))  
                else:  
                    # Skip any unexpected characters  
                    i += 1  
            i=-1  
            def rec(s):  
                nonlocal i  
                i+=1  
                node=TreeNode()  
                if i>=len(s):  
                    return None  
                if s[i]=='#':  
                    return None  
                 
                node.val=s[i]  
                 
                node.left=rec(s)  
                node.right=rec(s)  
                return node  
            return rec(tree) 
``` 
    Now this is building the Tree again from the string.  
    My String and my tree list.
  
    b1b2\#\#b3b4\#\#b5\#\#  
    \[1, 2, '\#', '\#', 3, 4, '\#', '\#', 5, '\#', '\#'\]  
      
    Doing it in almost Linear Time.  
      
28. **Tree Boundary Traversal**  
    Given a Binary Tree, find its Boundary Traversal. The traversal should be in the following order: 

1. Left Boundary: This includes all the nodes on the path from the root to the leftmost leaf node. You must prefer the left child over the right child when traversing. Do not include leaf nodes in this section.  
2. Leaf Nodes: All leaf nodes, in left-to-right order, that are not part of the left or right boundary.

3. Reverse Right Boundary: This includes all the nodes on the path from the rightmost leaf node to the root, traversed in reverse order. You must prefer the right child over the left child when traversing. Do not include the root in this section if it was already included in the left boundary.  
   Note: If the root doesn't have a left subtree or right subtree, then the root itself is the left or right boundary.   
     
     
   First Find all left boundary values that are not leaf then leaf values followed by the reversed right boundary values and finally combining  them to get the answer.  
```python 
   def boundaryTraversal(self, root):  
           def isleaf(node):  
               if not node.left and not node.right:  
                   return True  
               return False  
           ans=[]  
           def left(node):  
               node=node.left  
               while node:  
                   if not isleaf(node):  
                       ans.append(node.data)  
                   if node.left:  
                       node=node.left  
                   else:  
                       node=node.right  
           def leaf(node):  
               if node:  
                   if isleaf(node):  
                       ans.append(node.data)  
                       return  
                   leaf(node.left)  
                   leaf(node.right)  
           def right(node):  
               node=node.right  
               temp=[]  
               while node:  
                   if not isleaf(node):  
                       temp.append(node.data)  
                   if node.right:  
                       node=node.right  
                   else:  
                       node=node.left  
               return temp[::-1]  
           if not isleaf(root):  
               ans.append(root.data)  
           left(root)  
           leaf(root)  
           ans+=right(root)  
           return ans
```  
   Done with worst case time complexity of O(n)   
   