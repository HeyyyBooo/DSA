Report of Target 2025

# STACK AND QUEUE

1. **Implement Stack using Queues**  
     
So I am here using two queues q1 and q2. Where q1 is primary and q2 is used for fetching the top of the simulated stack.  
```cpp      
   class MyStack {  
   private:  
   queue<int>q1;  
   queue<int>q2;  
   public:  
       MyStack() {  
           q1={};  
           q2={};  
       }  
        
       void push(int x) {  
           q1.push(x);  
       }  
        
       int pop() {  
           while(q1.size()>1){  
               int x=q1.front();  
               q1.pop();  
               q2.push(x);  
           }  
           int ans=q1.front();  
           q1.pop();  
           while(!q2.empty()){  
               int x=q2.front();  
               q2.pop();  
               q1.push(x);  
           }  
           return ans;  
       }  
        
       int top() {  
           while(q1.size()>1){  
               int x=q1.front();  
               q1.pop();  
               q2.push(x);  
           }  
           int ans=q1.front();  
           q1.pop();  
           q2.push(ans);  
           while(!q2.empty()){  
               int x=q2.front();  
               q2.pop();  
               q1.push(x);  
           }  
           return ans;  
       }  
        
       bool empty() {  
           return q1.empty();  
       }  
   };  
```
The fetching time complexity of simulated stack is O(n)  
     
2. **Implement Queue using Stack**  
     
Exactly similar approach as before just using s1 and s2 stack now.  
     
```cpp 
   class MyQueue {  
   private:  
       stack<int> s1;  
       stack<int> s2;  
     
   public:  
       MyQueue() {  
           s1 = {};  
           s2 = {};  
       }  
     
       void push(int x) { s1.push(x); }  
     
       int pop() {  
           while (s1.size() > 1) {  
               int x = s1.top();  
               s1.pop();  
               s2.push(x);  
           }  
           int ans = s1.top();  
           s1.pop();  
           while (!s2.empty()) {  
               int x = s2.top();  
               s2.pop();  
               s1.push(x);  
           }  
           return ans;  
       }  
     
       int peek() {  
           while (s1.size() > 1) {  
               int x = s1.top();  
               s1.pop();  
               s2.push(x);  
           }  
           int ans = s1.top();  
           s1.pop();  
           s2.push(ans);  
           while (!s2.empty()) {  
               int x = s2.top();  
               s2.pop();  
               s1.push(x);  
           }  
           return ans;  
       }  
     
       bool empty() { return s1.empty(); }  
   };  
```  
     
3. **Min Stack**  
   Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.  
   Implement the MinStack class:  
* MinStack() initializes the stack object.  
* void push(int val) pushes the element val onto the stack.  
* void pop() removes the element on the top of the stack.  
* int top() gets the top element of the stack.  
* int getMin() retrieves the minimum element in the stack.  
You must implement a solution with O(1) time complexity for each function.


    
I will use two stacks, one for normal FILO and one for maintaining the minimum present. Whenever there is any element smaller than the top of the min tracker I will update the top.  
```cpp 
  class MinStack {  
  public:  
      stack<int> s;  
      stack<int> m;  
      MinStack() {  
          s={};  
          m={};  
      }  
      void push(int val) {  
          s.push(val);  
          if (m.empty()) {  
              m.push(val);  
          } else {  
              if (val <= m.top()) {  
                  m.push(val);  
              }  
          }  
      }  
    
      void pop() {  
          if (s.top() == m.top()) {  
              s.pop();  
              m.pop();  
          } else {  
              s.pop();  
          }  
      }  
    
      int top() { return s.top(); }  
    
      int getMin() { return m.top(); }  
  };
```

4. **Valid Parentheses**  
For open parentheses I will push it in the stack and for closing one I will check  whether its corresponding opening parentheses is present at the top of stack or not. If not then false . otherwise pop it.  
After all simulation if the stack is empty then its a valid parentheses.  
```cpp   
   bool isValid(string s) {  
           stack<char> st;  
           for (char c : s) {  
               if (c == '(' || c == '{' || c == '[') {  
                   st.push(c);  
               } else {  
                   if (st.empty() ||  
                       (c == ')' && st.top() != '(') ||  
                       (c == '}' && st.top() != '{') ||  
                       (c == ']' && st.top() != '[')) {  
                       return false;  
                   }  
                   st.pop();  
               }  
           }  
           return st.empty();  
       }  
```     
5. **Largest Rectangle in Histogram**  
   Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return *the area of the largest rectangle in the histogram*.  
     
For each bar at index `i`, we want to know how far left and right we can extend a rectangle of height `heights[i]` before we hit a bar **strictly shorter** than it. If those boundaries are at indices `L` and `R`, then the maximum width for bar `i` is width \= R − L − 1\.  
     
`nse[i]` \= index of the first bar to the right of i that is strictly shorter than heights\[i\],  
or n    if none exists.  
     
We do a mirror-image pass from right to left to find, for each `i`, the index of the first bar **strictly shorter** on the left side. We store this in `pse[i]`, defaulting to –1 if none exists  
     
     
Now our desired area is area\_i \= heights\[i\] \* (nse\[i\] \- pse\[i\] \- 1\)  
ans \= max(ans, area\_i) for all i.  
```cpp    
   int largestRectangleArea(vector<int>& heights) {  
           stack<int>st;  
           int n=heights.size();  
           vector<int>pse(n,-1);  
           vector<int>nse(n,n);  
           for(int i=0;i<n;i++){  
               while(!st.empty() and heights[st.top()]>heights[i]){  
                   nse[st.top()]=i;  
                   st.pop();  
               }  
               st.push(i);  
           }  
           st={};  
           for(int i=n-1;i>=0;i--){  
               while(!st.empty() and heights[st.top()]>heights[i]){  
                   pse[st.top()]=i;  
                   st.pop();  
               }  
               st.push(i);  
           }  
           int ans=0;  
           for(int i=0;i<n;i++){  
               ans=max(ans,(nse[i]-pse[i]-1)*heights[i]);  
           }  
           return ans;  
       }  
```   
Both Space and Time Complexity of the code is O(n)  
     
6. **Online Stock Span**  
   Design an algorithm that collects daily price quotes for some stock and returns **the span** of that stock's price for the current day.  
   The **span** of the stock's price in one day is the maximum number of consecutive days (starting from that day and going backward) for which the stock price was less than or equal to the price of that day.  
* For example, if the prices of the stock in the last four days is \[7,2,1,2\] and the price of the stock today is 2, then the span of today is 4 because starting from today, the price of the stock was less than or equal 2 for 4 consecutive days.  
* Also, if the prices of the stock in the last four days is \[7,34,1,2\] and the price of the stock today is 8, then the span of today is 3 because starting from today, the price of the stock was less than or equal 8 for 3 consecutive days.  
   
Implementation 1 : Using Vector  
```cpp   	
class StockSpanner {
  private:  
      vector<int>v;  
  public:  
      StockSpanner() {  
          v={};  
      }  
       
      int next(int price) {  
          v.push_back(price);  
          int i=v.size()-1;  
          int ans=0;  
          while(i>=0 and v[i]<=price){  
              ans++;  
              i--;  
               
          }  
          return ans;  
      }  
  };
```
    
    
    
Implementation : Using Stack  
```cpp 
class StockSpanner {

  private:  
      stack<pair<int, int>> s;  
  public:  
      StockSpanner() {  
          s={};  
      }  
      int next(int price) {  
          int res = 1;  
          while (!s.empty() && s.top().first <= price) {  
              res += s.top().second;  
              s.pop();  
          }  
          s.push({price, res});  
          return res;  
      }  
  };
```
  Now the implementation 2 that is for stack is important here it uses the concept of LIFO and we push the recent most value , span pair in stack and check weather its less than current price then pop it as its already considered. And push the new price , span pair into the stack.  
  The complexity here reduces as compared to the first implementation using vector arrays.  
    
7. **Evaluate Reverse Polish Notation**  
   You are given an array of strings tokens that represents an arithmetic expression in a [Reverse Polish Notation](http://en.wikipedia.org/wiki/Reverse_Polish_notation).  
Evaluate the expression. Return *an integer that represents the value of the expression*.  

```python
   class Solution:  
       def resolves(self, a, b, Operator):  
           if Operator == '+':  
               return a + b  
           elif Operator == '-':  
               return a - b  
           elif Operator == '*':  
               return a * b  
           return int(a / b)  
     
       def evalRPN(self, tokens):  
           stack = []  
           for token in tokens:  
               if len(token)==1 and not token.isnumeric():  
                   integer2 = stack.pop()      
                   integer1 = stack.pop()  
                   operator = token  
                   resolved_ans = self.resolves(integer1, integer2, operator)  
                   stack.append(resolved_ans)  
               else:  
                   stack.append(int(token))  
           return stack.pop()
```  
8. **Next Greater Element II**  
   Given a circular integer array nums (i.e., the next element of nums\[nums.length \- 1\] is nums\[0\]), return *the **next greater number** for every element in* nums.  
   The **next greater number** of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, return \-1 for this number.  
     
     
Again Implementation of Next Greater but this time using circular array.  
For all elements that are before i and less than nums\[i\] are present in stack will have nums\[i\] as their NGE.  
Else just put that element in stack.  
And to simulate the circular process I will do a Copy again of the nums or indexing copy and make list of size n+n.  
Now accessing of element will be used using modulo of n.  
```python  
def nextGreaterElements(self, nums: List[int]) -> List[int]:  
           n=len(nums)  
           nge=[-1 for _ in range(n+n)]  
           stack=[]  
           for i in range(n+n):  
               while len(stack)!=0 and nums[(stack[-1])%n]<nums[i%n]:  
                   nge[stack[-1]]=nums[i%n]  
                   stack.pop()  
               stack.append(i)  
           return nge[:n]  
```
It has a Time complexity of O(n).  
     
9. **Asteroid Collision**  
   We are given an array asteroids of integers representing asteroids in a row. The indices of the asteriod in the array represent their relative position in space.  
   For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.  
   Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.  
  

I am just simulating the process using stack.  
```python
   def asteroidCollision(self, asteroids: List[int]) -> List[int]:  
           stack=[]  
           for a in asteroids:  
               while stack and stack[-1]>0 and a<0:  
                   b=stack.pop()  
                   if abs(a)>abs(b):  
                       continue  
                   elif abs(b)>abs(a):  
                       stack.append(b)  
                       break  
                   else:  
                       break    
               else:  
                   stack.append(a)  
           return stack  
```

10. **132 Pattern**  
    Given an array of n integers nums, a **132 pattern** is a subsequence of three integers nums\[i\], nums\[j\] and nums\[k\] such that i \< j \< k and nums\[i\] \< nums\[k\] \< nums\[j\].  
    Return true *if there is a **132 pattern** in* nums*, otherwise, return* false*.*  
      
The stack approach of this problem is very good. Its tricky to understand that how I will implement the monotonic stack.  
    
```python
    class Solution:  
        def find132pattern(self, nums: List[int]) -> bool:  
            if len(nums)<3:  
                return False  
            stack=[]  
            curMin=nums[0]  
            for n in nums[1:]:  
                while stack and n>=stack[-1][0]:  
                    stack.pop()  
                if stack and n>stack[-1][1]:  
                    return True  
                stack.append([n,curMin])  
                curMin=min(curMin,n)  
            return False  
``` 
CurrMin stores the current minimum till index i. Now We will pop all the elements that are less than the current element that we traversed .  
      
So in stack I have stored the element and the minimum that is present before that element .  
Now ‘i’ is our last candidate element in ’132’ pattern that is ‘i’ is our ’ 2’ and we will pop until we get our ‘3’ and when we get ‘3’ we will check if its minimum present in left is less than our ith element. If its there then we found our element ‘1’.  
      
Otherwise we will continue the process by pushing values in stack and updating minimums.  
      
The Time complexity of this algorithm is O(n).
