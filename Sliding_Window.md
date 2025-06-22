Report of Target 2025

# SLIDING WINDOW

1. **Longest Substring Without Repeating Char.**  
   Given a string s, find the length of the **longest** **substring** without duplicate characters.  
     
So I am initializing the window with left as 0 and right as 0;In this case I am using *l* as left and *i*  as right in loop.  
I am keeping track of the last seen of a character in hashmap.   
Whenever its in the range of left to right I am reinitializing the left to lastseen+1.  
So I am skipping the redundant subarrays.  
And at each step I am taking max function on ans and right-left+1 that is window size.  
```cpp     
   int lengthOfLongestSubstring(string s) {  
           int ans=0;  
           int l=0;  
           map<char,int>mp;  
           for (int i=0;i<s.size();i++){  
               if(mp.find(s[i])!=mp.end() and mp[s[i]]>=l){  
                   l=mp[s[i]]+1;  
               }  
               ans=max(ans,i-l+1);  
               mp[s[i]]=i;  
           }  
           return ans;  
       }  
```
If I consider fetching from hash as constant time operation then time complexity of this code is O(n).  
     
2. **Sliding Window Maximum**  
   You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.  
   Return *the max sliding window*.  
     
     
For each window I am keeping track of the window maximum in the heapified multiset whenever a element is removed from window I am deleting it from the heap of multiset.   
```cpp     
   vector<int> maxSlidingWindow(vector<int>& nums, int k) {  
           int left=0;  
           int right=0;  
           multiset<int>ms;  
           vector<int>ans;  
           while(right<nums.size()){  
               ms.insert(nums[right]);  
               if(right-left+1==k){  
                   int maxi=(*(ms.rbegin()));  
                   ans.push_back(maxi);  
                   ms.erase(ms.find(nums[left]));  
                   left++;  
               }  
               right++;  
           }  
           return ans;  
       }  
```
The time complexity of the code is O(nlogn). 
In python we can use SortedList() to replicate the Multiset Behavior of LogN insertion and deletion.
```python
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans=[]
        multiset=SortedList()
        left,right=0,0
        while right<len(nums):
            multiset.add(nums[right])
            if right-left+1==k:
                ans.append(multiset[-1])
                multiset.remove(nums[left])
                left+=1
            right+=1
        return ans
```

Also Another approach is this problem is done by double ended queue.
```python
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()
        ans = []
        #Q front is the max elements index
        for i in range(len(nums)):
            #if max elements index is less than i - k means its outside window
            while q and q[0] < i - k + 1:
                q.popleft()
            #we are maintaining monotonic decreasing queue so removing redundant 
            while q and nums[q[-1]] < nums[i]:
                q.pop()
            q.append(i)
            # we have reached at first window now onwards Q's 1st element is ans
            if i >= k - 1 :
                ans.append(nums[q[0]])
        return ans
```
This has a linear Time Complexity.
     
3. **Sliding Window Median**  
   The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle values.  
* For examples, if arr \= \[2,3,4\], the median is 3\.  
* For examples, if arr \= \[1,2,3,4\], the median is (2 \+ 3\) / 2 \= 2.5.  
  You are given an integer array nums and an integer k. There is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.  
  Return *the median array for each window in the original array*. Answers within 10\-5 of the actual value will be accepted.


So I am keeping an multiset to store the window and an iterator to point in the mid of the sorted datastream of window present in multiset. While adding i check if the number I am adding is less than the mid then I shift the pointer to left as left becomes more populated. And while removing I am checking the number is left or not if yes then again adjusting the mid pointer to right.  

```cpp
  vector<double> medianSlidingWindow(vector<int>& nums, int k) {
          multiset<int>ms(nums.begin(),nums.begin()+k);
          auto mid=next(ms.begin(),k/2);
          vector<double>ans;
          int right=k;
          int left=0;
          while(right<nums.size()){
              ans.push_back((double(*mid)+*prev(mid,1-k%2))/2);
              ms.insert(nums[right]);
              if(nums[right]<*mid){
                mid--;
              }
              if(nums[left]<=*mid){
                mid++;
              }
              ms.erase(ms.find(nums[left]));
              left++;
              right++;
          }
          ans.push_back((double(*mid)+*prev(mid,1-k%2))/2);
          return ans;
      }
```
Overall cost of time is O(nlogn). And O(n) space for multiset implementation.


4. **Grumpy Bookstore Owner**  
   There is a bookstore owner that has a store open for n minutes. You are given an integer array customers of length n where customers\[i\] is the number of the customers that enter the store at the start of the ith minute and all those customers leave after the end of that minute.  
   During certain minutes, the bookstore owner is grumpy. You are given a binary array grumpy where grumpy\[i\] is 1 if the bookstore owner is grumpy during the ith minute, and is 0 otherwise.  
   When the bookstore owner is grumpy, the customers entering during that minute are not **satisfied**. Otherwise, they are satisfied.  
   The bookstore owner knows a secret technique to remain **not grumpy** for minutes consecutive minutes, but this technique can only be used **once**.  
   Return the **maximum** number of customers that can be *satisfied* throughout the day.

   

So first of all I am going to convert the grumpy array by complement for easy calculation then I will store all sum within minutes window possible without secret trick in a hash map and overall possible satisfaction without secret trick.  
Now for trick I will again go through with same windows and will find maximum possible ans if we consider that window as secret trick window;  
Mathematical formulation will be:  
Satisfaction{i,j}= overallSatisfaction \- windowSatisfaction \+ secretTrickSatisfaction  
For given window frame between i and j.  
Now Final Maximized Satisfaction \= max{ Satisfaction } across all {i,j}.  
```cpp     
   int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {  
           int right = 0;  
           int left = 0;  
           int sum = 0;  
           int overall = 0;  
           int ans=0;  
           for(int i=0;i<grumpy.size();i++){  
               grumpy[i]=!grumpy[i];  
           }  
           map<pair<int,int>,int>mp;  
           while(right<grumpy.size()){  
               sum+=customers[right]*grumpy[right];  
               overall+=customers[right]*grumpy[right];  
               if(right-left+1==minutes){  
                   mp[{left,right}]=sum;  
                   sum-=customers[left]*grumpy[left];  
                   left++;  
               }  
               right++;  
           }  
           left=0;  
           right=0;  
           sum=0;  
           while(right<customers.size()){  
               sum+=customers[right];  
               if(right-left+1==minutes){  
                   int curr=overall-mp[{left,right}]+sum;  
                   ans=max(ans,curr);  
                   sum-=customers[left];  
                   left++;  
               }  
               right++;  
           }  
           return ans;  
       }
```
The time complexity is based on the Hashing Data structure. For unordered one it will be O(n).

5. **Minimum Window Substring**  
   Given two strings s and t of lengths m and n respectively, return *the **minimum window*** ***substring** of* s *such that every character in* t *(**including duplicates**) is included in the window*. If there is no such substring, return *the empty string* "".  
   The test cases will be generated such that the answer is **unique**.


So I will maintain a fixed map of t and a variable hashmap for windows .

Now I will start with window 0 and keep on increasing the values in window hashmap when it matches the same frequency that is required. I will update the answer and will search for further smaller windows by moving the left pointer of the window. And keep on updating the window Answer of length and for each minimum I will update the left and right answer that I will later use to return the answer.

   

   
```python
   def minWindow(self, s: str, t: str) -> str:
           if len(s)==0 or len(t)==0:
               return ""
           t_count=Counter(t)
           req=len(t_count)
           left,right=0,0
           formed=0
           windowCount=defaultdict(int)
           ans=(-1,0,0)
           while right<len(s):
               windowCount[s[right]]+=1
               if s[right] in t_count.keys() and windowCount[s[right]]==t_count[s[right]]:
                   formed+=1
               while left<=right and formed==req:
                   c=s[left]
                   a,l,r=ans
                   if a==-1 or right-left+1<a:
                       ans=(right-left+1,left,right)
                   windowCount[c]-=1
                   if c in t_count and windowCount[c]<t_count[c]:
                       formed-=1
                   left+=1
               right+=1
           a,l,r=ans
           return "" if a==-1 else s[l:r+1]
```
The time complexity is O(m+n).