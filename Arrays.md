Report of Target 2025

# ARRAYS 

1. Two Sum  
     
   Given an array of integers nums and an integer target, return *indices of the two numbers such that they add up to target*.  
   You may assume that each input would have ***exactly*** **one solution**, and you may not use the *same* element twice.  
   You can return the answer in any order.  
     
So here comes my brute force approach , we will Traverse the array in double for-loop with keeping track of ‘i’ and ‘j’ indices . and will check for every ‘i’ and ‘j’ that sum of nums\[i\] and nums\[j\] adds up to our target value.  
```cpp
   vector<int> BruteForce(vector<int>& nums,int target){  
           for (int i=0;i<nums.size();i++){  
               for(int j=0;j<nums.size();j++){  
                   if(i!=j){  
                       if(nums[i]+nums[j]==target){  
                           return {i,j};  
                       }  
                   }  
               }  
           }  
           return {};  
       }  
   
```
But this double for-loop is causing a time complexity of O(n2) but I can further optimize by using a hash data structure.  
As there is only one solution so I will keep track of every past value and for every new value i will check for its sum complement. That is 

CurrentValue \+ Complement(CurrentValue)= Target  & Complement(CurrentValue) ɛ HashSet  
here Complement(CurrentValue)=Target \- Current Value

If there exist such value in hash then thats our Solution.  
```cpp
 vector<int> Optimal(vector<int>&nums,int target){  
        map<int,int>mp;  
        //map<value,index>  
        for (int i=0;i<nums.size();i++){  
            if(mp.find(target-nums[i])!=mp.end()){  
                return {i,mp[target-nums[i]]};  
            }  
            mp[nums[i]]=i;  
        }  
        return {};  
    }  
So the new Time Complexity is  O(nlogn).
```
2. **3Sum**  
   Given an integer array nums, return all the triplets \[nums\[i\], nums\[j\], nums\[k\]\] such that i \!= j, i \!= k, and j \!= k, and nums\[i\] \+ nums\[j\] \+ nums\[k\] \== 0\.  
   Notice that the solution set must not contain duplicate triplets.  
```cpp
   vector<vector<int>> BruteForce(vector<int>&nums){  
           vector<vector<int>>ans;  
           set<vector<int>>temp;  
           for(int i=0;i<nums.size();i++){  
               for(int j=0;j<nums.size();j++){  
                   for(int k=0;k<nums.size();k++){  
                       if(i!=j and j!=k){  
                           if(nums[i]+nums[j]+nums[k]==0){  
                               temp.insert({nums[i],nums[j],nums[k]});  
                           }  
                       }  
                   }  
               }  
           }  
           for (vector<int>v:temp){  
               ans.push_back(v);  
           }  
           return ans;  
       }
```
But This will lead to TLE , Not only TLE also Wrong answer for the duplicate values. So we need to sort it first then find the triplets.

I will fix one element and find its complement pair of elements .
```cpp
	vector<vector<int>>Optimal(vector<int>&nums){  
        sort(nums.begin(),nums.end());  
        int target=0;  
        set<vector<int>>s;  
        vector<vector<int>>ans;  
        for(int i=0;i<nums.size();i++){  
            int l=i+1;  
            int r=nums.size()-1;  
            while(l<r){  
                int sum=nums[i]+nums[l]+nums[r];  
                if(sum==target){  
                    s.insert({nums[i],nums[l],nums[r]});  
                    l++;  
                    r--;  
                }  
                else if(sum<target){  
                    l++;  
                }  
                else{  
                    r--;  
                }  
            }  
        }  
        for(vector<int>v:s){  
            ans.push_back(v);  
        }  
        return ans;  
    }
```
So using Two pointers in a sorted array I have written this optimal solution with the new Time Complexity of O(nlogn \+ n2)  ie. O(n2 )   
As Problem statement states that sum should be 0 . But I have this code for generalized ‘target’ Based on Input and constraints it can be changed.

3. **4Sum**  
   Given an array nums of n integers, return *an array of all the **unique** quadruplets* \[nums\[a\], nums\[b\], nums\[c\], nums\[d\]\] such that:  
* 0 \<= a, b, c, d \< n  
* a, b, c, and d are **distinct**.  
* nums\[a\] \+ nums\[b\] \+ nums\[c\] \+ nums\[d\] \== target  
  You may return the answer in **any order**.  
  As Clearly I can see  
* 1 \<= nums.length \<= 200  
  So Brute Force approach will cause a time complexity of O(n4) so clearly it will exceed the Time Limit.  
    
So what I am thinking is to fix one element and call the previous 3Sum Approach for finding the remaining triplets of the quadruplets .  
```cpp  
  vector<vector<int>> Tsum(vector<int>& num, int start, int end, long long target) {  
          vector<long long> nums;  
          for (int i = start; i <= end; i++) {  
              nums.push_back(num[i]);  
          }  
          set<vector<int>> s;  
          vector<vector<int>> ans;  
          for (int i = 0; i < nums.size(); i++) {  
              int l = i + 1;  
              int r = nums.size() - 1;  
              while (l < r) {  
                  long long sum = nums[i] + nums[l] + nums[r];  
                  if (sum == target) {  
                      s.insert({static_cast<int>(nums[i]),  
                                static_cast<int>(nums[l]),  
                                static_cast<int>(nums[r])});  
                      l++;  
                      r--;  
                  } else if (sum < target) {  
                      l++;  
                  } else {  
                      r--;  
                  }  
              }  
          }  
          for (vector<int> v : s) {  
              ans.push_back(v);  
          }  
          return ans;  
      }  
    
      vector<vector<int>> fourSum(vector<int>& nums, int target) {  
          if (nums.size() < 4) {  
              return {};  
          }  
          sort(nums.begin(), nums.end());  
          set<vector<int>> vec;  
          for (int i = 0; i < nums.size() - 3; i++) {  
              vector<vector<int>> temp =  
                  Tsum(nums, i + 1, nums.size() - 1, target - nums[i]);  
              for (vector<int> v : temp) {  
                  v.push_back(nums[i]);  
                  vec.insert(v);  
              }  
          }  
          vector<vector<int>> ans;  
          for (vector<int> v : vec) {  
              ans.push_back(v);  
          }  
          return ans;  
      }  
```
As the sum exceeds the integer limit thats why I changed the data type to ‘long long’ so that it can takes higher values and i have undone the process for output using static\_Cast\<int\>.  
    
So its basically n times 3sum Problem so the time complexity is O(n3) 


So for K-Sum we are just recursively calling it for (K-1)sum so for generalized K-Sum its time complexity is O(nk-1)  and for k=2 its O(nlogn) that is nothing but sorting complexity.
```cpp
vector<vector<int>> kSum(vector<int>& nums, long long target, int start,  
                             int k) {  
        vector<vector<int>> res;  
        // If we have run out of numbers to add, return res.  
        if (start == nums.size()) {  
            return res;  
        }  
        // There are k remaining values to add to the sum. The  
        // average of these values is at least target / k.  
        long long average_value = target / k;  
        // We cannot obtain a sum of target if the smallest value  
        // in nums is greater than target / k or if the largest  
        // value in nums is smaller than target / k.  
        if (nums[start] > average_value || average_value > nums.back()) {  
            return res;  
        }

        if (k == 2) {  
            return twoSum(nums, target, start);  
        }  
        for (int i = start; i < nums.size(); ++i) {  
            if (i == start || nums[i - 1] != nums[i]) {  
                for (vector<int>& subset:  
                  kSum(nums,static_cast<long long>(target)-nums[i],i+1,k-1)){  
                    res.push_back({nums[i]});  
                res.back().insert(end(res.back()), begin(subset),end(subset));  
                }  
            }  
        }

        return res;  
    }  
```	  
This is Official K-sum Solution . All it ends on Two Sum Implementation Only.

4. **Remove Duplicates From Sorted Array**  
   Given an integer array nums sorted in **non-decreasing order**, remove the duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) such that each unique element appears only **once**. The **relative order** of the elements should be kept the **same**. Then return *the number of unique elements in* nums.  
   Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:  
* Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially. The remaining elements of nums are not important as well as the size of nums.  
* Return k.  
    
So base case is like there is at least one unique element so I will initialize  the value of k with 1\.  
    
And at any ith state lets say I have K unique elements till now then I will  check if ith element is not equal to the i-1th element . that means I have found a new unique element so i will increment the k counter and will copy it to the k-1th spot at the array.  
    
So from beginning to kth position I have replaced all the elements with the unique k elements


```cpp
	int removeDuplicates(vector<int>& nums) {  
        int k=1;  
        for(int i = 1; i < nums.size(); i++){  
            if(nums[i] != nums[i - 1]){  
                nums[k] = nums[i];  
                k++;  
            }  
        }  
        return k;  
    }  
```
Simple code with simple time complexity of O(n) .

5. **Next Permutation**  
     
   A **permutation** of an array of integers is an arrangement of its members into a sequence or linear order.  
   Given an array of integers nums, *find the next permutation of* nums. The replacement must be [**in place**](http://en.wikipedia.org/wiki/In-place_algorithm) and use only constant extra memory.  

```cpp
   void nextPermutation(vector<int>& nums) {  
           next_permutation(nums.begin(),nums.end());  
       }
```  
     
This is a lazy solution that is directly importing  next\_permutation function from C++ STL  
     
But Now coming to the serious part , I will now try to do it using normal array implementation.  
So for  permutation I have to keep in ,mind that which elements are there in my set and will try to find next lexicographical version of the permutation.

The Brute Force approach is to compare with all the permutation but it will cost me a time complexity of O(n\!) 

   

So I tried to much to find the pattern but I quit so I studied how next permutation works in mathematics.

   

So we have to find the exact point from the end where it start decreasing  that is at ith stage  i-1th is the dip point we will replace it with the just next greater element present in the right side of the array. Lets say its jth element and so just swap it and reverse the right part.

```cpp
   void nextPermutation(vector<int>& nums) {
           int i = nums.size() - 2;
           while (i >= 0 && nums[i + 1] <= nums[i]) {
               i--;
           }
           if (i >= 0) {
               int j = nums.size() - 1;
               while (nums[j] <= nums[i]) {
                   j--;
               }
               swap(nums, i, j);
           }
           reverse(nums.begin() + i + 1, nums.end());
       }

``` 

Time Complexity of this copied 😂 Algorithm is O(n) .

   

   

6. **Trapping Rain Water**  
     
   Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.  
     

First I will store the highest wall present in the left of the current position , and Highest wall present in right of the current position .  
     
To store these values i will make two arrays of same size to store respective heights of left and right wall.  
     
Then the water stored is nothing but min{max(leftWalls),max(rightWalls)}-currHeight  
   
```cpp
int trap(vector<int>& height) {  
        vector<int>leftWall(height.size());  
        vector<int>rightWall(height.size());  
        leftWall[0]=height[0];  
        rightWall[height.size()-1]=height[height.size()-1];  
        for(int i=1;i<height.size();i++){  
            leftWall[i]=max(height[i],leftWall[i-1]);  
        }  
        for(int i=height.size()-2;i>=0;i--){  
            rightWall[i]=max(height[i],rightWall[i+1]);  
        }  
        int water=0;  
        for(int i=0;i<height.size();i++){  
            water+=min(leftWall[i],rightWall[i])-height[i];  
        }  
        return water;  
    }
```
So the overall time complexity of above algorithm is O(n).

7. **Rotate Image**  
     
   You are given an n x n 2D matrix representing an image, rotate the image by **90** degrees (clockwise).  
   You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.  
     
As Mentioned above ‘DO NOT’ allocate a ‘2D Matrix’ so I used a 1D array 😀  
     
```cpp 	  
    void rotate(vector<vector<int>>& matrix) {  
           int n=matrix.size();  
           vector<int>a;  
           for(int j=0;j<n;j++){  
               for(int i=n-1;i>=0;i--){  
                   a.push_back(matrix[i][j]); } }  
           int index=0;  
           for(int i=0;i<n;i++){  
               for(int j=0;j<n;j++){  
                   matrix[i][j]=a[index];index++;  
               }  
           }   
   } 
``` 
So  now funny things apart its time to solve it without allocating anything. I will start it with finding a pattern then I will try to execute it.  
     
So from my observation i noted that for n sized matrix with indices 0 to e (e=n-1) we have to do a fourway swap:  
     
(i,j)  to ( j,e-i ) and   (j,e-i) to (e-i,e-j) and (e-i),(e-j) to (e \-j,i) and to (i,j)  

```cpp
    void rotate(vector<vector<int>>& matrix) {  
           int n=matrix.size();  
           if(n==1){  
               return;  
           }  
           int e=n-1;  
           int i,j;  
           for(i=0;i<n/2;i++){  
               for (j=0;j<n-n/2;j++){  
                   int temp=matrix[e-j][i];  
                   matrix[e-j][i]=matrix[e-i][e-j];  
                   matrix[e-i][e-j]=matrix[j][e-i];  
                   matrix[j][e-i]=matrix[i][j];  
                   matrix[i][j]=temp;  
               }  
           }  
            
       }
```
Time Complexity of the above code is O(n2)  
And another famous solution of the above problem is to take transpose and reverse it.
```cpp
void rotate(vector<vector<int>>& matrix) {  
        int row = matrix.size();  
        for(int i=0;i<row; i++){  
            for(int j=0; j<=i;j++){  
                swap(matrix[i][j], matrix[j][i]);  
            }  
        }  
        for(int i=0;i<row;i++){  
            reverse(matrix[i].begin(), matrix[i].end());  
        }  
    }
```
Again the time complexity is O(n2)

8. **Merge Intervals**  
     
   Given an array of intervals where intervals\[i\] \= \[starti, endi\], merge all overlapping intervals, and return *an array of the non-overlapping intervals that cover all the intervals in the input*.  
     
So as the question states that only merge the overlapping intervals , I am going to sort the intervals array. And check for new start and end point with ,minimal start and maximum end for overlapping intervals.  


```cpp
   vector<vector<int>> merge(vector<vector<int>>& intervals) {  
           vector<vector<int>>ans;  
           sort(intervals.begin(),intervals.end());  
           int start=intervals[0][0];  
           int end=intervals[0][1];  
           for (int i=1;i<intervals.size();i++){  
               if(end>=intervals[i][0]){  
                   end=max(intervals[i][1],end);  
               }  
               else{  
                   ans.push_back({start,end});  
                   start=intervals[i][0];  
                   end=intervals[i][1];  
               }  
           }  
           ans.push_back({start,end});  
           return ans;  
       }  
```  
I am initializing the start and end with the initial lowest interval and then updating end with the max ending interval till breakage between intervals is found. Then adding it to new list of intervals then updating the start and end  with the next interval start and end.  
     
Time Complexity of the above code is O(nlogn) because of sorting and the main merging part is linear.  
     
     
     
9. **Sort colors**    
   Given an array nums with n objects colored red, white, or blue, sort them [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) so that objects of the same color are adjacent, with the colors in the order red, white, and blue.  
   We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively

As there is only three colors so i will fix two pointer for 0 and 2 respectively and keep 1 as it is.  
Whenever a 0 comes i will push it in left most with left pointer and for 2 i will push it right using right pointer.

```cpp
void sortColors(vector<int>& nums) {  
    int left = 0;  
    int right = nums.size() - 1;  
    int curr = 0;

    while (curr <= right) {  
        if (nums[curr] == 0) {  
            swap(nums[curr], nums[left]);  
            left++;  
            curr++;  // move forward because nums[curr] is now in the correct place  
        } else if (nums[curr] == 2) {  
            swap(nums[curr], nums[right]);  
            right--;  // don't increment curr because the swapped element still needs to be processed  
        } else {  
            curr++;  // move forward for the element 1, as it's already in the correct place  
        }  
    }  
}
```
The time complexity of the above algorithm is O(n) .

10. **Merge Sorted Arrays**  
      
    You are given two integer arrays nums1 and nums2, sorted in **non-decreasing order**, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.  
    **Merge** nums1 and nums2 into a single array sorted in **non-decreasing order**.  
    The final sorted array should not be returned by the function, but instead be *stored inside the array* nums1. To accommodate this, nums1 has a length of m \+ n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.  
      
    
As the naive approach is very simple is so I am directly going for a linear complexity approach so lets think of something that can simply do it in a single pass, I am thinking of a two pointer   approach . Everyone thinks of left to right but our space is in right side so we will move right to left in decreasing order. So our main ‘ptr’ is writing index starting with (m+n)-1 and will move till 0\. And for comparison  I am using two moving pointer a and b for nums1 and nums2 respectively.  

```cpp  
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {  
            int a=m-1;  
            int b=n-1;  
            int ptr=m+n-1;  
            while (b>=0){  
                if(a>=0 and nums1[a]>nums2[b]){  
                    nums1[ptr]=nums1[a];  
                    a--;  
                }  
                else{  
                    nums1[ptr]=nums2[b];  
                    b--;  
                }  
                ptr--;  
            }  
        }  
      
```
In first while loop I am checking that is our nums2 is exhausted or not in first if condition is checking nums1 is exhausted or nums1 has the greater value then replace it in the end of nums1 else insert  nums2 value in nums1.  
      
Time complexity is O(m+n)   
      
      
      
11. **Set Mismatch**   
You have a set of integers s, which originally contains all the numbers from 1 to n. Unfortunately, due to some error, one of the numbers in s got duplicated to another number in the set, which results in repetition of one number and loss of another number.  
You are given an integer array nums representing the data status of this set after the error.  
Find the number that occurs twice and the number that is missing and return *them in the form of an array*.

```cpp
vector<int> findErrorNums(vector<int>& nums) {  
        int n=nums.size();  
        vector<int>arr(n+1,0);  
        for(int x:nums){  
            arr[x]++;  
        }  
        vector<int>ans(2);  
        for(int i=1;i<=n;i++){  
            if(arr[i]==2){  
                ans[0]=i;  
            }  
            else if(arr[i]==0){  
                ans[1]=i;  
            }  
        }  
        return ans;  
    }  
```  
Bruteforce hash array implementation is the best implementation with O(n) Time complexity.

But there is a mathematical approach of this question also . as we know sum of numbers till n is  n(n+1)/2 . So we will subtract the unique sum with actual sum to get the missing element and array sum \- unique sum is duplicate element.

```cpp
vector<int> findErrorNums(vector<int>& nums) {  
       int n = nums.size();  
       int actual_sum = n * (n + 1) / 2;  
       int array_sum = 0;  
       int unique_sum = 0;  
       unordered_set<int> s(nums.begin(), nums.end());  
       for (int a : nums) {  
           array_sum += a;  
       }  
       for (int a : s) {  
           unique_sum += a;  
       }  
       int missing = actual_sum - unique_sum;  
       int duplicate = array_sum - unique_sum;  
       return {duplicate, missing};  
   } 
``` 
ie. missingVAL \= actualSUM \-UniqueSUM; & duplicateVAL \= arraySUM-UniqueSUM

12. **Majority Element**  
      
    Given an array nums of size n, return *the majority element*.  
    The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

    

The basic sorting and median approach and hash approach is good but thats very simple. The sorting one take O(nlogn) Time complexity for sorting and hash one takes O(n)  space and time complexity.  
      
**Follow-up:** Could you solve the problem in linear time and in O(1) space?  
      
So our task is to find our winning candidate using majority voting technique also known as the Moore's voting Algorithm.  
      
It declare the winning candidate in single pass without allocating a n sized data structure.  
```cpp     
    int majorityElement(vector<int>& nums) {  
            int count = 0;  
            int candidate = 0;  
             
      
            for (int num : nums) {  
                if (count == 0) {  
                    candidate = num;  
                }  
                 
                if (num == candidate) {  
                    count++;  
                } else {  
                    count--;  
                }  
            }  
             
            return candidate;  
        }  
```
Thanks to Moore I just copied it lol 😂.  
      
      
      
      
      
      
13. **Majority Element II**  
      
    Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.  
      
    
Extended Moore’s Algo time . So I modified the algo by seeing the editorial of the given famous question.  
      
As n/3 in a array should be maximum 2 elements. So Keeping track of two candidates ie first:f and second:s.  
```cpp     
    vector<int> majorityElement(vector<int>& nums) {  
            if(nums.size()==1){  
                return nums;  
            }  
            int c1=0;  
            int c2=0;  
            int f=nums[0];  
            int s=nums[1];  
            for(int x:nums){  
                if(c1==0 and x!=s){  
                    c1=1;  
                    f=x;  
                }  
                else if(c2==0 and x!=f){  
                    c2=1;  
                    s=x;  
                }  
                else if(f==x){  
                    c1++;  
                }  
                else if(s==x){  
                    c2++;  
                }  
                else{  
                    c1--;  
                    c2--;  
                }  
            }  
            c1=0;c2=0;  
            int th=nums.size()/3;  
            for (int x:nums){  
                if(x==f){  
                    c1++;  
                }  
                else if(x==s){  
                    c2++;  
                }  
            }  
            vector<int>ans;  
            if(c1>th){  
                ans.push_back(f);  
            }  
            if(c2>th){  
                ans.push_back(s);  
            }  
            return ans;  
             
        }  
```  
Basically i am initialising my top two candidates starting from arrays first two elements , then i am traversing the whole array and checking weather my counter exhausted to 0 , if yes then i am checking if it is the first candidate or second then i am updating new candidate and initializing counter with 1\. And if I am encountering the same as my candidate then I'm incrementing the counters, and if not the  decrementing the counter.  
After then pushing it into the output vector only if its frequency is greater than the threshold given in the problem statement that is n/3.  
      
Time complexity is O(n) and space complexity is O(1) .  
      
      
14. **Find The Duplicate**
    Given an array of integers nums containing n \+ 1 integers where each integer is in the range \[1, n\] inclusive.

    There is only **one repeated number** in nums, return *this repeated number*.

    You must solve the problem **without** modifying the array nums and using only constant extra space.


The brute force method can be two nested for-loops finding the duplicate with time complexity of O(n2) or a hash data structure with time complexity of minimum O(n)  and with a hash set with one pass can take up to O(nlogn) but it will take O(n) space.


To solve it using a constant extra space ie. O(1) space complexity we will use the ‘fast and slow’ approach of  a finite state space.
As there is exact n states with n+1 transition rules as given in the array. We can use the logic of state space.

    
```cpp
    int slow = nums[0];
    int fast = nums[0];
    while (true) {
                slow = nums[slow];
                fast = nums[nums[fast]];
                if (slow == fast) {
                    break;
                }
            }
```
This detects a cycle in a state space with index representing and value present in that index is the next transition.
But the actual answer is to find where the loop started so I will find the meeting point of the slow and slow2 pointer and the intersection point is our answer.
```cpp
    int slow2 = nums[0];
    while (slow != slow2) {
                slow = nums[slow];
                slow2 = nums[slow2];
            }
    return slow;

```
Space Complexity of the above code is O(1) 

  


15. **Top k Frequent Elements**

    Given an integer array nums and an integer k, return *the* k *most frequent elements*. You may return the answer in **any order**.

So straightforward approach is to store the frequencies and pop out from a stored priority queue  
The time complexity of this approach is O(nlogn) because we have heapify after each push.

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {  
        map<int,int>mp;  
        for(int i=0;i<nums.size();i++){  
            mp[nums[i]]++;  
        }  
        priority_queue<pair<int,int>>pq;  
        for(auto it:mp){  
            pq.push({it.second,it.first});  
        }  
        vector<int>ans;  
        while(k--){  
            auto item=pq.top();  
            ans.push_back(item.second);  
            pq.pop();  
        }  
        return ans;

    }
```
**Follow up:** Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

So I tried of thinking of many methods to solve this problem but I was unable to do so. By going through the editorial I came to know about Quick Select Algorithm based on modified Quick Sort Partition algo.
```cpp
class Solution {  
private:  
    vector<int> unique;  
    map<int, int> count_map;

public:  
    int partition(int left, int right, int pivot_index) {  
        int pivot_frequency = count_map[unique[pivot_index]];  
        // 1. Move the pivot to the end  
        swap(unique[pivot_index], unique[right]);

        // 2. Move all less frequent elements to the left  
        int store_index = left;  
        for (int i = left; i <= right; i++) {  
            if (count_map[unique[i]] < pivot_frequency) {  
                swap(unique[store_index], unique[i]);  
                store_index += 1;  
            }  
        }

        // 3. Move the pivot to its final place  
        swap(unique[right], unique[store_index]);

        return store_index;  
    }

    void quickselect(int left, int right, int k_smallest) {  
        /*  
        Sort a list within left..right till kth less frequent element  
        takes its place.  
        */

        // base case: the list contains only one element  
        if (left == right) return;

        int pivot_index = left + rand() % (right - left + 1);

        // Find the pivot position in a sorted list  
        pivot_index = partition(left, right, pivot_index);

        //If the pivot is in its final sorted position  
        if (k_smallest == pivot_index) {  
            return;  
        } else if (k_smallest < pivot_index) {  
            // go left  
            quickselect(left, pivot_index - 1, k_smallest);  
        } else {  
            // go right  
            quickselect(pivot_index + 1, right, k_smallest);  
        }  
    }

    vector<int> topKFrequent(vector<int>& nums, int k) {  
        // build hash map: element and how often it appears  
        for (int n : nums) {  
            count_map[n] += 1;  
        }

        // array of unique elements  
        int n = count_map.size();  
        for (pair<int, int> p : count_map) {  
            unique.push_back(p.first);  
        }

        // kth top frequent element is (n - k)th less frequent.  
        // Do a partial sort: from less frequent to the most frequent, till  
        // (n - k)th less frequent element takes its place (n - k) in a sorted array.  
        // All elements on the left are less frequent.  
        // All the elements on the right are more frequent.  
        quickselect(0, n - 1, n - k);  
        // Return top k frequent elements  
        vector<int> top_k_frequent(k);  
        copy(unique.begin() + n - k, unique.end(), top_k_frequent.begin());  
        return top_k_frequent;  
    }  
}; 
``` 
But the catch is its worst case time complexity is still O(n2)  but average case complexity  is O(n)

Another solution for this that I came across in the editorial and hints is the bucket sort technique.
```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {  
       unordered_map<int,int>mp;  
       for(int n:nums){  
           mp[n]++;  
       }  
       vector<vector<int>>bucket(nums.size()+1);  
       for (auto it:mp){  
           bucket[it.second].push_back(it.first);  
       }  
       vector<int>ans;  
       for (int i=nums.size();i>=0;i--){  
           for (int item:bucket[i]){  
               ans.push_back(item);  
               if(ans.size()==k){  
                   return ans;  
               }  
           }  
       }  
       return ans;  
   }
```
Here I am first keeping track of frequency and making buckets with respect to frequency. In bucket\[i\] I am storing all the elements with frequency equals to i. And atlast traversing the buckets in reverse (high freq to low) and obtaining the answer. With average case complexity  is O(n).

16. **Max Consecutive Ones.**  
      
Given a binary array nums, return *the maximum number of consecutive* 1*'s in the array*.  
      
So basic approach to do so is just by keeping a counter.  

```cpp   
    int findMaxConsecutiveOnes(vector<int>& nums) {  
            int ans=0;  
            int cnt=0;  
            for (int i=0;i<nums.size();i++){  
                if(nums[i]==0){  
                    ans=max(ans,cnt);  
                    cnt=0;  
                }  
                else{  
                    cnt++;  
                }  
            }  
            return max(ans,cnt);  
        }  
```
We are doing it in one pass so the time complexity is O(n).

As given in the question “Binary” so we can just use Xor operator with 0 in the stream and maximize the output.
```cpp
int findMaxConsecutiveOnes(vector<int>& nums) {  
        int ans=0;  
        for(int x: nums){  
            int res=(x^0==1)?res+1:0;  
            ans=max(ans,res);  
        }  
        return ans;  
    }
```
Again with time complexity of O(n).

17. **Find Median Of The Datastream**
    The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

* For example, for arr \= \[2,3,4\], the median is 3\.  
* For example, for arr \= \[2,3\], the median is (2 \+ 3\) / 2 \= 2.5.  
  Implement the MedianFinder class:  
* MedianFinder() initializes the MedianFinder object.  
* void addNum(int num) adds the integer num from the data stream to the data structure.  
* double findMedian() returns the median of all elements so far. Answers within 10\-5 of the actual answer will be accepted.  
  


As we know we can divide the datastream into two equal parts left and right with median at the centre.  
So I am using three data structure , one “double” data type container for median  , Max Heap for left subpart and its top representing left adjacent and possible candidate for median and Min Heap for right subpart and its top representing right adjacent and possible candidate for median.

And one counter for tracking number of elements in datastream. If counter is one then first element and its only the median.  
If counter is even ie. even elements , then median is avg of tops of both two elements.  
And if counter is odd then container storing median is median.

Now inserting a number in this data structure is based on current median . if the  number is greater than current median then push it into right heap then heapify and re calculate else in left heap and re calculate.

```cpp
class MedianFinder {  
private:  
    priority_queue<int>left;  
    priority_queue<int,vector<int>,greater<int>>right;  
    double median;  
    int counter;  
public:  
    MedianFinder() {  
        median=0;  
        left={};  
        right={};  
        counter=0;  
    }  
     
    void addNum(int num) {  
        counter++;  
        if(counter==1){  
            median=num;  
            return;  
        }  
        if(counter%2==0){  
            if(median<num){  
                right.push(num);  
                left.push(static_cast<int>(median));  
                median=(static_cast<double>(right.top())+static_cast<double>(left.top()))/2;  
            }  
            else{  
                left.push(num);  
                right.push(static_cast<int>(median));  
                median=(static_cast<double>(right.top())+static_cast<double>(left.top()))/2;  
            }  
        }else{  
            if(num>median){  
                right.push(num);  
                median=right.top();  
                right.pop();  
            }  
            else{  
                left.push(num);  
                median=left.top();  
                left.pop();  
            }  
        }  
    }    
     
    double findMedian() {  
        return median;  
    }  
};
```
So the time complexity for adding a number is same as heapify ie. O(logn).

18. **Set Matrix Zeroes**  
    Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's. You must do it [in place](https://en.wikipedia.org/wiki/In-place_algorithm).

So my naive approach is i will push row and col indices whenever i find a 0 and then I will make  	all the entries 0 using those stored row and cols.  
```cpp 
	void setZeroes(vector<vector<int>>& matrix) {  
        unordered_set<int>r;  
        unordered_set<int>c;  
        int m=matrix.size();  
        int n=matrix[0].size();  
        for(int i=0;i<m;i++){  
            for(int j=0;j<n;j++){  
                if(matrix[i][j]==0){  
                    r.insert(i);  
                    c.insert(j);  
                }  
            }  
        }  
        for(int i:r){  
            for(int j=0;j<n;j++){  
                matrix[i][j]=0;  
            }  
        }  
        for (int j:c){  
            for(int i=0;i<m;i++){  
                matrix[i][j]=0;  
            }  
        }  
    }
```
	It has a space  complexity of O(m+n) not bad as brute force one has O(mn) space complexity.And it has a time complexity of O(n) considering the set data structure as unordered.  
	  
  **Follow up:**

* A simple improvement uses O(m \+ n) space, but still not the best solution. But Could you devise a constant space solution?  
So I will use the first cell of row and and first cell of col for flag and then will use those flag to make all values 0\.  
Edge case is the first col and first row so for that I will maintain two variable/flags then will make all values of first row or column accordingly.  
```cpp  
  void setZeroes(vector<vector<int>>& matrix) {  
         int rows = matrix.size();  
         int cols = matrix[0].size();  
         bool firstRowHasZero = false;  
         bool firstColHasZero = false;  
         // Check if the first row contains zero  
         for (int c = 0; c < cols; c++) {  
             if (matrix[0][c] == 0) {  
                 firstRowHasZero = true;  
                 break;  
             }  
         }  
         // Check if the first column contains zero  
         for (int r = 0; r < rows; r++) {  
             if (matrix[r][0] == 0) {  
                 firstColHasZero = true;  
                 break;  
             }  
         }  
         // Use the first row and column as markers  
         for (int r = 1; r < rows; r++) {  
             for (int c = 1; c < cols; c++) {  
                 if (matrix[r][c] == 0) {  
                     matrix[r][0] = 0;  
                     matrix[0][c] = 0;  
                 }  
             }  
         }  
         // Set the marked rows to zero  
         for (int r = 1; r < rows; r++) {  
             if (matrix[r][0] == 0) {  
                 for (int c = 1; c < cols; c++) {  
                     matrix[r][c] = 0;  
                 }  
             }  
         }  
         // Set the marked columns to zero  
         for (int c = 1; c < cols; c++) {  
             if (matrix[0][c] == 0) {  
                 for (int r = 1; r < rows; r++) {  
                     matrix[r][c] = 0;  
                 }  
             }  
         }  
         // Set the first row to zero if needed  
         if (firstRowHasZero) {  
             for (int c = 0; c < cols; c++) {  
                 matrix[0][c] = 0;  
             }  
         }  
         // Set the first column to zero if needed  
         if (firstColHasZero) {  
             for (int r = 0; r < rows; r++) {  
                 matrix[r][0] = 0;  
             }  
         }         
     }

```
So the new space complexity of this problem is O(1).

19. **Search in a 2D Matrix**

You are given an m x n integer matrix matrix with the following two properties:

* Each row is sorted in non-decreasing order.  
* The first integer of each row is greater than the last integer of the previous row.

Given an integer target, return true *if* target *is in* matrix *or* false *otherwise*.

So the brute force approach for this question is to traverse the whole matrix and find the target . But worst case time complexity of this code will be O(m\*n) 

So my first approach is to make the top right corner as my starting pointer and moving in the direction of desired target if found then will return true otherwise at the end will return false.

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {  
       int m=matrix.size();  
       int n=matrix[0].size();  
       int l=0;  
       int r=n-1;  
       while(l>=0 && r>=0 && l<m && r<n){  
           if(matrix[l][r]==target){  
               return true;  
           }  
           else if(matrix[l][r]<target){  
               l++;  
           }  
           else{  
               r--;  
           }  
       }  
       return false;  
   }  
```
So the worst case time complexity is O(m+n) , Now this is linear but we can optimize further by considering the whole matrix as a long array of size m\*n then applying binary search on it. 

Initializing m and n. 
```cpp 
   m=matrix.size();  
   n=matrix[0].size(); 
``` 
Mapping 1D index to 2D. 
```cpp 
   vector<int> mapping(int x){  
       int i=x/n;  
       int j=x%n;  
       return {i,j};  
   }  
```
And then applying the traditional binary search algorithm.  
```cpp
       int low =0;  
       int high=m*n-1;
       while (low <= high) {  
       int mid = low + (high - low) / 2;  
       vector<int>ij=mapping(mid);  
       int i=ij[0];  
       int j=ij[1];  
       if (matrix[i][j] == target)  
           return true;  
       if (matrix[i][j] < target)  
           low = mid + 1;  
       else  
           high = mid - 1;  
   	}  
   	return false;
```
So the time complexity of the code is now O(log(m\*n)).

20. **Maximum Absolute Sum**  
    You are given an integer array nums. The **absolute sum** of a subarray \[numsl, numsl+1, ..., numsr-1, numsr\] is abs(numsl \+ numsl+1 \+ ... \+ numsr-1 \+ numsr).  
    Return *the **maximum** absolute sum of any **(possibly empty)** subarray of* nums.  
      
      
Now the key idea in this problem is we have to use kadane’s Algorithm two times, one with positive values and one with negative.  
Our answer will be the maximum of any two.  
```cpp  
    int maxAbsoluteSum(vector<int>& nums) {  
            int maxSum = INT_MIN, currSum = 0;  
      
            for (int num : nums) {  
                currSum = max(num, currSum + num);  
                maxSum = max(maxSum, currSum);  
            }  
            int posi = maxSum;  
            for (int i = 0; i < nums.size(); i++) {  
                nums[i] = nums[i] * -1;  
            }  
            maxSum = INT_MIN, currSum = 0;  
      
            for (int num : nums) {  
                currSum = max(num, currSum + num);  
                maxSum = max(maxSum, currSum);  
            }  
            int negi = maxSum;  
            return max(posi, negi);  
        }  
```
The time complexity is O(n).