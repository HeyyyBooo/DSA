Report of Target 2025

# DYNAMIC PROGRAMMING

1. **Longest Palindromic Substring**  
   Given a string s, return *the longest* *palindromic* *substring* in s.  
   

I am Initializing a boolean dp data structure where dp(i,j) represents if the substring from i 	to j is palindrome or not.   
And two variable to keep track of the desired length (longest palindromic substring) and the 	starting point of the substring.  
	  
Now initially bestLen=1 bestStart=0 and dp(i,j)=true ∀ i=j.  
```cpp	  
                int n = s.size();  
        if (n < 2) return s;

        vector<vector<bool>> dp(n, vector<bool>(n, false));  
        int bestLen = 1, bestStart = 0;

        for (int i = 0; i < n; ++i) {  
            dp[i][i] = true;  
        }  
```	  
Now for any given i and j its true if its of size two and chars at i and j is same and another one is that char at i and j is same as well as sub string in between is also palindrome making	 overall palindrome.
```cpp
// Check substrings of length >= 2  
        for (int len = 2; len <= n; ++len) {  
            for (int i = 0; i + len - 1 < n; ++i) {  
                int j = i + len - 1;  
                if (s[i] == s[j]) {  
                    // If it's length 2, just check ends; otherwise check the inside substring  
                    if (len == 2 || dp[i+1][j-1]) {  
                        dp[i][j] = true;  
                        if (len > bestLen) {  
                            bestLen = len;  
                            bestStart = i;  
                        }  
                    }  
                }  
                // else dp[i][j] stays false  
            }  
        }

        return s.substr(bestStart, bestLen);  
```
The substring starting from bestStart having the bestLen size is our desired answer.  
The Time complexity of this solution is O(n2).

2. **Longest Common Subsequence**   
   Given two strings text1 and text2, return *the length of their longest **common subsequence**.* If there is no **common subsequence**, return 0\.  
   A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.  
* For example, "ace" is a subsequence of "abcde".  
  A **common subsequence** of two strings is a subsequence that is common to both strings.


    
Now its a standard problem of Dynamic Programming so just implementing it.  
```cpp
  int longestCommonSubsequence(string text1, string text2) {  
          int text1Length = text1.size(), text2Length = text2.size();  
          int dp[text1Length + 1][text2Length + 1];  
          memset(dp, 0, sizeof dp);  
          for (int i = 1; i <= text1Length; ++i) {  
              for (int j = 1; j <= text2Length; ++j) {  
                  if (text1[i - 1] == text2[j - 1]) {  
                      dp[i][j] = dp[i - 1][j - 1] + 1;  
                  } else {  
                      dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);  
                  }  
              }  
          }  
          return dp[text1Length][text2Length];  
      }  
```
Base Condition: if size of both strings remains 0 then the common subsequence in between is also of length 0\.  
DP Choice & Hypothesis: For a i,j state if chars are same then we just increment 1 from the dp\[i-1\]\[j-1\] as its just direct increase of char in subsequence length.  
But if its not same we have two option consider 1 char of string1 and move pointer of string2 to check and vice versa so we take maximum of our two options.


The time complexity of above code is O(mn).

3. **Coin Change**  
   You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.  
   Return *the fewest number of coins that you need to make up that amount*. If that amount of money cannot be made up by any combination of the coins, return \-1.  
   You may assume that you have an infinite number of each kind of coin.  
     
     
This direct implementation of unbounded knapsack problem. Here I am going to initialize the dp data structure such a way that and use min functioned unbounded knapsack to get the answer.  
```cpp  
   int coinChange(vector<int>& coins, int n) {  
           int m = coins.size();  
           int dp[m + 1][n + 1];  
           //dp[#coins][sumOfCoins]  
           for (int i = 1; i <= m; i++) {  
               dp[i][0] = 0;  
               //0 coins needed to make sum=0  
           }  
     
           for (int i = 0; i <= n; i++) {  
               dp[0][i] = INT_MAX - 1;  
               //infinite coins needed for any sum if #coins is 0  
           }  
     
           for (int i = 1; i <= n; i++) {  
               //if it is possible to build sum using that coin only ie. divisible  
               dp[1][i] = (i % coins[0] == 0) ? (i / coins[0]) : (INT_MAX - 1);  
           }  
     
     
           for (int i = 2; i <= m; i++) {  
               for (int j = 1; j <= n; j++) {  
                   //if possible then minimum of not take & take else not take  
                   dp[i][j] =  
                       (coins[i - 1] <= j)  
                           ? (min(dp[i - 1][j], 1 + dp[i][j - coins[i - 1]]))  
                           : dp[i - 1][j];  
               }  
           }  
           //if not possible then dp[m][n] will contain inf  
           return (dp[m][n] == INT_MAX - 1) ? -1 : dp[m][n];  
       }  
```      
The Time complexity of this code is O(n2).  
     
4. **Minimum Path Sum**  
   Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.  
   **Note:** You can only move either down or right at any point in time.  
     
     
Building the recursion first, that is , Initially sum is grid\[0\]\[0\] in top left then for any i and j its sum of current cell value and the minimum of the two paths.  
```cpp
   int rec(vector<vector<int>>&g,int i,int j){  
           if(i==0 and j==0){  
               return g[i][j];  
           }  
           else if (i<0 or j<0){  
               return INT_MAX-1;  
           }  
           return g[i][j]+min(rec(g,i-1,j),rec(g,i,j-1));  
       }
```  
Now Init condition lets say we shift each index 1 to shift indexing to 1 based then the base condition will first row and col as INT\_MAX-1 and dp\[1\]\[1\] will be grid\[0\]\[0\]. And follow the same logic for filling the dp table.  
```cpp  
   int minPathSum(vector<vector<int>>& grid) {  
           int m=grid.size();  
           int n=grid[0].size();  
           vector<vector<int>>dp(m+1,vector<int>(n+1,INT_MAX-10));  
           for(int i=1;i<=m;i++){  
               for(int j=1;j<=n;j++){  
                   if(i==1 and j==1){  
                       dp[1][1]=grid[0][0];  
                   }  
                   else{  
                       dp[i][j]=grid[i-1][j-1]+min(dp[i-1][j],dp[i][j-1]);  
                   }  
               }  
           }  
           return dp[m][n];  
       }  
```
Now the time complexity of the code reduced to O(mn).  
     
5. **Unique Paths**  
   There is a robot on an m x n grid. The robot is initially located at the **top-left corner** (i.e., grid\[0\]\[0\]). The robot tries to move to the **bottom-right corner** (i.e., grid\[m \- 1\]\[n \- 1\]). The robot can only move either down or right at any point in time.  
   Given the two integers m and n, return *the number of possible unique paths that the robot can take to reach the bottom-right corner*.  
     
     
So I am first writing the simple recursive algorithm that is to reach top left there is only one way and for all right and up movement just sum it.
```cpp  
   int rec(int i,int j){  
           if(i<0 or j<0){  
               return 0;  
           }  
           if(i==0 and j==0){  
               return 1;  
           }  
           return rec(i-1,j)+rec(i,j-1);  
       }  
   int uniquePaths(int m, int n) {  
           return rec(m-1,n-1);  
       }  
``` 
     
Now for Dynamic Programming approach I will initialize  the dp  with 0 and dp\[1\]\[1\] as one then will move forward with the same above logic.  
```cpp 
   int uniquePaths(int m, int n) {  
           vector<vector<int>>dp(m+1,vector<int>(n+1,0));  
           for(int i=1;i<=m;i++){  
               for(int j=1;j<=n;j++){  
                   if(i==1 and j==1){  
                       dp[i][j]=1;  
                   }  
                   else{  
                       dp[i][j]=dp[i-1][j]+dp[i][j-1];  
                   }  
               }  
           }  
           return dp[m][n];  
       }
```
Now the time complexity of the code reduced to O(mn).

6. **Edit Distance**  
   Given two strings word1 and word2, return *the minimum number of operations required to convert word1 to word2*.  
   You have the following three operations permitted on a word:  
* Insert a character  
* Delete a character  
* Replace a character  
  We define a 2D DP array of size `(word1.length + 1)` x `(word2.length + 1)`.  
  First row: Converting an empty `word1` to the first `i` characters of `word2` requires `i` insertions.  
  First column: Converting the first `i` characters of `word1` to an empty `word2` requires `i` deletions.
```cpp
    int dp[word1.size()+1][word2.size()+1];

          for(int i=0;i<=word2.size();i++){  
              dp[0][i]=i;  
          }  
          for(int i=0;i<=word1.size();i++){  
              dp[i][0]=i;  
          }  
``` 
    
    
  **If they match**, no operation is needed, so we carry over the value from the diagonal:   
  `dp[i][j] = dp[i-1][j-1]`.  
  **If they don't match**, we consider all 3 operations:
```cpp
  Replace: `dp[i-1][j-1] + 1`  
  Delete: `dp[i-1][j] + 1`  
  Insert: `dp[i][j-1] + 1`
```
So
```cpp
  for(int  i=1;i<=word1.size();i++){

              for(int j=1;j<=word2.size();j++){  
                  if(word1[i-1]==word2[j-1]){  
                      dp[i][j]=dp[i-1][j-1];  
                  }  
                  else{

                    dp[i][j]=1+min(dp[i-1][j-1],min(dp[i-1][j],dp[i][j-1]));  
                }  
            }  
        }  
        return dp[word1.size()][word2.size()];
```  
So the overall time complexity is  O(mn).

7. **Maximum Product Subarray**  
   Given an integer array nums, find a subarray that has the largest product, and return *the product*.  
     
So by observation we can say that the issue is with negative and zero coming in the flow of the Subarray stream.  
Now whenever there is a negative then the answer is either the left subarray sum or right subarray sum.  
Which I am storing in prefix and suffix arrays.  
 ![Screenshot-2023-08-05-174139](https://github.com/user-attachments/assets/e2dc87bb-52e7-4865-aef2-f832833d0580)

And to handle the case where the product becomes zero I am re-initializing the prefix or suffix counter as the current case.  
```cpp
   int maxProduct(vector<int>& nums) {  
           int n = nums.size();  
           vector<int> pre(n);  
           vector<int> suf(n);  
           pre[0] = nums[0];  
           for (int i = 1; i < n; i++) {  
               pre[i] = pre[i - 1] * nums[i];  
               if (pre[i - 1] == 0) {  
                   pre[i] = nums[i];  
               }  
           }  
           suf[n - 1] = nums[n - 1];  
           for (int i = n - 2; i >= 0; i--) {  
               suf[i] = suf[i + 1] * nums[i];  
               if (suf[i + 1] == 0) {  
                   suf[i] = nums[i];  
               }  
           }  
           int ans = INT_MIN;  
           for (int i = 0; i < n; i++) {  
               ans = max(ans, max(pre[i], suf[i]));  
           }  
           return ans;  
       }  
```
So time complexity of this optimized code is O(n).  
     
8. **Longest Increasing Subsequence**  
   Given an integer array nums, return *the length of the longest strictly increasing subsequence*.  
     
The basic take not take approach:  
Recursion tree is that if its increasing from previously used index then take or not take otherwise not take.  
```cpp
   int rec(vector<int>&nums,int i,int prev){  
           if(i==nums.size()){  
               return 0;  
           }  
           int not_take=rec(nums,i+1,prev);  
           int take=0;  
           if(prev==-1 or nums[i]>nums[prev]){  
               take=1+rec(nums,i+1,i);  
           }  
           return max(not_take,take);  
       }  
       int lengthOfLIS(vector<int>& nums) {  
           return rec(nums,0,-1);  
       }  
```
This has an exponential Time complexity.  
Now to optimize this I used the memoization technique directly on this recursive code. 
```cpp 
   int rec(vector<int>&nums,int i,int prev,vector<vector<int>>&dp){  
           if(i==nums.size()){  
               return 0;  
           }  
           if(dp[i][prev+1]!=-1) return dp[i][prev+1];  
           int not_take=rec(nums,i+1,prev,dp);  
           int take=0;  
           if(prev==-1 or nums[i]>nums[prev]){  
               take=1+rec(nums,i+1,i,dp);  
           }  
           return dp[i][prev+1]= max(not_take,take);  
       }  
       int lengthOfLIS(vector<int>& nums) {  
           vector<vector<int>>dp(nums.size()+1,vector<int>(nums.size(),-1));  
           return rec(nums,0,-1,dp);  
       }  
```
Now the time complexity reduces to O(n2).  
   

The Top-down Recursive approach is :
```cpp
	int dec(vector<int>& nums, int i, int prev) {  
        if (i < 0) {  
            return 0;  
        }  
        int not_take = dec(nums, i - 1, prev);  
        int take = 0;  
        if (prev == nums.size() || nums[i] < nums[prev]) {  
            take = 1 + dec(nums, i - 1, i);  
        }  
        return max(take, not_take);  
    }

    int lengthOfLIS(vector<int>& nums) {  
        return dec(nums, nums.size() - 1, nums.size());  
    }  
```
And its corresponding iterative  DP implementation is:
```cpp  
	int lengthOfLIS(vector<int>& nums) {  
        int n=nums.size();  
        vector<vector<int>>dp(n,vector<int>(n+1,0));  
        for(int i=0;i<n;i++){  
            for(int j=0;j<=n;j++){  
                if(j==n or nums[i]<nums[j]){  
                    if(i!=0)  
                    dp[i][j]=max(1+dp[i-1][i],dp[i-1][j]);  
                    else{  
                        dp[i][j]=1;  
                    }  
                }  
                else{  
                    if(i!=0)  
                    dp[i][j]=dp[i-1][j];  
                }  
            }  
        }  
        return dp[n-1][n];

    }  
```
The Time complexity of this code is O(n2).

And the following two are the most famous Approach of LIS.  
Using 1D DP:  
```cpp
	int lengthOfLIS(vector<int>& nums) {  
        int n = nums.size();  
        vector<int> dp(n, 1);  
        for (int i = 0; i < n; ++i)  
            for (int j = 0; j < i; ++j)  
                if (nums[i] > nums[j] && dp[i] < dp[j] + 1)  
                    dp[i] = dp[j] + 1;  
        return *max_element(dp.begin(), dp.end());  
    }  
```
Using Binary Search
```cpp  
	int lengthOfLIS(vector<int>& nums) {  
        vector<int> sub;  
        for (int x : nums) {  
            if (sub.empty() || sub[sub.size() - 1] < x) {  
                sub.push_back(x);  
            } else {  
                auto it = lower_bound(sub.begin(), sub.end(), x); // Find the index of the first element >= x  
                *it = x; // Replace that number with x  
            }  
        }  
        return sub.size();  
    }
```
In this Case the Time Complexity reduces to O(nlogn).

9. **Partition Equal Sum**  
   Given an integer array nums, return true *if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or* false *otherwise*.  
     
Sum of the two sums of the partition has to be equal to the total sum of whole array nums. So both partitions have to be of sum equals to half of the sum.  
And if we apply target sum on that will give the correct desired output.  

```cpp 
   bool canPartition(vector<int>& nums) {  
           int sum=accumulate(nums.begin(),nums.end(),0);  
           if(sum%2!=0) return false;  
           int n=nums.size();  
           int k=sum/2;  
           vector<vector<bool>>dp(n+1,vector<bool>(k+1,false));  
           //dp[items_considere_till_index][sum_reached_currently]  
           for(int i=0;i<=n;i++){  
               dp[i][0]=true;  
           }  
           for(int i=1;i<=n;i++){  
               for(int j=1;j<=k;j++){  
                   if(nums[i-1]<=j){  
                       dp[i][j]=dp[i-1][j-nums[i-1]] || dp[i-1][j];  
                   }  
                   else{  
                       dp[i][j]=dp[i-1][j];  
                   }  
               }  
           }  
           return dp[n][k];  
     
       }  
``` 

So if sum is not divisible by 2 means its is clearly not possible to have equal sum partition and if we do have even sum then we are finding the exact target is possible to achieve or not using the concept of take not\_take knapsack.  
The time complexity of the above approach is O(n2).

10. **Maximum Subarray Sum**  
    Given an integer array nums, find the subarray with the largest sum, and return *its sum*.  


```cpp
    int maxSubArray(vector<int>& nums) {      
            vector<vector<int>> dp(2, vector<int>(size(nums), -1));  
            return solve(nums, 0, false, dp);  
        }  
        int solve(vector<int>& A, int i, bool mustPick, vector<vector<int>>& dp) {  
            if(i >= size(A)) return mustPick ? 0 : -1e5;  
            if(dp[mustPick][i] != -1) return dp[mustPick][i];  
            if(mustPick)  
                return dp[mustPick][i] = max(0, A[i] + solve(A, i+1, true, dp));  
            return dp[mustPick][i] = max(solve(A, i+1, false, dp), A[i] + solve(A, i+1, true, dp));  
        }
```  
This is Dynamic Programming Approach using memoization.  
We can do it using famous Kadane Algorithm  
```cpp
    int maxSubArray(vector<int>& nums) {  
            int mx=INT_MIN;  
            int cur=0;  
            for(int i=0;i<nums.size();i++){  
                cur+=nums[i];  
                mx=max(mx,cur);  
                if(cur<0){  
                    cur=0;  
                }  
            }  
            return mx;  
      
        } 
``` 
As its follow up suggest to do it using Divide and Conquer Approach .  
Here it is.  
```cpp
    int maxSubArray(vector<int>& nums) {  
            return solve(nums,0,nums.size()-1);  
        }  
        int solve(vector<int>&nums,int l,int r){  
            if(l>r) return INT_MIN;  
            int mid=(l+r)/2;  
            int left_sum=0;  
            int right_sum=0;  
            int curr_sum=0;  
            for(int i=mid-1;i>=l;i--){  
                curr_sum+=nums[i];  
                left_sum=max(left_sum,curr_sum);  
            }  
            for(int i=mid+1,curr_sum=0;i<=r;i++){  
                curr_sum+=nums[i];  
                right_sum=max(right_sum,curr_sum);  
            }  
            return max({solve(nums,l,mid-1),solve(nums,mid+1,r),left_sum+right_sum+nums[mid]});  
        }  
```    
The time complexity of this approach is O(nlogn) better then the dynamic Programming which is having O(n2) complexity. And the Kadanes has the best O(n) time complexity.  
      
11. **Minimum Cost To Cut the a Stick**  
    Given a wooden stick of length n units. The stick is labelled from 0 to n. For example, a stick of length **6** is labelled as follows:
![statement](https://github.com/user-attachments/assets/479ca7ce-3f81-4538-a37d-615a45bed4a9)

   

Given an integer array cuts where cuts\[i\] denotes a position you should perform a cut at.  
You should perform the cuts in order, you can change the order of the cuts as you wish.  
The cost of one cut is the length of the stick to be cut, the total cost is the sum of costs of all cuts. When you cut a stick, it will be split into two smaller sticks (i.e. the sum of their lengths is the length of the stick before the cut). Please refer to the first example for a better explanation.  
Return *the minimum total cost* of the cuts.  
      
      
Now here comes the pattern of Matrix Chain Multiplication also known as the partition DP  
Our goal is to identify the sub problem that we will be calling for all the values in loop.  
```python      
for mid in range(left to right): 
```
solve for left to mid and mid to right and get optimal ans.  
      
Initialization 
```cpp
vector<vector<int>>dp; vector<int>newCuts; 
``` 
We will perform MCM in newCuts array that is sorted array in range of 0 to n containing intermediate cut locations.
```cpp
      int m=cuts.size();

            newCuts=cuts;  
            newCuts.push_back(0);  
            newCuts.push_back(n);  
            sort(newCuts.begin(),newCuts.end());  
            dp.assign(m+2,vector<int>(m+2,-1));  
```

And lastly we will call the MCM recursion with dp. : 

```cpp
    return cost(0,newCuts.size()-1);  
```
Now the main function of our code is based on partition DP is below.  
```cpp
        int cost(int i,int j){  
            if(dp[i][j]!=-1){  
                return dp[i][j];  
            }  
            if(j-i==1){  
                return 0;  
            }  
            int ans=INT_MAX;  
            for(int k=i+1;k<j;k++){  
                int c=cost(i,k)+cost(k,j)+newCuts[j]-newCuts[i];  
                ans=min(ans,c);  
            }  
            return dp[i][j]=ans;  
      
        }
```
The base cost is newCuts\[j\]-newCuts\[i\] 	That is the length of that states sticks size.  
      
We can also use Python3 where this can be implemented in very small code size.  
      
      
```python      
    def minCost(self, n: int, cuts: List[int]) -> int:  
            cut_points = [0] + sorted(cuts) + [n]  
      
            @cache  
            def get_optimal_cost(s, e):  
                return min(get_optimal_cost(s, m) + get_optimal_cost(m, e) for m in range(s + 1, e)) +   
                    cut_points[e] - cut_points[s] if s + 1 < e else 0  
              
            return get_optimal_cost(0, len(cut_points) - 1)  
```  
Here the  @cache  is doing the in build hash memoization.  
      
The time complexity of this is O(m3).  
      
12. **Palindrome Partitioning**  
    Given a string s, partition s such that every substring of the partition is a **palindrome**. Return *all possible palindrome partitioning of* s.  
      
Now coming to base case if in ith state lets say the size of string(ith sate) become null then the partitions till now is a part of valid answer so we will append it into answer.  
If the substring from left to mid is not palindrome we will not check the other half and if the left part is palindrome then we will append that it partitions and check for right partitions recursively.  
To check a substring is palindrome  or not we will reverse it and check for equality.  
```cpp
    bool isPalindrome(string s){  
            string s2=s;  
            reverse(s2.begin(),s2.end());  
            return s==s2;  
        }  
```
And we will call the partition dp function using the empty answer ans partitions array.  
```cpp
    vector<vector<string>> partition(string s) {  
            vector<vector<string>> ans;  
            vector<string> partitions;  
      
            getParts(s,partitions,ans);  
            return ans;  
        }  
```
Now the partitioning based (MCM) function is:  
```cpp
    void getParts(string s,vector<string>&partitions,vector<vector<string>>& ans){  
            if(s.size()==0){  
                ans.push_back(partitions);  
                return;  
            }  
            for(int i=0;i<s.size();i++){  
                string part = s.substr(0,i+1);  
      
                if(isPalindrome(part)){ //if left part is palindrome then move further  
                    partitions.push_back(part);  
                    getParts(s.substr(i+1),partitions,ans);  
                    partitions.pop_back(); //backtracking  
                }  
            }  
        }
```  
It is a pattern of Dynamic Programming MCM but we have not used the memoization here as its very tough , instead pruned some recursion subtrees. As the constraint is 1 \<= s.length \<= 16  
      
The time complexity of this code is O(n .2n)  
      
13. **Word Break**  
    Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.  
    **Note** that the same word in the dictionary may be reused multiple times in the segmentation.  
    
Now its same as the previous partitioning problem but the catch is we can use Dynamic Programming to memoize this backtracking.

	  
Now first I will store all the words in the dictionary in a hashmap then for all substring i will check for left partition then run for right partitions recursively.
```cpp
    unordered_map<string,int>m;  
    unordered_map<string,bool>dp;  
    bool fun(string s){  
        if(s.size()==0){  
            return true;  
        }  
        if(dp.find(s)!=dp.end()){  
            return dp[s];  
        }  
        for(int i=1;i<=s.size();i++){  
            if(m.find(s.substr(0,i))!=m.end() && fun(s.substr(i))){  
                return dp[s.substr(0,i)]=true;  
            }  
        }  
        return dp[s]=false;  
    }
```
The main task is now done just calling of our helper function fun( ) is remaining.  
```cpp
    bool wordBreak(string s, vector<string>& wordDict) {  
        for(string k: wordDict){  
            m[k]++;  
        }  
         
        return  fun(s);  
    }
```
The time complexity of partitioning approach in this problem is reduced to O(n3).

14. **Restore IP Addresses**  
    A **valid IP address** consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (**inclusive**) and cannot have leading zeros.  
* For example, "0.1.2.201" and "192.168.1.1" are **valid** IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are **invalid** IP addresses.  
  Given a string s containing only digits, return *all possible valid IP addresses that can be formed by inserting dots into* s. You are **not** allowed to reorder or remove any digits in s. You may return the valid IP addresses in **any** order.  
    

So the direct modified implantation of the partitioning Backtracking problem. Just modified loop and a isvalid function.  
```cpp  
  bool isvalid(string s) {  
          if (s.size() > 1 && s[0] == '0') return false;  
          if (s.size() > 3) return false;  
          int n = stoi(s);  
          if (n > 255) return false;  
          return true;  
      }  
```    
Now to convert the IP array into IP string format I have written a simple function.  
```cpp  
  string convertIP(vector<string>& ips) {  
          string ip = "";  
          for (string s : ips) {  
              ip += s;  
              ip.push_back('.');  
          }  
          ip.pop_back();  
          return ip;  
      }  
```
And the  main partition backtracking helper function is as follows.  
```cpp
  void getIPs(string s,vector<string>&ip){  
          if(ip.size()==4){  
              if(s.size()==0) ans.push_back(convertIP(ip));  
              return;  
          }  
          for (int i = 1; i <= 3 && i <= s.size(); i++) {  
              string part = s.substr(0, i);  
              if (isvalid(part)) {  
                  ip.push_back(part);  
                  getIPs(s.substr(i), ip);  
                  ip.pop_back();  
              }  
          }  
      }  
```
The loop here is restricting the size of substring to 3 as valid solutions has to be max 3 sized substring.  
    
And finally calling this into our main function getIPs(s,ip);  
Again this has an exponential time complexity.  
    
15. **Ones and Zeros**  
    You are given an array of binary strings strs and two integers m and n.  
    Return *the size of the largest subset of strs such that there are **at most*** m 0*'s and* n 1*'s in the subset*.  
    A set x is a **subset** of a set y if all elements of x are also elements of y.  
    This is direct implementation of the 2 dimensional Knapsack and 3 dimensional Dynamic programming approach.  
   
We will just follow the take not take approach.  
And for this we will build our own items array, that is the ones array and zeroes array containing counts of zeroes and ones in the original string present in that index  
```cpp
    vector<int> counts(string s) {  
            int z = 0;  
            for (char c : s) {  
                if (c == '0') {  
                    z++;  
                }  
            }  
            return {(int)s.size() - z, z};  
        }  
```
This is a helper function to build that array.  
Based on constraints build the DP matrix.  
```cpp
    int findMaxForm(vector<string>& strs, int m, int n) {  
            int dp[601][101][101];  
            vector<int> ones;  
            vector<int> zeros;  
            for (string s : strs) {  
                vector<int> cnts = counts(s);  
                ones.push_back(cnts[0]);  
                zeros.push_back(cnts[1]);  
            }  
            memset(dp, 0, sizeof(dp));  
            for (int i = 1; i <= strs.size(); i++) {  
                int one = ones[i - 1];  
                int zero = zeros[i - 1];  
                for (int j = 0; j <= m; j++) {  
                    for (int k = 0; k <= n; k++) {  
                        if (j >= zero && k >= one) {  
                            dp[i][j][k] = max(  
                                dp[i - 1][j][k], // not take  
                                dp[i - 1][j - zero][k - one] + 1 // take  
                            );  
                        } else {  
                            dp[i][j][k] = dp[i - 1][j][k]; // not take  
                        }  
                    }  
                }  
            }  
            return dp[size(strs)][m][n];  
        } 
``` 
The time complexity of this algorithm is O(n3).
