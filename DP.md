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
![][image1]  
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

    ![][image2]

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

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>