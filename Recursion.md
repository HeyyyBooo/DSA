Report of Target 2025

# RECURSION

1. **Sudoku Solver**  
   Write a program to solve a Sudoku puzzle by filling the empty cells.

The '.' character indicates empty cells.	

A check function to check weather the number can be entered or not.  
```cpp
 bool check(vector<vector<char>>& board, int i, int j,char ch){  
        int p=i/3;  
        int q=j/3;  
        p*=3;  
        q*=3;  
        for(int m=p;m<p+3;m++){  
            for(int n=q;n<q+3;n++){  
                if(board[m][n]==ch){  
                    return false;  
                }  
            }  
        }  
        for(int m=0;m<9;m++){  
            if(board[m][j]==ch){  
                return false;  
            }  
        }  
        for(int m=0;m<9;m++){  
            if(board[i][m]==ch){  
                return false;  
            }  
        }  
        return true;

    }
```
Now the main recursion for current {i,j} check if its in the range of row , whether the row needs to be changed or we just ended.  
If  there is already a number other than ‘.’ we skip it.  
If not then we try all combination from 1 to 9 which is valid.
```cpp
bool rec(int row,int col,vector<vector<char>>& board){  
        if(row==9) return true;  
        if(col==9) return rec(row+1,0,board);  
        if(board[row][col]!='.') return rec(row,col+1,board);  
        for(char ch='1';ch<='9';ch++){  
            if(check(board,row,col,ch)){  
                board[row][col]=ch;  
                if(rec(row,col+1,board)){  
                    return true;  
                }  
            }  
            board[row][col]='.';  
        }  
        return false;  
    }
```
Then we call this helper recursion function in the main function with the init value of i and j as 0 & 0\.
```cpp
  void solveSudoku(vector<vector<char>>& board) {  
        bool temp=rec(0,0,board);  
    }
```
The time complexity of this code is O(9(n\*n)) where n is 9\.

2. **Combination Sum**  
   Given an array of **distinct** integers candidates and a target integer target, return *a list of all **unique combinations** of* candidates *where the chosen numbers sum to* target*.* You may return the combinations in **any order**.  
   The **same** number may be chosen from candidates an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.  
   The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.  
   

	  
	It follows the take not take approach of multiple combinations picking.  
	First we take the number and find if it reaches target number as sum then recurse to find other way by not taking the number. 
```cpp
void rec(vector<int>&nums,int i,int target,vector<int>&temp,vector<vector<int>>&ans){  
        if(target==0){  
            ans.push_back(temp);  
            return;  
        }  
        if(target<0 or i>=nums.size()){  
            return;  
        }  
         
        temp.push_back(nums[i]);  
        rec(nums,i,target-nums[i],temp,ans);  
        temp.pop_back();  
        rec(nums,i+1,target,temp,ans);

    }  
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {  
        vector<vector<int>>ans;  
        vector<int>temp;  
        rec(candidates,0,target,temp,ans);  
        return ans;  
    }
```
The time complexity of the code is O(2n).

3. **Combination Sum II**  
   Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.  
   Each number in candidates may only be used **once** in the combination.  
   **Note:** The solution set must not contain duplicate combinations.  
     
     
   The main catch of this is question is that duplicates are not allowed and duplicates combination is also not allowed.  
   So for combination I used set in c++ then transferred unique elements in vector and for python I used function to get unique elements in list.  
     
```python    
   def get_unique_elements(input_list):  
              seen = []  
              for element in input_list:  
                  if element not in seen:  
                      seen.append(element)  
              return seen  
```  
     
And for unique elements consideration I sorted it and skipped all same elements for the not take part.  
   	  
     
Python version as follows:  
```python   
   def rec(nums,i,target,lis):  
              if target == 0 :  
                  ans.append(lis[:])  
                  return  
              if i==len(nums) or target<0:  
                  return  
              lis.append(nums[i])  
              rec(nums,i+1,target-nums[i],lis)  
              lis.pop()  
              nxt=i+1  
              while nxt<len(nums) and nums[nxt]==nums[i] :  
                  nxt+=1  
              rec(nums,nxt,target,lis)  
              return 
``` 
C++ version :  
```cpp
   set<vector<int>>ans;  
      void rec(vector<int>& nums,int i,int target,vector<int>&temp){  
          if(target==0){  
              ans.insert(temp);  
              return;  
          }  
          if(i==nums.size() or target<0){  
              return;  
          }  
          temp.push_back(nums[i]);  
          rec(nums,i+1,target-nums[i],temp);  
          temp.pop_back();  
          int nxt=i+1;  
          while(nxt<nums.size() and nums[nxt]==nums[i]){  
              nxt++;  
          }  
          rec(nums,nxt,target,temp);  
          return;  
      }  
```
     
And atlast the result compilation as vector and List.  
     
```cpp   
   vector<vector<int>> combinationSum2(vector<int>& nums, int target) {  
          sort(nums.begin(),nums.end());  
          vector<int>temp;  
          rec(nums,0,target,temp);  
          vector<vector<int>>res;  
          for(vector<int>v:ans){  
              res.push_back(v);  
          }  
          return res;  
      }  
```   
And for python   
     
   
```python 
    candidates.sort()

          rec(candidates,0,targets,li)  
          res=list(get_unique_elements(ans))  
          return res  
```
The overall worst case time complexity of the code is O(2n).  
     
4. **Permutations**  
   Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.  
     
Taking all options at I state taking and not taking. That is nothing but recursion with loop.  
     
```cpp  
   vector<vector<int>>ans;  
      bool isthere(vector<int>&temp,int n){  
          for(int x:temp){  
              if(x==n){  
                  return true;  
              }  
          }  
          return false;  
      }  
      void rec(vector<int>&nums,vector<int>&temp){  
          if(temp.size()==nums.size()){  
              ans.push_back(temp);  
              return;  
          }  
          for(int i=0;i<nums.size();i++){  
              if(isthere(temp,nums[i])) continue;  
              temp.push_back(nums[i]);  
              rec(nums,temp);  
              temp.pop_back();  
          }  
          return;  
      }  
     
      vector<vector<int>> permute(vector<int>& nums) {  
          vector<int>temp;  
          rec(nums,temp);  
          return ans;  
      }  
```
5. **Permutation II**  
   Given a collection of numbers, nums, that might contain duplicates, return *all possible unique permutations in any order.*  
     
     
   Using a flag  boolean array to check we have used the same or not and to tackle duplicates within loop sorted the array before calling the recursion.  
```cpp 
   vector<vector<int>>ans;  
      void rec(vector<int>&nums,int i,vector<int>&temp,vector<bool>&flag){  
          if(temp.size()==nums.size()){  
              ans.push_back(temp);  
              return;  
          }  
          for(int i=0;i<nums.size();i++){  
              if(flag[i] or (i>0 and nums[i]==nums[i-1] and !flag[i-1])){  
                  continue;  
              }  
              flag[i]=true;  
              temp.push_back(nums[i]);  
              rec(nums,i+1,temp,flag);  
     
              flag[i]=false;  
              temp.pop_back();  
          }  
          return;  
      }  
      vector<vector<int>> permuteUnique(vector<int>& nums) {  
          vector<int>temp;  
          vector<bool>flag(nums.size(),false);  
          sort(nums.begin(),nums.end());  
          rec(nums,0,temp,flag);  
          return ans;  
      }  
``` 
     
     
      
     
6. **Subsets II**  
   	  
   Given an integer array nums that may contain duplicates, return *all possible* *subsets (the power set)*.  
   The solution set must not contain duplicate subsets. Return the solution in any order.  
     
To again tackle the duplicate value we will sort and use set data structure and basic take not take.  
     
```cpp   
   set<vector<int>>unq;  
      void rec(vector<int>&nums,int i,vector<int>&temp){  
          if(i>=nums.size()){  
              unq.insert(temp);  
              return;  
          }  
          temp.push_back(nums[i]);  
          rec(nums,i+1,temp);  
          temp.pop_back();  
          rec(nums,i+1,temp);  
          return;  
      }  
      vector<vector<int>> subsetsWithDup(vector<int>& nums) {  
          sort(nums.begin(),nums.end());  
          vector<int>temp;  
          rec(nums,0,temp);  
          vector<vector<int>>ans;  
          for(vector<int> v:unq){  
              ans.push_back(v);  
          }  
          return ans;  
      }  
```
Time Complexity is O(2n) .  
     
     
7. **Permutation Sequence**  
   The set \[1, 2, 3, ..., n\] contains a total of n\! unique permutations.  
   By listing and labeling all of the permutations in order, we get the following sequence for n \= 3:  
     
   Given n and k, return the kth permutation sequence.  
     
     
So my first Approach is to use the previously implemented nextPermutation function k times.  
```cpp
   class Solution {  
   private:  
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
               int temp = nums[i];  
               nums[i] = nums[j];  
               nums[j] = temp;  
           }  
           reverse(nums.begin() + i + 1, nums.end());  
       }  
    public:  
       string getPermutation(int n, int k) {  
           vector<int>prm;  
           for(int i=1;i<=n;i++){  
               prm.push_back(i);  
           }  
           k--;  
           while(k--){  
               nextPermutation(prm);  
           }  
           string ans="";  
           for(int x:prm){  
               ans+=to_string(x);  
           }  
           return ans;  
       }  
   };  
``` 
The time complexity of this code is O(kn).  
     
Another approach is mathematical finding of nth using dictionary analogy.  
For which I will need factorials so Initially I precomputed all the needed factorials.  
     
```cpp     
   string getPermutation(int n, int k) {  
           vector<int>fact(n+1,1);  
           for(int i=1;i<=n;i++){  
               fact[i]=i*fact[i-1];  
           }  
           vector<int>mp(n+1,1);  
           string temp;  
           for(int i=1;i<=n;i++){  
               for(int j=1;j<=n;j++){  
                   if(mp[j]){  
                       int fc=fact[n-i];  
                       if(fc>=k){  
                           temp+=to_string(j);  
                           mp[j]=0;  
                           break;  
                       }  
                       else{  
                           k-=fc;  
                       }  
                   }  
               }  
           }  
           return temp;  
       }  
```   
     
     
Here mp\[\] is keeping track of the number that is already used. And fc denotes the number of possible permutations after fixing that number. If its greater than k then it contains the combination so append the current number j and mark mp as used and jump to next iteration.   
Now if fc\<k then its then the permutation is in this cycle only so we decrease the number with the possibilities.  
The worst case time complexity O(n2)  
     
     
8. **POW(X,N)**  
   Implement [pow(x, n)](http://www.cplusplus.com/reference/valarray/pow/), which calculates x raised to the power n (i.e., xn).  
     
```cpp  
   class Solution {  
   private:  
       double rec(double x, long n) {  
           if (n == 0)  
               return 1.0;  
           return (n % 2 ? x : 1) * rec(x * x, n / 2);  
       }  
     
   public:  
       double myPow(double x, int n) {  
           if (n < 0) {  
               x = 1 / x;  
           }  
     
           long num = labs(n);  
           return rec(x, num);  
       }  
   };  
```
Direct binary Exponentiation implementation.  
Time Complexity of this implementation is *O*(*Log*∣*N*∣)