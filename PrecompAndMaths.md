Report of Target 2025

# **PRECOMPUTATION & MATH**

1. **Count Primes**  
   Given an integer n, return *the number of prime numbers that are strictly less than* n.  
     
   So what we do here is precompute all numbers primality using a sieve of Eratosthenes. That is using initial prime 2 we mark all its multiples as not prime then check for next available prime and continue to do so till square root of max limit.  
     
Here is the optimized version.  
```python 
   def sieve(self,n: int)->int :  
           N=6*1000000  
           isprime=[True]*N  
           isprime[0]=False  
           isprime[1]=False  
           for i in range(2,int(n**0.5)+1):  
               if(isprime[i]):  
                   for j in range(i*i,n,i):  
                       isprime[j]=False 
```
This has almost linear time complexity.  
Here is the complete code using sieve of  Eratosthenes of the problem statement.  
```python 
   def countPrimes(self, n: int) -> int:  
           if n<=2:  
               return 0  
           isprime=[1 for _ in range(n)]  
           def sieve():  
               isprime[0]=0  
               isprime[1]=0  
               for i in range(2,int(n**0.5)+1):  
                   if isprime[i]:  
                       for j in range(i*i,n,i):  
                           isprime[j]=0  
           sieve()  
           return sum(isprime) 
``` 
Slightly higher time complexity then the linear time complexity.  
     
2. **Range Sum Query 2D**  
   Given a 2D matrix matrix, handle multiple queries of the following type:  
* Calculate the **sum** of the elements of matrix inside the rectangle defined by its **upper left corner** (row1, col1) and **lower right corner** (row2, col2).  
  Implement the NumMatrix class:  
* NumMatrix(int\[\]\[\] matrix) Initializes the object with the integer matrix matrix.  
* int sumRegion(int row1, int col1, int row2, int col2) Returns the **sum** of the elements of matrix inside the rectangle defined by its **upper left corner** (row1, col1) and **lower right corner** (row2, col2).  
You must design an algorithm where sumRegion works on O(1) time complexity.

I am going to precompute the rectangle sum from 0,0 to i,j in sums\[i\]\[j\]. Then using rectangle property I will subtract smaller extra rectangles from top and left to get the desired answer.

```python 
  class NumMatrix:

      def __init__(self, matrix: List[List[int]]):

          m=len(matrix)

          n=len(matrix[0])

          self.sums=[[0 for _ in range(n+1)] for __ in range(m+1)]

          for i in range(1,m+1):

              for j in range(1,n+1):

                  self.sums[i][j]=matrix[i-1][j-1]+self.sums[i-1][j]+self.sums[i][j-1]-self.sums[i-1][j-1]


      def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:

          return self.sums[row2+1][col2+1]-self.sums[row1][col2+1]-self.sums[row2+1][col1]+self.sums[row1][col1]
```

Having a time complexity of O(mn).

3. **Multiply Strings**  
   Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.  
```python  
   def multiply(self, num1: str, num2: str) -> str:  
           n,m=len(num1),len(num2)  
           res=[0]*(n+m)  
           for i in range(n-1,-1,-1):  
               for j in range(m-1,-1,-1):  
                   mul=(ord(num1[i])-ord('0'))*(ord(num2[j])-ord('0'))  
                   sums=mul+res[i+j+1]  
                   res[i+j+1]=sums%10  
                   res[i+j]+=sums//10  
           product=''.join(map(str,res)).lstrip('0')  
           return product if product else "0"  
```  
Using in hand approach of multiplication using fancy indexing.  
     
4. **Bulls & Cows**  
   You are playing the [**Bulls and Cows**](https://en.wikipedia.org/wiki/Bulls_and_Cows) game with your friend.  
   You write down a secret number and ask your friend to guess what the number is. When your friend makes a guess, you provide a hint with the following info:  
* The number of "bulls", which are digits in the guess that are in the correct position.  
* The number of "cows", which are digits in the guess that are in your secret number but are located in the wrong position. Specifically, the non-bull digits in the guess that could be rearranged such that they become bulls.  
Given the secret number and your friend's guess, return *the hint for your friend's guess*.  
The hint should be formatted as "xAyB", where x is the number of bulls and y is the number of cows. Note that both secret and guess may contain duplicate digits.  
    
```python 
  def getHint(self, secret: str, guess: str) -> str:  
          bulls = 0  
          secret_counter = Counter()  
          guess_counter = Counter()  
           
          for i in range(len(secret)):  
              if secret[i] == guess[i]:  
                  bulls += 1  
              else:  
                  secret_counter[secret[i]] += 1  
                  guess_counter[guess[i]] += 1  
    
          cows = 0  
          for ch in secret_counter:  
              if ch in guess_counter:  
                  cows += min(secret_counter[ch], guess_counter[ch])  
    
          return f"{bulls}A{cows}B"  
```
Direct implementation using all edge cases .  
    
5. **Matrix Block Sum**   
   Given a m x n matrix mat and an integer k, return *a matrix* answer *where each* answer\[i\]\[j\] *is the sum of all elements* mat\[r\]\[c\] *for*:  
* i \- k \<= r \<= i \+ k,  
* j \- k \<= c \<= j \+ k, and  
* (r, c) is a valid position in the matrix.  
So I am going to use that Precomputation class of matrix that I  wrote .earlier.  
```python     
  class NumMatrix:"""Same as previous"""   
  class Solution:  
      def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:  
          precom=NumMatrix(mat)  
          m=len(mat)  
          n=len(mat[0])  
          ans=[[0 for _ in range(n)] for __ in range(m)]  
          def isvalidi(i):  
              if i<0 or i>=m:  
                  return False  
              return True  
          def isvalidj(j):  
              if j<0 or j>=n:  
                  return False  
              return True  
          for i in range(m):  
              for j in range(n):  
                  from_i,from_j=i-k,j-k  
                  to_i,to_j=i+k,j+k  
                  while not isvalidi(from_i):  
                      from_i+=1  
                  while not isvalidj(from_j):  
                      from_j+=1  
                  while not isvalidi(to_i):  
                      to_i-=1  
                  while not isvalidj(to_j):  
                      to_j-=1  
                  ans[i][j]=precom.sumRegion(from_i,from_j,to_i,to_j)  
          return ans  
    
```
    
6. **Diagonal Traverse**  
   Given an m x n matrix mat, return *an array of all the elements of the array in a diagonal order*  
     
On visualising I get the pattern.  
Here it is.  
```python 
   def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:  
           m=len(mat)  
           n=len(mat[0])  
           def isvalid(i,j):  
               if i<0 or i>=m:  
                   return False  
               if j<0 or j>=n:  
                   return False  
               return True  
           counter=0  
           ans=[]  
           i,j=0,0  
           while(len(ans)!=m*n-1):  
               ans.append(mat[i][j])  
               if counter%2==0:  
                   if isvalid(i-1,j+1):  
                       i-=1  
                       j+=1  
                   elif isvalid(i,j+1):  
                       j+=1  
                       counter+=1  
                   else:  
                       i+=1  
                       counter+=1  
               else:  
                   if isvalid(i+1,j-1):  
                       i+=1  
                       j-=1  
                   elif isvalid(i+1,j):  
                       i+=1  
                       counter+=1  
                   else:  
                       j+=1  
                       counter+=1  
           ans.append(mat[i][j])  
           return ans  
```
With a linear Time complexity.  
7. **Construct Product Matrix**  
   Given a **0-indexed** 2D integer matrix grid of size n \* m, we define a **0-indexed** 2D matrix p of size n \* m as the **product** matrix of grid if the following condition is met:  
* Each element p\[i\]\[j\] is calculated as the product of all elements in grid except for the element grid\[i\]\[j\]. This product is then taken modulo 12345.  
Return *the product matrix of* grid.  
    
This is 2D implementation of suffix and prefix multiplication precomputation.  
```python     
  def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:  
          r,c=len(grid),len(grid[0])  
          n=r*c  
          def RedDim(i,j):  
              return i*c+j  
          def IncDim(k):  
              i=k//c  
              j=k%c  
              return i,j  
          pre=[1]*n  
          suf=[1]*n  
          for i in range(1,n):  
              j,k=IncDim(i-1)  
              pre[i]=(grid[j][k]*pre[i-1])%12345  
          for i in range(n-2,-1,-1):  
              j,k=IncDim(i+1)  
              suf[i]=(grid[j][k]*suf[i+1])%12345  
          ans=[[1 for _ in range(c)] for __ in range(r)]  
          for i in range(r):  
              for j in range(c):  
                  ans[i][j]=(pre[RedDim(i,j)]*suf[RedDim(i,j)])%12345  
          return ans  
```
So the time complexity is O(rc).