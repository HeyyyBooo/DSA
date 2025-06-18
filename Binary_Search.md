Report of Target 2025

# BINARY SEARCH

1. **Count The Number of Fair Pairs**  
   Given a **0-indexed** integer array nums of size n and two integers lower and upper, return *the number of fair pairs*.  
   A pair (i, j) is **fair** if:  
* 0 \<= i \< j \< n, and  
* lower \<= nums\[i\] \+ nums\[j\] \<= upper  
  As addition is commutative so we sort the nums first and fix ith index and search for all valid bounds of the range .  
  On Rearranging the constraints we get lower \- nums\[i\] \<= nums\[j\] \<= upper \- nums\[i\] so we find lower-nums\[i\] and upper-nums\[i\] positions in sorted array from i+1 to length of nums using binary search.  
    
  ```python
from bisect import bisect_left, bisect_right

def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
    nums.sort()
    ans = 0
    for i in range(len(nums)):
        left = bisect_left(nums, lower - nums[i], i + 1, len(nums))
        right = bisect_right(nums, upper - nums[i], i + 1, len(nums))
        ans += (right - left)
    return ans
```
---

  The overall Time Complexity of this problem is O(nlogn).  
    
2. **Find First and Last Position  Element in a Sorted Array**  
   Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.  
   If target is not found in the array, return \[-1, \-1\].  
   Direct Implementation of binary search.  
   ```python
from bisect import bisect_left

def searchRange(self, nums: List[int], target: int) -> List[int]:
    index = bisect_left(nums, target)
    if len(nums) == index or nums[index] != target:
        return [-1, -1]
    ans = [index, index]
    while index < len(nums) and nums[index] == target:
        ans[1] = index
        index += 1
    return ans
```

---
     
3. **Peak Index in Mountain Array**  
   You are given an integer **mountain** array arr of length n where the values increase to a **peak element** and then decrease.  
   Return the index of the peak element.  
     
```python
def peakIndexInMountainArray(self, arr: List[int]) -> int:
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[mid - 1] and arr[mid] > arr[mid + 1]:
            return mid
        elif arr[mid] < arr[mid - 1]:
            right = mid
        else:
            left = mid + 1
    return mid
```

---
     

   Again simple direct implementation of binary search with tweaked search criteria as peak that is both left and right element is lower than mid.  
     
4. **Find Right Intervals**  
   You are given an array of intervals, where intervals\[i\] \= \[starti, endi\] and each starti is **unique**.  
   The **right interval** for an interval i is an interval j such that startj \>= endi and startj is **minimized**. Note that i may equal j.  
   Return *an array of **right interval** indices for each interval i*. If no **right interval** exists for interval i, then put \-1 at index i.  
     
     
   So we know we have to sort the array based on starting time and have to find a position that starts just after our current interval ends.  
   We also have to keep track of the original index as its what we have to return.  
   Make a list of starts and indexes then sorting it then performing binary search of ending intervals.  

      ```python
from bisect import bisect_left

def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
    n = len(intervals)
    starts = sorted((interval[0], i) for i, interval in enumerate(intervals))
    starts_only = [start for start, _ in starts]
    ans = []
    for interval in intervals:
        end = interval[1]
        index = bisect_left(starts_only, end)
        if index < n:
            ans.append(starts[index][1])
        else:
            ans.append(-1)
    return ans
```

---  
     
     
  
   It traverses the original intervals and finds the best match of the index that starts exactly after the current interval.  
   So problem is done in average case time complexity of O(nlogn).  
     
5. **Search in a Rotated Sorted Array**  
   There is an integer array nums sorted in ascending order (with **distinct** values).  
   Prior to being passed to your function, nums is **possibly rotated** at an unknown pivot index k (1 \<= k \< nums.length) such that the resulting array is \[nums\[k\], nums\[k+1\], ..., nums\[n-1\], nums\[0\], nums\[1\], ..., nums\[k-1\]\] (**0-indexed**). For example, \[0,1,2,4,5,6,7\] might be rotated at pivot index 3 and become \[4,5,6,7,0,1,2\].  
   Given the array nums **after** the possible rotation and an integer target, return *the index of* target *if it is in* nums*, or* \-1 *if it is not in* nums.  
     
   Our task is to find the pivot from where it started rotating and then do two binary searches on two splits.  
   But to find the pivot we also have to do it in logn so we have to find the pivot using binary search only.  
   Our base case will be an element whose prev and next is greater than the mid element. And if the mid element is greater than the first element then we have to search in the right part and if the mid element is less than the last element then we have to search in the left part.  
     
     
     
      
   ```python
from bisect import bisect_left

def search(self, nums: List[int], target: int) -> int:
    def findpivot():
        left, right = 0, len(nums) - 1
        if nums[left] < nums[right]:
            return 0
        while left <= right:
            mid = (left + right) // 2
            prev = (mid - 1) % len(nums)
            nxt = (mid + 1) % len(nums)
            if nums[mid] <= nums[prev] and nums[mid] <= nums[nxt]:
                return mid
            elif nums[mid] >= nums[0]:
                left = mid + 1
            elif nums[mid] <= nums[len(nums) - 1]:
                right = mid - 1
        return 0

    index = findpivot()
    ans1 = bisect_left(nums, target, 0, index)
    if ans1 < len(nums) and target == nums[ans1]:
        return ans1
    ans2 = bisect_left(nums, target, index, len(nums))
    if ans2 < len(nums) and target == nums[ans2]:
        return ans2
    return -1
```

---
   So the time complexity of this search is O(logn).  
     
6. **Koko Eating Bananas**  
   Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles\[i\] bananas. The guards have gone and will come back in h hours.  
   Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.  
   Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.  
   Return *the minimum integer* k *such that she can eat all the bananas within* h *hours*.  
     
     
   So this is a problem on binary search for solutions. we will be finding the best solution  until our search space is ended.  
   First we initialize the low and high from search space.  
   Then for mid we calculate the solution then navigate towards the best desired solution.  
   ```python
from math import ceil

def minEatingSpeed(self, piles: List[int], h: int) -> int:
    low = 1
    high = max(piles)

    def getHours(k):
        hrs = 0
        for pile in piles:
            hrs += ceil(pile / k)
        return hrs

    ans = high
    while low <= high:
        mid = (low + high) // 2
        hrs = getHours(mid)
        if hrs <= h:
            ans = min(ans, mid)
            high = mid - 1
        else:
            low = mid + 1
    return ans
```

---
   Same in this case I calculated the solution that is hours required to eat banana is calculated using mid value. Then using direction we moved towards the better solution.  
     
7. **Find the smallest Divisor**  
   Given an array of integers nums and an integer threshold, we will choose a positive integer divisor, divide all the array by it, and sum the division's result. Find the **smallest** divisor such that the result mentioned above is less than or equal to threshold.  
   Each result of the division is rounded to the nearest integer greater than or equal to that element. (For example: 7/3 \= 3 and 10/2 \= 5).  
   The test cases are generated so that there will be an answer.

     
   

   Same as above problem just different solution function.  
   ```python
from math import ceil

def smallestDivisor(self, nums: List[int], threshold: int) -> int:
    low = 1
    high = max(nums)
    ans = high

    def getSum(d):
        return sum(ceil(x / d) for x in nums)

    while low <= high:
        mid = (low + high) // 2
        sum_ = getSum(mid)
        if sum_ <= threshold:
            ans = min(mid, ans)
            high = mid - 1
        else:
            low = mid + 1
    return ans
```

---
   Or also 
   ```python
   def getSum(d):  
               return sum(ceil(x/d) for x in nums)  
               ```
   This also get the same answer.  
     
8. **Minimum Garden Perimeter to collect NeededApples**  
   In a garden represented as an infinite 2D grid, there is an apple tree planted at **every** integer coordinate. The apple tree planted at an integer coordinate (i, j) has |i| \+ |j| apples growing on it.  
   You will buy an axis-aligned **square plot** of land that is centered at (0, 0).  
   Given an integer neededApples, return *the **minimum perimeter** of a plot such that **at least*** neededApples *apples are **inside or on** the perimeter of that plot*.  
     
   All things are the same as the above problems. We just have to formulate how to get a solution and use it in our solution functions.  
   

   ### **🍎 Apples per Layer:**

The **`k-th layer`** (a ring at distance `k`) contributes:  
	Apples at layer k=12k2  
	  
Every square layer of side `2k` has:

* 8 corners: each contributes `k` apples  
* 4 sides: each side (excluding corners) has `2k - 1` points, and the apple count increases linearly with distance from center.

  ### **🍎 Total Apples for n Layers:**

Now, sum the apples from layer 1 to `n`:  
	TotalApples(n)= Σ12k2  
Using sum of squared formula it simplifies to  
 12 \* (n(n+1)(2n+1))/6 \= 2n(n+1)(2n+1)

	```python
def minimumPerimeter(self, neededApples: int) -> int:
    low = 1
    high = 100000

    def getSol(n):
        return 2 * n * (n + 1) * (2 * n + 1)

    edge = high
    while low <= high:
        mid = (low + high) // 2
        apples = getSol(mid)
        if apples >= neededApples:
            edge = min(edge, mid)
            high = mid - 1
        else:
            low = mid + 1
    return 8 * edge
```

---

The time complexity is O(logn)  where maximum n is 100000.

9. **Longest Repeating Character**  
   You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.  
   Return *the length of the longest substring containing the same letter you can get after performing the above operations*.  
   

	The Binary search approach of the above problem is as follows.  
	```python
from collections import defaultdict

def characterReplacement(self, s: str, k: int) -> int:
    low, high = 1, len(s)

    def getSol(windowSize):
        mp = defaultdict(int)
        left = right = 0
        while right < len(s):
            mp[s[right]] += 1
            if right - left + 1 == windowSize:
                maxi = max(mp.values())
                if windowSize - maxi <= k:
                    return True
                mp[s[left]] -= 1
                left += 1
            right += 1
        return False

    ans = low
    while low <= high:
        mid = (low + high) // 2
        if getSol(mid):
            ans = max(ans, mid)
            low = mid + 1
        else:
            high = mid - 1
    return ans
```

Another approach is pure sliding window with slightly better time complexity as follows.
```python
from collections import defaultdict

def characterReplacement(self, s: str, k: int) -> int:
    wind = defaultdict(int)
    l = 0
    res = 0
    max_count = 0
    for r, c in enumerate(s):
        wind[c] += 1
        max_count = max(max_count, wind[c])
        if r - l + 1 - max_count > k:
            wind[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1)
    return res
```

---
The time complexity is O(nlogn) & O(n) respectively.

10. **132 Pattern**  
    Given an array of n integers nums, a **132 pattern** is a subsequence of three integers nums\[i\], nums\[j\] and nums\[k\] such that i \< j \< k and nums\[i\] \< nums\[k\] \< nums\[j\].  
    Return true *if there is a **132 pattern** in* nums*, otherwise, return* false*.*  
      
    I will store the minimum number present in the left of the current index and we will find our candidate that is greater than my current minimum but less than my current element present in right.  
    I will sort the right part and will do a binary search for bounds.  
    ```python
from bisect import bisect_left, insort
from typing import List

def find132pattern(self, nums: List[int]) -> bool:
    if len(nums) < 3:
        return False
    mini = [0 for _ in range(len(nums))]
    mini[0] = nums[0]
    for i in range(1, len(nums)):
        mini[i] = min(mini[i - 1], nums[i])
    sorted_nums = []
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] > mini[i]:
            ind = bisect_left(sorted_nums, mini[i] + 1)
            if ind < len(sorted_nums) and sorted_nums[ind] < nums[i]:
                return True
            insort(sorted_nums, nums[i])
    return False
```
    This has a time complexity of O(nlogn).