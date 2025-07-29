### Next Greater Numerically Balanced Number
An integer x is numerically balanced if for every digit d in the number x, there are exactly d occurrences of that digit in x.
Given an integer n, return the smallest numerically balanced number strictly greater than n.

```python
class Solution:
    def nextBeautifulNumber(self, n: int) -> int:
        #hardcoded base permutation of balanced number under constraint
        seeds = ["122", "1333", "14444", "155555", "22333", "224444", "122333"]
        res = [1,22, 333, 4444, 55555, 666666, 1224444]
        #backtracking funtion to genrate permutation of balanced number base
        def backtrack(s, path, visited, result):
            if len(path) == len(s):
                num = int("".join(path))
                result.append(num)
                return
            for i in range(len(s)):
                if visited[i]:
                    continue
                if i > 0 and s[i] == s[i - 1] and not visited[i - 1]:
                    continue
                visited[i] = True
                path.append(s[i])
                backtrack(s, path, visited, result)
                path.pop()
                visited[i] = False
        #generating all balanced number permutation
        for seed in seeds:
            visited = [False] * len(seed)
            backtrack(list(seed), [], visited, res)
        #binary search implementation of bisect_right
        def bs(arr, target):
            left = 0
            right = len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            return left
        #sorting to do binary search
        res.sort()
        #returning next greater balanced number
        return res[bs(res, n)]
```
### Choose Edges to Maximize Score in a Tree
You're given a rooted tree with n nodes (0 to n − 1).

Represented by edges array: edges[i] = [par_i, weight_i], where par_i is the parent of node i and weight_i is the weight of that edge. Root node has edges[0] = [-1, -1].

Select a subset of edges so that no two selected edges share a node (i.e. are adjacent), and the sum of their weights is maximized.

Return that maximum sum. Weights can be negative, and choosing no edge yields 0.

```python
class Solution:
    def maxScore(self, edges: List[List[int]]) -> int:
        n = len(edges)

        # Step 1: Build the tree as an adjacency list
        tree = [[] for _ in range(n)]
        for child in range(1, n):
            parent, weight = edges[child]
            tree[parent].append((child, weight))

        # Step 2: DFS function to return (take[u], notTake[u])
        def dfs(u):
            notTake_u = 0  # max score if we don't take any edge from u to its children
            deltas = []    # possible benefits if we do take an edge (u–v)

            # Step 3: Process all children of u
            for v, w in tree[u]:
                take_v, notTake_v = dfs(v)

                # Always add take[v] to notTake[u] (default plan: don't pick any u–v edge)
                notTake_u += take_v

                # If we *do* take edge (u–v), then we gain `w + notTake[v]` instead of `take[v]`
                # So the net improvement compared to notTake_u is:
                delta = w + notTake_v - take_v
                deltas.append(delta)

            # Step 4: Try taking one edge (u–v), choose the one that gives max benefit
            bestDelta = max(deltas, default=float('-inf'))

            # If taking one edge improves the total, use it
            take_u = notTake_u + max(0, bestDelta)

            return (take_u, notTake_u)

        # Step 5: Run DFS from the root node (0)
        take0, notTake0 = dfs(0)

        # Step 6: Return the max of taking or not taking at root
        return max(take0, notTake0)
```

### Number of Same‑End Substrings
You’re given a string s (lowercase letters) and multiple queries [l, r]. For each query, consider the substring s[l..r]. You need to count how many substrings within that range start and end with the same character, including single-character substrings.

Preprocess:
Build a prefix sum array cnt[char][i] that counts how many times each character appears in s[0..i−1]. (26 arrays, one per letter)
Answer each query:
For query [l, r], get the frequency k = cnt[c][r+1] − cnt[c][l] for each character c.
Sum up k*(k + 1)/2 over all 26 letters.
Append that sum to results.

```cpp
vector<long long> solve(string s, vector<vector<int>>& queries) {
    int n = s.size();
    vector<vector<int>> cnt(26, vector<int>(n+1, 0));
    for (int i = 1; i <= n; i++) {
        for (int c = 0; c < 26; c++)
            cnt[c][i] = cnt[c][i-1];
        cnt[s[i-1]-'a'][i]++;
    }
    vector<long long> ans;
    for (auto &q : queries) {
        int l = q[0], r = q[1];
        long long total = 0;
        for (int c = 0; c < 26; c++) {
            long long k = cnt[c][r+1] - cnt[c][l];
            total += k * (k + 1) / 2;
        }
        ans.push_back(total);
    }
    return ans;
}
```
### Single Element in a Sorted Array
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once.

Return the single element that appears only once.

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left,right=0,len(nums)-1
        #odd sized array
        while left < right:
            mid=(left+right) // 2
            # if mid is odd make it even 
            if mid%2 == 1:
                mid-=1
            #pattern [even1 odd1 even2 odd2]
            #if even1 and odd1 is not same then pairity is disturbed in left side
            if nums[mid] != nums[mid+1]:
                right=mid
                #search left
            else:
                left=mid+2
                #search right
        return nums[left]
```