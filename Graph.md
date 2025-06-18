Report of Target 2025

# GRAPH 

1. Rotting Oranges  
   You are given an m x n grid where each cell can have one of three values:  
* 0 representing an empty cell,  
* 1 representing a fresh orange, or  
* 2 representing a rotten orange.  
  Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.  
  Return *the minimum number of minutes that must elapse until no cell has a fresh orange*. If *this is impossible, return* \-1.


	Direct level wise BFS implementation.  
```python
def orangesRotting(self, grid: List[List[int]]) -> int:  
        m=len(grid)  
        n=len(grid[0])  
        def valid(i,j):  
            if i<0 or i>=m :  
                return False  
            if j<0 or j>=n :  
                return False  
            return True  
        q=deque()  
        for i  in range(m):  
            for j in range(n):  
                if(grid[i][j]==2):  
                    q.append((i,j,0))  
        ans=0  
        while q:  
            i,j,t=q.popleft()  
            ans=max(ans,t)  
            if valid(i+1,j):  
                if grid[i+1][j]==1:  
                    grid[i+1][j]=2  
                    q.append((i+1,j,t+1))  
            if valid(i,j+1):  
                if grid[i][j+1]==1:  
                    grid[i][j+1]=2  
                    q.append((i,j+1,t+1))  
            if valid(i-1,j):  
                if grid[i-1][j]==1:  
                    grid[i-1][j]=2  
                    q.append((i-1,j,t+1))  
            if valid(i,j-1):  
                if grid[i][j-1]==1:  
                    grid[i][j-1]=2  
                    q.append((i,j-1,t+1))  
        for i in range(m):  
            for j in range(n):  
                if grid[i][j]==1 :  
                    return -1  
        return ans  
```
If direction is valid in the grid then append it into the queue with incremented time.  
Time complexity is O(mn).

2. **Number of Islands**  
   Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return *the number of islands*.  
   An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water  
   Direct implementation of multiple dfs count.
   ```python  
   def numIslands(self, grid: List[List[str]]) -> int:  
           m=len(grid)  
           n=len(grid[0])  
           def valid(i,j):  
               if i<0 or i>=m :  
                   return False  
               if j<0 or j>=n :  
                   return False  
               return True  
           vis=set()  
     
           def dfs(i,j):  
               if valid(i,j):  
                   if (i,j) not in vis:  
                       if grid[i][j]=="1":        
                           vis.add((i,j))  
                           dfs(i+1,j)  
                           dfs(i-1,j)  
                           dfs(i,j+1)  
                           dfs(i,j-1)  
           ans=0  
           for i in range(m):  
               for j in range(n):  
                   if grid[i][j]=="1":  
                       if (i,j) not in vis:  
                           dfs(i,j)  
                           ans+=1  
           return ans  
    ```
   Time complexity is O(mn).  
     
3. **Number of Enclaves**  
   You are given an m x n binary matrix grid, where 0 represents a sea cell and 1 represents a land cell.  
   A **move** consists of walking from one land cell to another adjacent (**4-directionally**) land cell or walking off the boundary of the grid.  
   Return *the number of land cells in* grid *for which we cannot walk off the boundary of the grid in any number of **moves***.  
     
   Reverse Bfs from outside of the grid. I am making a graph matrix with extending the grid boundary as \-1. From all \-1 I am running the bfs until queue ends and we get only the remaining 1s which cant be covered by the border.  
    ```python
   def numEnclaves(self, grid: List[List[int]]) -> int:  
           m=len(grid)  
           n=len(grid[0])  
           graph = [[-1 for _ in range(n+2)] for _ in range(m+2)]  
           for i in range(m):  
               for j in range(n):  
                   graph[i+1][j+1]=grid[i][j]  
           m+=2  
           n+=2  
           def valid(i,j):  
               if i<0 or i>=m :  
                   return False  
               if j<0 or j>=n :  
                   return False  
               return True  
           q=deque()  
           for i  in range(m):  
               for j in range(n):  
                   if(graph[i][j]==-1):  
                       q.append((i,j))  
           while q:  
               i,j=q.popleft()  
               if valid(i+1,j):  
                   if graph[i+1][j]==1:  
                       graph[i+1][j]=-1  
                       q.append((i+1,j))  
               if valid(i,j+1):  
                   if graph[i][j+1]==1:  
                       graph[i][j+1]=-1  
                       q.append((i,j+1))  
               if valid(i-1,j):  
                   if graph[i-1][j]==1:  
                       graph[i-1][j]=-1  
                       q.append((i-1,j))  
               if valid(i,j-1):  
                   if graph[i][j-1]==1:  
                       graph[i][j-1]=-1  
                       q.append((i,j-1))  
           ans=0  
           for i  in range(m):  
               for j in range(n):  
                   if(graph[i][j]==1):  
                       ans+=1  
           return ans  
    ```
   Time complexity is O(mn).  
     
4. **Flood Fill**  
   You are given an image represented by an m x n grid of integers image, where image\[i\]\[j\] represents the pixel value of the image. You are also given three integers sr, sc, and color. Your task is to perform a **flood fill** on the image starting from the pixel image\[sr\]\[sc\].  
   To perform a **flood fill**:  
1. Begin with the starting pixel and change its color to color.  
2. Perform the same process for each pixel that is **directly adjacent** (pixels that share a side with the original pixel, either horizontally or vertically) and shares the **same color** as the starting pixel.  
3. Keep **repeating** this process by checking neighboring pixels of the *updated* pixels and modifying their color if it matches the original color of the starting pixel.  
4. The process **stops** when there are **no more** adjacent pixels of the original color to update.  
   Return the **modified** image after performing the flood fill.  
   Direct implementation of colour based DFS.  
   ```python
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:  
           m=len(image)  
           n=len(image[0])  
           def valid(i,j):  
               if i<0 or i>=m :  
                   return False  
               if j<0 or j>=n :  
                   return False  
               return True  
           vis=set()  
           main_clr=image[sr][sc]  
           def dfs(i,j):  
               if valid(i,j):  
                   if (i,j) not in vis:  
                       if image[i][j]==main_clr:  
                           image[i][j]=color  
                           vis.add((i,j))  
                           dfs(i+1,j)  
                           dfs(i,j+1)  
                           dfs(i-1,j)  
                           dfs(i,j-1)  
           dfs(sr,sc)  
           return image  
    ```
   Time complexity is O(mn).  
     
5. **Course Schedule**   
   There are a total of numCourses courses you have to take, labeled from 0 to numCourses \- 1\. You are given an array prerequisites where prerequisites\[i\] \= \[ai, bi\] indicates that you **must** take course bi first if you want to take course ai.  
* For example, the pair \[0, 1\], indicates that to take course 0 you have to first take course 1\.  
  Return *the ordering of courses you should take to finish all courses*. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return **an empty array**.  
  Direct implementation of topological sort using Kahn's Algorithm.  
    ```python
  def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:  
          adj=[[] for _ in range(numCourses)]  
          for edge in prerequisites:  
              adj[edge[0]].append(edge[1])  
          indegree=[0 for _ in range(numCourses)]  
          for u in range(numCourses):  
              for v in adj[u]:  
                  indegree[v]+=1  
          q=deque()  
          for node,degree in enumerate(indegree):  
              if degree==0:  
                  q.append(node)  
          topo=[]  
          while q:  
              node=q.popleft()  
              topo.append(node)  
              for v in adj[node]:  
                  indegree[v]-=1  
                  if indegree[v]==0 :  
                      q.append(v)  
          topo.reverse()  
          return topo if len(topo)==numCourses else [] 
    ``` 
  The time complexity is O(numCourses \+ len(Prerequisites)).  
    
  Another approach using Stack and DFS where I am checking cycle using tricoloured flag 0 means not visited , 1 means visiting and 2 visited successfully. And in between if there is 1 then it means there is cycle as  I again reached 1 withing completing that node in dfs.  
    
    ```python
  def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:  
          adj=[[] for _ in range(numCourses)]  
          for edge in prerequisites:  
              adj[edge[0]].append(edge[1])  
    
          vis=[0]*numCourses  
          stk=deque()  
          cycle=False  
    
          def dfs(node):  
              nonlocal cycle  
              if vis[node]==1:  
                  cycle=True  
                  return  
              if vis[node]==2:  
                  return  
              vis[node]=1  
              for v in adj[node]:  
                  dfs(v)  
              vis[node]=2  
              stk.append(node)  
    
          for node in range(numCourses):  
              if vis[node]==0:  
                  dfs(node)  
          topo=list(stk)  
          return topo if not cycle else []  
    ```
    
  With again Time complexity of O(numCourses \+ len(Prerequisites))  
    
6. **Clone Graph**  
   Given a reference of a node in a [**connected**](https://en.wikipedia.org/wiki/Connectivity_\(graph_theory\)#Connected_graph) undirected graph.  
   Return a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) (clone) of the graph.  
   Each node in the graph contains a value (int) and a list (List\[Node\]) of its neighbors.

Same Implementation as linked list deep copy using hashmap and visited set by the help of recursion.
```python
"""  
# Definition for a Node.  
class Node:  
    def __init__(self, val = 0, neighbors = None):  
        self.val = val  
        self.neighbors = neighbors if neighbors is not None else []  
"""

class Solution:  
    def cloneGraph(self, source: Optional['Node']) -> Optional['Node']:  
        if not source:  
            return None  
        mp=defaultdict(Node)  
        vis=set()  
        def rec(node):  
            vis.add(node)  
            temp=Node(node.val)  
            mp[node]=temp  
            for neigh  in node.neighbors:  
                if neigh not in vis:  
                    rec(neigh)  
        rec(source)  
        for node in mp.keys():  
            n=[]  
            for neigh in node.neighbors:  
                n.append(mp[neigh])  
            mp[node].neighbors=n  
        return mp[source]  
```
With Time complexity of O(N+E) and space complexity of O(N).

7. **Is Graph Bipartite**  
     
   There is an **undirected** graph with n nodes, where each node is numbered between 0 and n \- 1. You are given a 2D array graph, where graph\[u\] is an array of nodes that node u is adjacent to. More formally, for each v in graph\[u\], there is an undirected edge between node u and node v.  
   A graph is **bipartite** if the nodes can be partitioned into two independent sets A and B such that **every** edge in the graph connects a node in set A and a node in set B.  
   Return true *if and only if it is **bipartite***  
   So it's a color matching problem for each  node. If it has the same color as parent then it's false we will color the node with opposite color  of parent using DFS.  
    ```python
   def isBipartite(self, graph: List[List[int]]) -> bool:  
           n=len(graph)  
           color=[-1]*n  
           def dfs(node,parent):  
               if color[parent]==-1:  
                   color[node]=1  
               else:  
                   color[node]=not color[parent]  
               for v in graph[node]:  
                   if color[v]==-1:  
                       ans=dfs(v,node)  
                       if not ans:  
                           return False  
                   elif color[v]==color[node]:  
                       return False  
               return True  
           for i in range(n):  
               if color[i]==-1:  
                   temp=dfs(i,-1)  
                   if not temp:  
                       return temp  
           return True
    ```

	\-1: Not colored 0: Color0 1: Color1  
If the Sub problem is false then overall false in that we are not exploring while set.  
The time complexity is O(V+E) and O(V) for color array.

8. **Alien Dictionary**  
   A new alien language uses the English alphabet, but the order of letters is unknown. You are given a list of words\[\] from the alien language’s dictionary, where the words are claimed to be sorted lexicographically according to the language’s rules.  
   Your task is to determine the correct order of letters in this alien language based on the given words. If the order is valid, return a string containing the unique letters in lexicographically increasing order as per the new language's rules. If there are multiple valid orders, return any one of them.  
   However, if the given arrangement of words is inconsistent with any possible letter ordering, return an empty string ("").  
     
   So I have to do topological ordering here but the main problem is that how to build that graph.  
   Now to build the graph I have to find out the relationship between words.  
   Now The words which came first in the list are lexicographically smaller. So for two consecutive words the character which is different is the key for lexicography. Character that changes, appear in the first word will have a directed edge toward the second word's different character.  
   ```python
   from collections import deque  
   class Solution:  
       def findOrder(words):  
           adj=[[] for _ in range(26)]  
           indegree = [0 for _ in range(26)]  
           avail=[False for _ in range(26)]  
           for word in words:  
               for c in word:  
                   avail[ord(c)-97]=True  
           for i in range(1,len(words)):  
               j=0  
               minlen=min(len(words[i-1]),len(words[i]))  
               while(j<minlen and words[i-1][j]==words[i][j]):  
                   j+=1  
               if j<minlen:  
                   adj[ord(words[i-1][j])-97].append(ord(words[i][j])-97)  
                   indegree[ord(words[i][j])-97]+=1  
               elif len(words[i-1])>len(words[i]):  
                   return ''  
           n=sum(avail)  
           q=deque()  
           for i in range(len(indegree)):  
               if avail[i]:  
                   if indegree[i]==0:  
                       q.append(i)  
           topo=[]  
           while q:  
               node=q.popleft()  
               topo.append(node)  
               for v in adj[node]:  
                   indegree[v]-=1  
                   if indegree[v]==0:  
                       q.append(v)  
        return ''.join([chr(c + 97) for c in topo]) if len(topo) == n else ""  
    ```
   Now the edge case is if the order given words is wrong, that is len(words\[i-1\])\>len(words\[i\]) then it will be an empty string and if the topo list is not the same as n then again empty string is the answer.  
     
9. **Word Ladder**  
   A **transformation sequence** from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord \-\> s1 \-\> s2 \-\> ... \-\> sk such that:  
* Every adjacent pair of words differs by a single letter.  
* Every si for 1 \<= i \<= k is in wordList. Note that beginWord does not need to be in wordList.  
* sk \== endWord  
  Given two words, beginWord and endWord, and a dictionary wordList, return *the **number of words** in the **shortest transformation sequence** from* beginWord *to* endWord*, or* 0 *if no such sequence exists.*  
  So my flow is to find whether I can find a word that is there in the wordlist by changing exactly one character.  
  For each level I will find all the possible words and so on if that is present in the wordlist.

  And to this level wise I will simulate this process using queue BFS.  
  And for each level I will increment change by 1 and return changes if I reach the endWord.

  And edge case of this problem is if endWord not in the WordList in that case return 0 directly.


```python
  def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:  
          words = set(wordList)  
          if endWord not in words:  
              return 0  
    
          q = deque()  
          q.append(beginWord)  
          vis = set()  
          vis.add(beginWord)  
    
          changes = 1  
          while q:  
              for _ in range(len(q)):  
                  word = q.popleft()  
                  if word == endWord:  
                      return changes  
                  for i in range(len(word)):  
                      for c in 'abcdefghijklmnopqrstuvwxyz':  
                          newWord = word[:i] + c + word[i+1:]  
                          if newWord in words and newWord not in vis:  
                              vis.add(newWord)  
                              q.append(newWord)  
              changes += 1  
          return 0  
```
  The worst case time complexity is O(len(wordList)\*len(word)2)  
    
10. **Word Ladder II**  
    A **transformation sequence** from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord \-\> s1 \-\> s2 \-\> ... \-\> sk such that:  
* Every adjacent pair of words differs by a single letter.  
* Every si for 1 \<= i \<= k is in wordList. Note that beginWord does not need to be in wordList.  
* sk \== endWord  
  Given two words, beginWord and endWord, and a dictionary wordList, return *all the **shortest transformation sequences** from* beginWord *to* endWord*, or an empty list if no such sequence exists. Each sequence should be returned as a list of the words* \[beginWord, s1, s2, ..., sk\].  
    
  Now same as above I have to build the exact shortest path. Now for this I have to build adjMap. And to build graph I will use the word ladder I approach of finding path using bfs.

```python
    words=set(wordList)

          if endWord not in words:  
              return [] 
```
  Made the set of words and handled the edge case.
```python
    graph = defaultdict(list)

          level={beginWord}  
          vis=set()  
          found=False  
```
  Initialized all the variables required to build the graph.
  ```python  
  while level and not found:  
              next_level=set()  
              vis |= level # Mark all current level nodes as visited 
``` 
  For every word at this level:  
* Generate **one-letter different words.**  
* If a valid word is found (in wordSet and not visited), it’s a **valid neighbor**  
    ```python
              for word in level:  
                  for i in range(len(word)):  
                      for c in "abcdefghijklmnopqrstuvwxyz":  
                          newWord=word[:i]+c+word[i+1:]  
                          if newWord in words and newWord not in vis:  
                              graph[word].append(newWord)  
                              if newWord==endWord:  
                                  found=True  
                              next_level.add(newWord)  
              level=next_level  
        ```
  Graph building phase using BFS.  
  Example Graph:
  ```python  
  graph = { "hit": ["hot"], "hot": ["dot", "lot"], "dot": ["dog"], "lot": ["log"], "dog": ["cog"], "log": ["cog"]}

    ans=[]

          def dfs(path,word):  
              if word==endWord:  
                  ans.append(path[:])  
              for v in graph[word]:  
                  path.append(v)  
                  dfs(path,v)  
                  path.pop()  
          dfs([beginWord],beginWord)  
          return ans  
    ```
  Now DFS backtracks to create the path.  
  The worst case time complexity is O(len(wordList)\*len(word)2+PossiblePaths)  
    
  **\>\>Dijkstra:**  
   ```python 
  from queue import PriorityQueue  
  class Solution:  
      def dijkstra(self, V, edges, src):  
          adj=[[]for _ in range(V)]  
          for u,v,w in edges:  
              adj[u].append((v,w))  
              adj[v].append((u,w))  
          pq=PriorityQueue()  
          dist=[float('inf') for _ in range(V)]  
          dist[src]=0  
          pq.put((0,src))  
          while not pq.empty():  
              dis,node=pq.get()  
              for v,w in adj[node]:  
                  if dis+w<dist[v]:  
                      dist[v]=dis+w  
                      pq.put((dist[v],v))  
          for i in range(V):  
              if dist[i]==float('inf'):  
                  dist[i]=-1  
          return dist  
    ```
               
11. **Cheapest Flights Within K stops**  
    There are n cities connected by some number of flights. You are given an array flights where flights\[i\] \= \[fromi, toi, pricei\] indicates that there is a flight from city fromi to city toi with cost pricei.  
    You are also given three integers src, dst, and k, return ***the cheapest price** from* src *to* dst *with at most* k *stops.* If there is no such route, return \-1.  
      
      
    It is level wise BFS using weighted distance relaxation using pruning of branches where we cant reach using less that equals to k stops.  
    ```python
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:  
            q=deque()  
            dist=[float('inf') for _ in range(n)]  
            adj=[[] for _ in range(n)]  
            for edge in flights:  
                adj[edge[0]].append((edge[1],edge[2]))  
            dist[src]=0  
            q.append((0,src,0))  
            while q:  
                stops,node,dis=q.popleft()  
                if stops>k:  
                    continue  
                for v,w in adj[node]:  
                    if dis+w<dist[v] and stops<=k :  
                        dist[v]=dis+w  
                        q.append((stops+1,v,dist[v]))  
            return -1 if dist[dst]==float('inf') else dist[dst]  
    ```
    This gives us the desired output that is weight relaxation like dijkstra but not  priority wise queuing.  
      
12. **Network Delay Time**  
    You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times\[i\] \= (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.  
    We will send a signal from a given node k. Return *the **minimum** time it takes for all the* n *nodes to receive the signal*. If it is impossible for all the n nodes to receive the signal, return \-1  
    Direct implementation of **Bellman Ford**  Algorithm .  
    ```python
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:  
            dist=[float('inf') for _ in range(n)]  
            dist[k-1]=0  
            for _ in range(n):  
                for u,v,w in times:  
                    if dist[u-1]+w<dist[v-1]:  
                        dist[v-1]=dist[u-1]+w  
            if float('inf') in dist:  
                return -1  
            return max(dist)
    ```  
13. **Number of Ways to Arrive at Destination**   
    You are in a city that consists of n intersections numbered from 0 to n \- 1 with **bi-directional** roads between some intersections. The inputs are generated such that you can reach any intersection from any other intersection and that there is at most one road between any two intersections.  
    You are given an integer n and a 2D integer array roads where roads\[i\] \= \[ui, vi, timei\] means that there is a road between intersections ui and vi that takes timei minutes to travel. You want to know in how many ways you can travel from intersection 0 to intersection n \- 1 in the **shortest amount of time**.  
    Return *the **number of ways** you can arrive at your destination in the **shortest amount of time***. Since the answer may be large, return it **modulo** 109 \+ 7\.  
      
      
    Implementation of Flyodd Warshal’s Dynamic Programming approach using additional argument of number of paths.  
    Initialize a 3D DP table dp\[n\]\[n\]\[2\] where: dp\[src\]\[dest\]\[0\] stores the minimum time to reach dest from src. dp\[src\]\[dst\]\[1\] stores the number of ways to achieve the minimum time.  
      ```python
     def countPaths(self, n: int, roads: List[List[int]]) -> int:  
            dp=[[[0 for _ in range(2)] for __ in range(n)] for ___ in range(n)]  
            #dp[src][dst][time] and dp[src][dst][ways]  
      
            for src in range(n):  
                for dst in range(n):  
                    if src!=dst :  
                        dp[src][dst][0]=int(1e12) #time  
                        dp[src][dst][1]=0 #ways  
                    else:  
                        dp[src][src][0]=0 #already there 0 time  
                        dp[src][src][1]=1 #trivial 1 way stay there  
            for u,v,t in roads:  
                dp[u][v][0]=t  
                dp[v][u][0]=t  
                #time is given and way is 1  
                dp[u][v][1]=1  
                dp[v][u][1]=1  
            for via in range(n):  
                for src in range(n):  
                    for dst in range(n):  
                        if src!=via and dst!=via:  
                            newTime=dp[src][via][0]+dp[via][dst][0]  
                            if newTime<dp[src][dst][0]:  
                                dp[src][dst][0]=newTime  
                                #newPath shortest path Found  
                                #discard old and replace with via paths comnbinations  
                                dp[src][dst][1]=(dp[src][via][1]*dp[via][dst][1])%MOD  
                            elif newTime==dp[src][dst][0]:  
                                #another path found with same time  
                                #add new via path Combinations to old paths  
                                dp[src][dst][1]=(dp[src][dst][1]+dp[src][via][1]*dp[via][dst][1])%MOD  
            return dp[n-1][0][1] 
    ``` 
    Time complexity of this approach is O(n3)  
      
    The Second approach is using Dijkstra with dynamic programming to store number of paths.  
      ```python
    def countPaths(self, n: int, roads: List[List[int]]) -> int:  
            adj=[[] for _ in range(n)]  
            for u,v,t in roads:  
                adj[u].append((v,t))  
                adj[v].append((u,t))  
            dist=[float('inf') for _ in range(n)]  
            path=[0 for _ in range(n)]  
            dist[0]=0  
            path[0]=1  
            pq=PriorityQueue()  
            pq.put((0,0))  
            while not pq.empty():  
                time,node=pq.get()  
                for v,t in adj[node]:  
                    if t+time<dist[v]:  
                        dist[v]=t+time  
                        path[v]=path[node]  
                        pq.put((dist[v],v))  
                    elif t+time == dist[v]:  
                        path[v]=(path[v]+path[node])%MOD  
            return path[n-1] 
    ``` 
    If time reduces then update path otherwise increment the path same as previous implementation. This approach is having time complexity of O((m+n)logn).  
    **\>\>Flyodd Warshal:**  
    ```python
    	dist=[[10**5 for _ in range(n)] for _ in range(n)]  
            for u,v,w in edges:  
                dist[u][v]=w  
                dist[v][u]=w  
            for node in range(n):  
                dist[node][node]=0  
            for via in range(n):  
                for src in range(n):  
                    for dst in range(n):  
                        dist[src][dst]=min(  
                            dist[src][dst],  
                            dist[src][via]+dist[via][dst]  
                            )  
      ```
             
    **\>\> Disjoint Set Union**   
    ```python
    class DSU:  
       def __init__(self,n):  
           self.parent=[i for i in range(n)]  
           self.rank=[0 for _ in range(n)]  
           self.size=[1 for _ in range(n)]  
      
       def find(self,node):  
           if self.parent[node]!=node:  
               self.parent[node]=self.find(self.parent[node])  
           return self.parent[node]  
      
       def unionByRank(self,x,y):  
           px,py=self.find(x),self.find(y)  
           if px==py:  
               return False  
           if self.rank[px]<self.rank[py]:  
               self.parent[px]=py  
           elif self.rank[px]>self.rank[py]:  
               self.parent[py]=px  
           else:  
               self.parent[py]=px  
               self.rank[px]+=1  
           return True  
      
      
       def unionBySize(self,x,y):  
           px,py=self.find(x),self.find(y)  
           if px==py:  
               return False  
           if self.size[px]<self.size[py]:  
               self.parent[px]=py  
               self.size[py]+=self.size[px]  
           else:  
               self.parent[py]=px  
               self.size[px]+=self.size[py]  
           return True  
    ```
14. **Number of operations to make Network Connected**  
    There are n computers numbered from 0 to n \- 1 connected by ethernet cables connections forming a network where connections\[i\] \= \[ai, bi\] represents a connection between computers ai and bi. Any computer can reach any other computer directly or indirectly through the network.  
    You are given an initial computer network connections. You can extract certain cables between two directly connected computers, and place them between any pair of disconnected computers to make them directly connected.  
    Return *the minimum number of times you need to do this in order to make all the computers connected*. If it is not possible, return \-1.  
      
      
    This is direct implementation of DSU whenever we connect a new element in disjoint first time increment the count and answer is going to be minimum edges to connect minus the counts.  
      ```python
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:  
           dsu=DSU(n)  
           ans=0  
           have=0  
           if len(connections)<n-1:  
               return -1  
           for u,v in connections:  
               if dsu.unionByRank(u,v):  
                   ans+=1  
           return n-ans-1 
    ``` 
15. **Number of Islands II**  
    You are given a n,m which means the row and column of the 2D matrix and an array of  size k denoting the number of operations. Matrix elements is 0 if there is water or 1 if there is land. Originally, the 2D matrix is all 0 which means there is no land in the matrix. The array has k operator(s) and each operator has two integer A\[i\]\[0\], A\[i\]\[1\] means that you can change the cell matrix\[A\[i\]\[0\]\]\[A\[i\]\[1\]\] from sea to island. Return how many island are there in the matrix after each operation.You need to return an array of size k.  
    Note : An island means group of 1s such that they share a common side.  
      ```python
    def numOfIslands(self, rows: int, cols: int, operators: List[List[int]]) -> List[int]:  
           def dimRed(i, j):  
               return i * cols + j  
      
           dsu = DSU(rows * cols)  
           visited = [[0] * cols for _ in range(rows)]  
           ans = []  
           count = 0  
           for i, j in operators:  
               if visited[i][j]:  
                   ans.append(count)  
                   continue  
               visited[i][j] = 1  
               count += 1  
               index = dimRed(i, j)  
               for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  
                   ni, nj = i + dx, j + dy  
                   if 0 <= ni < rows and 0 <= nj < cols and visited[ni][nj]:  
                       neighbor_index = dimRed(ni, nj)  
                       if dsu.union(index, neighbor_index):  
                           count -= 1   
               ans.append(count)  
           return ans  
    ```
    Another implementation of DSU where nodes are flattened Dimension Reduced index of matrix.  
    If Union is made then we reduce the existing island count.  
      
16. **Making a Large Island**  
    You are given an n x n binary matrix grid. You are allowed to change **at most one** 0 to be 1\.  
    Return *the size of the largest **island** in* grid *after applying this operation*.  
    An **island** is a 4-directionally connected group of 1s.  
      
      
    I am going to do this problem in three step , first union the nodes that are adjacent, second find parent and its size , and then lastly the edge case handling of all ones or initial max island configuration.  
      
    

\# Step 1 : Making Union of 1s 
```python 
        for i in range(n):  
            for j in range(n):  
                if grid[i][j] == 1:  
                    for dx, dy in directions:  
                        if isvalid(i+dx, j+dy) and grid[i + dx][j + dy] == 1:  
                            dsu.union(RedDim(i, j), RedDim(i + dx, j + dy)) 
``` 
\# Step 2 : Maximizing the Large Islands using 0s  
```python
        maxi = 0  
        for i in range(n):  
            for j in range(n):  
                if grid[i][j] == 0:  
                    comps = set()  
                    for dx, dy in directions:  
                        if isvalid(i+dx,j+dy) and grid[i + dx][j + dy] == 1:  
                            comps.add(dsu.find(RedDim(i + dx, j + dy)))  
                    size = 0  
                    for node in comps:  
                        size += dsu.getSize(node)  
                    maxi = max(size + 1, maxi)
```
\# Step 3 : Edge case of having max in initial comps only  
```python
        for i in range(n * n):  
            maxi = max(maxi, dsu.getSize(i))  
        return maxi  
    
	The time complexity is O(n2).

17. **Most Stones Removed with Same Row or Column**  
    On a 2D plane, we place n stones at some integer coordinate points. Each coordinate point may have at most one stone.  
    A stone can be removed if it shares either **the same row or the same column** as another stone that has not been removed.  
    Given an array stones of length n where stones[i] = [xi, yi] represents the location of the ith stone, return *the largest possible number of stones that can be removed*.  
      
      
    I am Considering each row and column as nodes of a graph and connect if there is a stone like for (X,Y) connect nodeX with nodeY . Now Numbering of nodes is : for rows it's the same as row numbers but for columns we are serializing it in straight line and adding cols after the rows end.  
    And finally we have to calculate all the connected components then our desired answer is   
    RemovableStones = NumStones- ConnectedComps.  
    Algo:  
1. Serialize each `(x, y)` into two DSU nodes: `x`, and `maxRow + y + 1`  
2. Use DSU `union(x, y_serialized)` for each stone  
3. Track all unique nodes seen  
4. Count how many unique parents exist (`find(node) == node`)  
5. Return `n - components` 
```python 
   def removeStones(self, stones: List\[List\[int\]\]) \-\> int:  
           n \= len(stones)  
           maxRow \= 0  
           maxCol \= 0  
           for x, y in stones:  
               maxRow \= max(maxRow, x)  
               maxCol \= max(maxCol, y)  
            
           dsu \= DSU(maxRow \+ maxCol \+ \+2)  
           unqNodes \= set()  
            
           for row, col in stones:  
               nodeRow \= row  
               nodeCol \= maxRow \+ col \+1   
               dsu.union(nodeRow, nodeCol)  
               unqNodes.add(nodeRow)  
               unqNodes.add(nodeCol)  
            
           count \= 0  
           for node in unqNodes:  
               if dsu.find(node) \== node:  
                   count \+= 1  
            
           return n \- count  
```
   The Time complexity is O(stones\*α(stones)).  
     
     
     
     
     
     
     
   **\>\> Kosaraju’s Strongly  Connected Components**  
```python
   from collections import deque  
   class Solution:  
       def kosaraju(self, adj):  
           #step 1: Do DFS and find first one that ends (sort of topo sort)  
           vis=set(),stk=deque()  
           def dfs(node):  
               vis.add(node)  
               for nei in adj[node]:  
                   if nei not in vis:  
                       dfs(nei)  
               stk.append(node)  
           V=len(adj)  
           for node in range(V):  
               if node not in vis:  
                   dfs(node)  
           #Now we have nodes in stack,The topmost finished the last  
           #Now Inversing the edges and going to do new DFS  
           adjI=[[] for _ in range(V)]  
           for u in range(V):  
               for v in adj[u]:  
                   adjI[v].append(u)  
           ans=0,visI=set()  
           comps=set()  
           def dfsI(node,temp):  
               visI.add(node)  
               temp.append(node)  
               for nei in adjI[node]:  
                   if nei not in visI:  
                       dfsI(nei,temp)  
           #The dfs order is the same as the stack order  
           while stk:  
               node=stk.pop()  
               if node not in visI:  
                   temp=[]  
                   dfsI(node,temp)  
                   ans+=1  
                   comps.add(tuple(temp))  
           #The number of time DFS is performed is our Answer  
           print(comps)  
           return ans  
```

 **18.Critical Connections**

	There are n servers numbered from 0 to n \- 1 connected by undirected server-to-server connections forming a network where connections\[i\] \= \[ai, bi\] represents a connection between servers ai and bi. Any server can reach other servers directly or indirectly through the network.  
A *critical connection* is a connection that, if removed, will make some servers unable to reach some other server.

Return all critical connections in the network in any order.

Direct Implementation of Tarjan’s Algo on bridges.
```python  
def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:  
        adj=[[] for _ in range(n)]  
        for u,v in connections:  
            adj[u].append(v)  
            adj[v].append(u)  
        vis=set()  
        tin=[10**6 for _ in range(n)]  
        low=[10**6 for _ in range(n)]  
        bridges=[]  
        timer=1  
        def dfs(node,parent):  
            nonlocal timer  
            vis.add(node)  
            tin[node],low[node]=timer,timer  
            timer+=1  
            for neigh in adj[node]:  
                if neigh!=parent:  
                    if neigh not in vis:  
                        dfs(neigh,node)  
                        low[node]=min(low[node],low[neigh])  
                        if low[neigh]>tin[node]:  
                            bridges.append([node,neigh])  
                    else:  
                        low[node]=min(low[node],low[neigh])  
        dfs(0,-1)  
        return bridges 
``` 
`tin[node]`: Time when the node is first visited.  
`low[node]`: Lowest discovery time reachable from the node (including back-edges).  
The time complexity is O(n + m) where n is the number of nodes and m is the number of edges.