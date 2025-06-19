Report of Target 2025

# STRINGS 

**1.String to Integer (atoi)**  
Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.  
The algorithm for myAtoi(string s) is as follows:

1. Whitespace: Ignore any leading whitespace (" ").  
2. Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity if neither present.  
3. Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0\.  
4. Rounding: If the integer is out of the 32-bit signed integer range \[-231, 231 \- 1\], then round the integer to remain in the range. Specifically, integers less than \-231 should be rounded to \-231, and integers greater than 231 \- 1 should be rounded to 231 \- 1\.  
   Return the integer as the final result.

As the constraint is suggesting that i should first remove spaces. So i traverse from left to right to skip all the leading spaces. And then keeping track of the pointer where the first alphanum is coming.  
```cpp
	while(i<s.length() && s[i]==' ')  
            i++; 
``` 
Then checking weather the number is negative or not if yes then I am using flags to keep record.  
```cpp
    if(s[i]=='-')  
        {  
            sign=-1;  
            i++;  
        }  
        else if(s[i]=='+')  
            i++; 
```

Then just converting and rounding off to form the output number.  
```cpp
    while(i<s.length())  
        {  
            if(s[i]>='0' && s[i]<='9')  
            {  
                ans=ans*10+(s[i]-'0');  
                if(ans>INT_MAX && sign==-1)  
                    return INT_MIN;  
                else if(ans>INT_MAX && sign==1)  
                    return INT_MAX;  
                i++;  
            }  
            else  
                return ans*sign;  
        }  
        return (ans*sign);
```  
The time complexity of this implementation is O(n).

**2.Roman to Integer**  
	**Symbol       Value**  
**I             1**  
**V             5**  
**X             10**  
**L             50**  
**C             100**  
**D             500**  
**M             1000**  
   

So main constraint is **IV and IX** or any Symbol with leading **I** is exception so I am traversing from right to left then adding values in the output and tadaaa. Our desired output is here.
```cpp
int romanToInt(string s) {  
        int sum=0;  
        for(int i=s.size()-1;i>=0;i--){  
            if(s[i]=='I'){  
                sum+=1;  
            }  
            else if(s[i]=='V'){  
                if(i>0 && s[i-1]=='I'){  
                    sum+=4;  
                    i--;  
                }  
                else{  
                    sum+=5;  
                }  
            }  
            else if(s[i]=='X'){  
                if(i>0 && s[i-1]=='I'){  
                    sum+=9;  
                    i--;  
                }  
                else{  
                    sum+=10;  
                }  
            }  
            else if(s[i]=='L'){  
                if(i>0 && s[i-1]=='X'){  
                    sum+=40;  
                    i--;  
                }  
                else{  
                    sum+=50;  
                }  
            }  
            else if(s[i]=='C'){  
                if(i>0 && s[i-1]=='X'){  
                    sum+=90;  
                    i--;  
                }  
                else{  
                    sum+=100;  
                }  
            }  
            else if(s[i]=='D'){  
                if(i>0 && s[i-1]=='C'){  
                    sum+=400;  
                    i--;  
                }  
                else{  
                    sum+=500;  
                }  
            }  
            else if(s[i]=='M'){  
                if(i>0 && s[i-1]=='C'){  
                    sum+=900;  
                    i--;  
                }  
                else{  
                    sum+=1000;  
                }  
            }  
        }  
        return sum; }
```
**3\. Longest Common Prefix**  
	Write a function to find the longest common prefix string amongst an array of strings.  
If there is no common prefix, return an empty string "".

So I am sorting the string lexicographically and after sorting checking whether first and last string at the sorted array of string having what common prefix.
```cpp
string longestCommonPrefix(vector<string>& v) {  
        string ans="";  
        sort(v.begin(),v.end());  
        int n=v.size();  
        string first=v[0],last=v[n-1];  
        for(int i=0;i<min(first.size(),last.size());i++){  
            if(first[i]!=last[i]){  
                return ans;  
            }  
            ans+=first[i];  
        }  
        return ans;    
    }
```
Now its time complexity is O(nlogn).   
Now another good approach of this problem is to do it using binary search. 
```cpp 
low = 0 ; high= min(sizeof(allStrings))
```
and  checker function is checking whether the length obtained in mid is a candidate answer or not. 
```cpp
    bool isCommonPrefix(vector<string>& strs, int len) {  
            string str1 = strs[0].substr(0, len);  
            for (int i = 1; i < strs.size(); i++)  
                if (strs[i].find(str1) != 0) return false;  
            return true;  
        }

 string longestCommonPrefix(vector<string>& strs) {  
        if (strs.empty()) return "";  
        int minLen = INT_MAX;  
        for (string str : strs) minLen = min(minLen, (int)str.length());  
        int low = 1;  
        int high = minLen;  
        while (low <= high) {  
            int middle = (low + high) / 2;  
            if (isCommonPrefix(strs, middle))  
                low = middle + 1;  
            else  
                high = middle - 1;  
        }  
        return strs[0].substr(0, (low + high) / 2);  
    }
```
The time complexity is O(Slogm) where S is the sum of all characters in all strings. The algorithm makes logm iterations.

**4\. Find the Index of the First Occurrence in a String**  
	Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or \-1 if needle is not part of haystack.	

Naive approach is matching character wise and if we don't find match we are just restarting the indexing and again matching. 
```cpp 
int strStr(string haystack, string needle) {  
        int hLen = haystack.length();  
        int nLen = needle.length();  
        int nIndex = 0;  
        for(int i=0; i<hLen; i++){  
            // as long as the characters are equal, increment needleIndex  
            if(haystack[i]==needle[nIndex]){  
                nIndex++;  
            }  
            else{  
                // start from the next index of previous start index  
                i=i-nIndex;  
                // needle should start from index 0  
                nIndex=0;  
            }  
            // check if needleIndex reached needle length  
            if(nIndex==nLen){  
                // return the first index  
                return i-nLen+1;  
            }  
        }  
        return -1;  
    }
```
Now coming to the main KMP Algo the Knuth-Morris-Pratt Algorithm. This algorithm calculates the longest suffix and prefix index matching point of chars within a string and stores in the LPS array.  
Then what I was doing in naive was to came back but now this will be done using this LPS array to come reverse and to find the matching point.
```cpp
	vector<int> computeLPS(const string& pattern) {  
        int n = pattern.size();  
        vector<int> lps(n, 0);  
        int len = 0, i = 1;

        while (i < n) {  
            if (pattern[i] == pattern[len]) {  
                len++;  
                lps[i] = len;  
                i++;  
            } else {  
                if (len != 0) {  
                    len = lps[len - 1];  
                } else {  
                    lps[i] = 0;  
                    i++;  
                }  
            }  
        }

        return lps;  
    } 
``` 
This function creates the LPS  array.  Now I am going to use this array for pattern matching.

Here LPS\[i\] is calculated size of the suffix and prefix that is present  in the needle from 0 to i.
```cpp
vector<int> lps = computeLPS(needle);  
        int i = 0, j = 0;

        while (i < haystack.size()) {  
            if (haystack[i] == needle[j]) {  
                i++;  
                j++;  
            }

            if (j == needle.size()) {  
                return i - j;  
            }  
            if (i < haystack.size() && haystack[i] != needle[j]) {  
                if (j != 0) {  
                    j = lps[j - 1];  
                } else {  
                    i++;  
                }  
            }  
        }  
        return -1;  
```
Now this last part, if the haystack char and needle char are not matched then I am going back to that index of needle where the last match of suffix and prefix was found and which is stored in LPS\[j-1\].

Now the time complexity reduces to O(m+n). with linear increase in space complexity.

**5\. Count and Say**  
	The **count-and-say** sequence is a sequence of digit strings defined by the recursive formula:

* countAndSay(1) \= "1"  
* countAndSay(n) is the run-length encoding of countAndSay(n \- 1).

Given a positive integer n, return *the* nth *element of the **count-and-say** sequence*.

Creating the RLE() that will return Run Length Encoding using map and taking count of frequencies.  

```cpp
string RLE(string s){  
        unordered_map<char,int>mp;  
        string ans="";  
        mp[s[0]]=1;  
        for(int i=1;i<s.size();i++){  
            if(s[i]!=s[i-1]){  
                ans+=to_string(mp[s[i-1]]);  
                mp[s[i-1]]=0;  
                mp[s[i]]=1;  
                ans.push_back(s[i-1]);  
            }  
            else{  
                mp[s[i]]++;  
            }  
        }  
        for(auto it:mp){  
            if(it.second>0){  
                ans+=to_string(it.second);  
                ans.push_back(it.first);  
            }  
        }  
        return ans;  
    } 
```
Now as the question states that it depends on the previous value so I am going to use basic iterative DP.
```cpp
string countAndSay(int n) {  
        vector<string>dp(n+1);  
        dp[0]="";  
        dp[1]="1";  
        for(int i=2;i<n+1;i++){  
            dp[i]=RLE(dp[i-1]);  
        }  
        return dp[n];  
    }
```
The worst case time complexity of the code is O(n2).

**6.Repeated String Matching**

Given two strings a and b, return *the minimum number of times you should repeat string* a *so that string* b *is a substring of it*. If it is impossible for b​​​​​​ to be a substring of a after repeating it, return \-1.

So the first condition that I found is that the size of string a has to be greater than b for pattern matching. Secondly if the string size becomes greater after n iterations then we have to just consider nth and n+1th iteration as the pattern starts repeating after that.  
So we have to find the minimum of both iterations and return it otherwise \-1.
```cpp
int repeatedStringMatch(string a, string b) {  
       int m=a.size();  
       int n=b.size();  
       string heystack=a;  
       int ans=1;  
       while(m<n){  
           heystack+=a;  
           m=heystack.size();  
           ans++;  
       }  
       int ind = heystack.find(b);  
       if(ind != string::npos) return ans;  
       else {  
           heystack+=a;  
           ind=heystack.find(b);  
           if(ind!=string::npos) return ans+1;  
       }  
       return -1;  
        
   }
```  
Now in place of .find() I can use the KMP matching algo and obtain the same result. So the time complexity of the problem is same as the Pattern matching problem Time complexity.