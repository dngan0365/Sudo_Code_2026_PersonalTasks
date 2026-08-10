from typing import List

# Approach 1: Hash Map
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)
        counter = Counter(nums)
        i = 0
        for num in counter:
            nums[i] = num
            counter[num] -= 1
            i += 1
            if counter[num] >= 1:
                nums[i] = num
                i += 1
        return i    

# Approach 2: Two Pointers
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        track = 0
        for curr in range(1, len(nums)):
            if nums[curr] != nums[track]:
                k = 1
                nums[track + 1] = nums[curr]
                track += 1
            elif k <= 1:
                k += 1
                nums[track + 1] = nums[curr]
                track += 1
        return track + 1

# Approach 3: Two Pointer (Optimal)
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 0
        for r in range(len(nums)):
            if l < 2 or nums[r] != nums[l-2]:
                nums[l] = nums[r]
                l += 1
        return l