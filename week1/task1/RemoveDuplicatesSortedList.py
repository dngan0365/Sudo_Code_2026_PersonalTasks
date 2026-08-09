from typing import List

# Aproach 1: Remove duplicate elements
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        curr = 1
        while curr < len(nums):
            if nums[curr] == nums[curr - 1]:
                nums.pop(curr)
            else:
                curr += 1
        return len(nums)
    
# Approach 2: Sorted Set
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        unique = sorted(set(nums))
        nums[:len(unique)] = unique
        return len(unique)
    
# Aproach 3: two pointers
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        track = 0
        curr = 1
        while curr < len(nums):
            if nums[curr] != nums[track]: # if the current value is not the equal to the previous value in unique sorted list
                track += 1
                nums[track] = nums[curr] #  Add the curr value to the list
                curr += 1 # Plus the curr with 1 for check next value
            else:
                curr += 1
        return track + 1