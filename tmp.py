from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        i = [0] * len(heights)
        best = 0
        while any([h > 0 for h in heights]):
            sum_rec = 0
            minh = min([h for h in heights if h > 0])
            heights = [h - minh for h in heights]
            i = [ii + minh for ii in i]
            for idx, x in enumerate(i):
                if heights[idx] < 0:
                    best = max(best, sum_rec)
                    sum_rec = 0
                else:
                    sum_rec += x
            best = max(best, sum_rec)

        return best


Solution().largestRectangleArea([10, 4])
