def quicksort(arr):
	zeroth=0
	last=len(arr)
	if len(arr)>1:
		mid=(zeroth+last)//2
		pivot=arr[mid]
		left=[x for x in arr if x<pivot]
		middle=[x for x in arr if x==pivot]
		right=[x for x in arr if x>pivot]
		print("left",left)
		print("right",right)
		print("middle",middle)
		return quicksort(left)+middle+quicksort(right)
	else:
		print(arr)
		return arr

arr=[6,5,4,3,2,1]
print(quicksort(arr))