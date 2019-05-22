arr=[1,2,2,2,2,4,4,24]
def duplicates(arr):
	for i in range(0,len(arr)-1):
		for j in range(i+1,len(arr)):
			if arr[i] == arr[j]:
				print(arr[i],"is duplicate")
def multipledup(arr):

	for i in range(0,len(arr)-1):
		count=0
		for j in range(i+1,len(arr)):
			if arr[i] == arr[j]:
				count+=1
		print("there no of %d is %d"%(arr[i],count+1))

def recmultiple(arr):
	if len(arr)>1:
		left=0
		right=len(arr)
		mid=(0+right)//2
		pivot=arr[mid]
		test = [x for x in arr if x==pivot]
		less=[x for x in arr if x<pivot]
		more = [x for x in arr if x > pivot]
		print(test)
		listing.append(test)
		return recmultiple(less)+recmultiple(more)
	else:
		return arr
	
listing=[]	
recmultiple(arr)
print(listing)
