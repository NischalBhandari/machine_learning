arr=[1,2,3,5,6,7,9,10]

for i in range(0,len(arr)-1):
	if(arr[i]+1 != arr[i+1]):
		print(arr[i]+1)