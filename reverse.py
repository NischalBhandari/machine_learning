arr1=[1,2,3,4,5]
arr2=[x for x in range(len(arr1))]
place=len(arr1)-1
for i in range(len(arr1)):
		print(place)
		arr2[i]=arr1[place]
		place-=1
print(arr2)