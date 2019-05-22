import numpy as np 
import matplotlib.pyplot as plt
planets=['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
AverageDistance=[57900000,108160000 ,149600000,	227936640 ,778369000,1427034000, 1427034000 , 4496976000 ]
Diameter=[4878,12104,12756,6794,142984,120536,51118,49532]
OrbitSize=[]
Frequency=[59,243,23.56,24.37,9.55,10.39,17.14,16.07]
frequency_decimal=[]
OrbitTime=[88,224,365.25,687,4331.865,10592.25,30681,60193.2]
inverseorbit=[]
Gravity=[0.38,0.9,1,0.38,2.64,1.16,1.11,1.21]
Moons=[0,0,1,2,79,62,27,14]
def inversing():
	for i in OrbitTime:
		
		i=1/i
		print(i)
		inverseorbit.append(i)
def findorbitshape():
	for i in Diameter:
		i=3.14*i 
		OrbitSize.append(i)
def FrequencyCalculator(arr):
	hours_container=[]
	minute_container=[]
	
	for i in Frequency:
		if(i>=0):
			hours=i//1
			minutes=((i-hours)*100)//1
			print(hours)
			minute_container.append(minutes)
			hours_container.append(hours)
			frequency_decimal.append(hours+(((minutes/60)*100)/100))

	print(hours_container)
	print(minute_container)
	print(frequency_decimal)
print(FrequencyCalculator(Frequency))
print(np.average(frequency_decimal))
print(inversing())
print(findorbitshape())
print(OrbitSize)

plt.plot(AverageDistance,inverseorbit)
plt.show()