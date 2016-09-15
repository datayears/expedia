from average_precision import apk,mapk

# this code test the accuracy of the MAP scoring function
# 

actual = [1]

predicted = [1,2,3,4,5]

print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [2,1,3,4,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [3,2,1,4,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [4,2,3,1,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

predicted = [4,2,3,5,1]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',apk(actual,predicted,5) )

print mapk([[1],[1],[1],[1],[1]],[[1,2,3,4,5],[2,1,3,4,5],[3,2,1,4,5],[4,2,3,1,5],[4,2,3,5,1]], 5)