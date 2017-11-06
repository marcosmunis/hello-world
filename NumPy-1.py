# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:09:14 2017

@author: Marcos Munis
         @MMunis
         marcosmunis@gmail.com


  ////////////////////////
 //  ESTUDOS DE NumPy  //
////////////////////////


"""

import numpy as np

from io import BytesIO, StringIO





x = [[ 1., 0., 0.],[ 0., 1., 2.]]
print( x )

print( '------------------------' )

a = np.arange(15)
print( a )
a = np.arange(15).reshape(5, 3)
print( a )
a = np.arange(15).reshape(3, 5)
print( a )
print( a.shape )
print( a.ndim )
print( a.dtype.name )
print( a.itemsize )
print( a.size )
print( type(a) )

print( '------------------------' )

b = np.array([6, 7, 8])
print( b )
print( type(b) )

print( '------------------------' )

a = np.array([2,3,4])
print( a )
print( a.dtype )
b = np.array([1.2, 3.5, 5.1])
print( b )
print( b.dtype )

print( '------------------------' )

b = np.array([ (1.5,2,3), (4,5,6) ])
print( b )

b = np.array([ [1,4,7], [2,5,8], [3,6,9] ])
print( b )

b = np.array([ (1,4,7), (2,5,8), (3,6,9) ])
print( b )

b = np.array([ (1,4,7), (2,5,8), (3,6,9) ], dtype=complex )
print( b )


print()
print( '------------------------' )
print()


print()
print("// zeros, ones, empty                ")
print("//-----------------------------------")
print()
'''
The function zeros creates an array full of zeros, the function 
ones creates an array full of ones, and the function empty creates 
an array whose initial content is random and depends on the state 
of the memory. By default, the dtype of the created array is float64.
'''

x = np.zeros( (3,10) )                   # 3 linhas, 4 colunas
print(x)
print()

x = np.ones( (2,3,4), dtype=np.int16 )  # 2 tabelas, 3 linhas, 4 colunas
print(x)
print()

x = np.empty( (2,4) )                   # uninitialized, output may vary
print(x)
print()

print()
print("// arange                            ")
print("//-----------------------------------")
print()

x = np.arange( 10, 30, 5 )  # init=10, end<30, 5++
print(x)
x = np.arange( 0, 2, 0.3 )  # it accepts float arguments
print(x)
x = np.arange( 0, 2, 0.1 )  # it accepts float arguments
print(x)
print()

print('Circunferencia')
x = np.arange( 0, 2*np.pi, np.pi/6 )  # it accepts float arguments
print( x )
f = np.sin(x)
print( f )


print()


print()
print("// linspace                          ")
print("//-----------------------------------")
print()

x = np.linspace( 0, 2, 5 )  # 5 numbers from 0 to 2 (inclusive)
print(x)
x = np.linspace( 0, 2, 9 )  # 9 numbers from 0 to 2 (inclusive)
print(x)
print()
print('Circunferencia')
x = np.linspace( 0, 2*np.pi, 13 )  # useful to evaluate function at lots of points
print( x )
f = np.sin(x)
print( f )
print()


print()
print("// arange                            ")
print("//-----------------------------------")
print()

a = np.arange(6)                         # 1d array
print(a)
b = np.arange(12).reshape(4,3)           # 2d array
print(b)
c = np.arange(24).reshape(2,3,4)         # 3d array
print(c)



print()
print("// operacoes com array               ")
print("//-----------------------------------")
print()

a = np.array( [20,30,40,50] )
print( a )
b = np.arange( 4 )
print( b )
c = a-b
print( c )
print( b**2 )
print( 10*np.sin(a) )
print( a<35 )







A = np.array( [[1,1],
               [0,1]] )

B = np.array( [[2,0],
               [3,4]] )

print( "A:" )
print( A )
print()

print( "B:" )
print(  B )
print()

print( "A*B (elementwise product):" ) 
print( A*B )

print( "A.dot(B) (matrix product):" ) 
print( A.dot(B) )               # matrix product

print( "dot(A,B) (matrix product):" ) 
print( np.dot(A, B) )           # another matrix product

print( "dot(B,A) (matrix product):" ) 
print( np.dot(B, A) )           # another matrix product





print() 
print( "// " ) 
print( "-------------------------------------------------" ) 

a = np.ones((2,3), dtype=int)
print( "a:" )
print( a )
print()

b = np.random.random((2,3))
print( "b:" )
print( b )
print()

a *= 3
print( "a *= 3" )
print( a )
print()

b += a
print( "b += a" )
print( b )
print()





a = np.ones(3, dtype=np.int32)
print( a )
print()

b = np.linspace(0,np.pi,3)
print( b )
print( b.dtype.name )
print()

c = a+b
print( c )
print( c.dtype.name )
print()

a = np.random.random((2,3))
print( a )
print()

print( "sum: {}".format( a.sum() ) )
print( "min: {}".format( a.min() ) )
print( "max: {}".format( a.max() ) )



print() 
print( "// b = np.arange(12).reshape(3,4) " ) 
print( "-------------------------------------------------" ) 

b = np.arange(12).reshape(3,4)
print( b )
print()

print( "     {}".format( "b.sum(axis=0)"    ) ) # sum of each column
print( "     {}".format(  b.sum(axis=0)     ) ) # sum of each column
print()
print( "     {}".format( "b.sum(axis=1)"    ) ) # sum of each column
print( "     {}".format(  b.sum(axis=1)     ) ) # sum of each column
print()
print( "     {}".format( "b.min(axis=1)"    ) ) # min of each row
print( "     {}".format(  b.min(axis=1)     ) ) # min of each row
print()




print() 
print( "// B = np.arange(3)                              " ) 
print( "-------------------------------------------------" ) 

B = np.arange(3)
print( B )
print( "     {}".format( np.exp(B)  ) )
print( "     {}".format( np.sqrt(B) ) )
C = np.array([2., -1., 4.])
print( C )

































