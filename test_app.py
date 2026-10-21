import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_no_future_feature(self):
        x,y=design(np.arange(30));self.assertEqual(x[0,13],13);self.assertEqual(y[0],14)
    def test_linear_fit(self):
        x=np.arange(50.).reshape(-1,1);y=x[:,0]*3+2;w,b=fit(x,y,.0001);self.assertLess(rmse(y,x@w+b),.001)
    def test_seasonal_index(self):
        x,_=design(np.arange(30));self.assertEqual(x[0,-9],7)
