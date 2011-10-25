#include <iostream>

#include <tr1/cmath>
using std::tr1::cyl_bessel_i;
using std::tr1::cyl_bessel_j;
using std::tr1::cyl_bessel_k;
using std::tr1::cyl_neumann;

//#include <boost/math/tr1.hpp>
//using boost::math::tr1::cyl_bessel_i;
//using boost::math::tr1::cyl_bessel_j;
//using boost::math::tr1::cyl_bessel_k;
//using boost::math::tr1::cyl_neumann;
// Compile with '$ g++ test_bessel.cpp -lboost_math_tr1'

//#include <boost/math/special_functions/bessel.hpp>
//using boost::math::cyl_bessel_i;
//using boost::math::cyl_bessel_j;
//using boost::math::cyl_bessel_k;
//using boost::math::cyl_neumann;

int main(int argc, char * argv[])
{
    double nu = 0.0;
    double x = 1.23;
    double i = cyl_bessel_i(nu, x);
    double j = cyl_bessel_j(nu, x);
    double k = cyl_bessel_k(nu, x);
    double y = cyl_neumann(nu, x);
    std::cout << i << std::endl;
    std::cout << j << std::endl;
    std::cout << k << std::endl;
    std::cout << y << std::endl;
    return 0;
}

