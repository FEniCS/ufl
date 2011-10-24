#include <tr1/cmath>
#include <iostream>

using std::tr1::cyl_bessel_j;
using std::tr1::cyl_neumann;
using std::tr1::cyl_bessel_i;
using std::tr1::cyl_bessel_k;

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

